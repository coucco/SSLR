import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(dims, activation=nn.GELU, norm=True):
    """Строит MLP по списку размерностей"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            if norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(activation())
    return nn.Sequential(*layers)

class ComponentEncoder(nn.Module):
    """Кодирует вход в Gaussian (mu, logvar)"""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden // 2])
        self.mu_head = nn.Linear(hidden // 2, out_dim)
        self.lv_head = nn.Linear(hidden // 2, out_dim)
    
    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h).clamp(-4, 4)

class StructuredController(nn.Module):
    def __init__(self, cfg, roll_flat_dim, t_frames):
        super().__init__()
        H = cfg.hidden_dim
        MVZ = cfg.musicvae_z_dim
        zs = cfg.z_struct_dim
        ze = cfg.z_energy_dim
        zr = cfg.z_rhythm_dim
        zh = cfg.z_harmony_dim
        zm = cfg.z_melody_dim
        role_dim = t_frames * 128
        
        # Encoders
        self.struct_enc = ComponentEncoder(roll_flat_dim, H, zs)
        self.energy_enc = ComponentEncoder(roll_flat_dim, H, ze)
        self.rhythm_enc = ComponentEncoder(role_dim, H // 2, zr)
        self.harmony_enc = ComponentEncoder(role_dim, H // 2, zh)
        self.melody_enc = ComponentEncoder(role_dim, H // 2, zm)
        
        # Decoder
        total_z = zs + ze + zr + zh + zm
        self.decoder = mlp([total_z, H, H, H, MVZ])
        
        # Auxiliary heads
        self.energy_head = nn.Sequential(
            mlp([ze, H // 4, t_frames], norm=False), 
            nn.Sigmoid()
        )
        self.rhythm_head = mlp([zr + zs, H // 2, role_dim], norm=False)
        self.harmony_head = mlp([zh + zs, H // 2, role_dim], norm=False)
        self.melody_head = mlp([zm + zs, H // 2, role_dim], norm=False)
        
        self.dims = dict(struct=zs, energy=ze, rhythm=zr, harmony=zh, melody=zm)
        self.cfg = cfg
        self.t = t_frames
    
    def reparam(self, mu, lv):
        if self.training:
            return mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return mu
    
    def encode(self, x_flat):
        B = x_flat.size(0)
        T = self.t
        x3d = x_flat.view(B, T, 128, 3)
        xr = x3d[:, :, :, 0].reshape(B, -1)
        xh = x3d[:, :, :, 1].reshape(B, -1)
        xm = x3d[:, :, :, 2].reshape(B, -1)
        
        ms, ls = self.struct_enc(x_flat)
        me, le = self.energy_enc(x_flat)
        mr, lr = self.rhythm_enc(xr)
        mh, lh = self.harmony_enc(xh)
        mm, lm = self.melody_enc(xm)
        
        return dict(
            mu=dict(struct=ms, energy=me, rhythm=mr, harmony=mh, melody=mm),
            lv=dict(struct=ls, energy=le, rhythm=lr, harmony=lh, melody=lm),
        )
    
    def sample_z(self, enc_out):
        return {k: self.reparam(enc_out['mu'][k], enc_out['lv'][k]) 
                for k in enc_out['mu']}
    
    def decode_to_musicvae_z(self, z_parts):
        z_cat = torch.cat([z_parts['struct'], z_parts['energy'],
                          z_parts['rhythm'], z_parts['harmony'], 
                          z_parts['melody']], dim=1)
        return self.decoder(z_cat)
    
    def forward(self, x_flat):
        enc = self.encode(x_flat)
        z = self.sample_z(enc)
        z_mv_hat = self.decode_to_musicvae_z(z)
        
        e_pred = self.energy_head(z['energy'])
        r_pred = self.rhythm_head(torch.cat([z['rhythm'], z['struct']], 1))
        h_pred = self.harmony_head(torch.cat([z['harmony'], z['struct']], 1))
        m_pred = self.melody_head(torch.cat([z['melody'], z['struct']], 1))
        
        return dict(z_mv_hat=z_mv_hat, z_parts=z, enc=enc,
                   e_pred=e_pred, r_pred=r_pred, h_pred=h_pred, m_pred=m_pred)
    
    def sample_structured_z(self, n, device):
        """Сэмплирует из prior N(0,I)"""
        return {k: torch.randn(n, d, device=device) 
                for k, d in self.dims.items()}
    
    def swap(self, z_a, z_b, component):
        """Меняет один компонент"""
        z_new = {k: v.clone() for k, v in z_a.items()}
        z_new[component] = z_b[component].clone()
        return z_new
    
    def interpolate(self, z_a, z_b, alpha):
        """Интерполяция между двумя z"""
        return {k: (1 - alpha) * z_a[k] + alpha * z_b[k] for k in z_a}