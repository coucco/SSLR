import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def tc_penalty(z_parts):
    """Total Correlation penalty"""
    z_cat = torch.cat(list(z_parts.values()), dim=1)
    z_norm = (z_cat - z_cat.mean(0)) / (z_cat.std(0) + 1e-8)
    B = z_cat.size(0)
    corr = (z_norm.T @ z_norm) / B
    D = corr.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=corr.device)
    return corr[mask].abs().mean()

def kl_div(enc):
    """KL divergence для всех компонентов"""
    kl = 0.0
    for k in enc['mu']:
        mu, lv = enc['mu'][k], enc['lv'][k]
        kl += -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1).mean()
    return kl

def total_loss(out, z_mv_target, x_flat, energy, cfg, beta):
    """Вычисление total loss"""
    B, T = x_flat.size(0), energy.size(1)
    
    # Recon loss
    recon_loss = F.mse_loss(out['z_mv_hat'], z_mv_target)
    
    # KL loss
    kl = kl_div(out['enc'])
    
    # Energy loss
    energy_loss = F.mse_loss(out['e_pred'], energy)
    
    # Role losses
    x3d = x_flat.view(B, T, 128, 3)
    xr = x3d[:, :, :, 0].reshape(B, -1)
    xh = x3d[:, :, :, 1].reshape(B, -1)
    xm = x3d[:, :, :, 2].reshape(B, -1)
    role_loss = (F.binary_cross_entropy_with_logits(out['r_pred'], xr) +
                F.binary_cross_entropy_with_logits(out['h_pred'], xh) +
                F.binary_cross_entropy_with_logits(out['m_pred'], xm)) / 3.0
    
    # Disentanglement loss
    dis_loss = tc_penalty(out['z_parts'])
    
    total = (cfg.lambda_recon * recon_loss + beta * kl +
            cfg.lambda_energy * energy_loss + cfg.lambda_role * role_loss +
            cfg.lambda_dis * dis_loss)
    
    parts = dict(recon=recon_loss.item(), kl=kl.item(),
                energy=energy_loss.item(), role=role_loss.item(),
                dis=dis_loss.item(), total=total.item())
    return total, parts

def beta_schedule(epoch, cfg):
    """Schedule for KL weight"""
    return min(cfg.lambda_kl, cfg.lambda_kl * epoch / cfg.kl_warmup)

def train_epoch(model, loader, optimizer, cfg, beta, device):
    """Одна эпоха обучения"""
    model.train()
    sums = {k: 0.0 for k in ['total','recon','kl','energy','role','dis']}
    
    for z_mv, x_flat, energy in loader:
        z_mv = z_mv.to(device)
        x_flat = x_flat.to(device)
        energy = energy.to(device)
        
        out = model(x_flat)
        loss, parts = total_loss(out, z_mv, x_flat, energy, cfg, beta)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        for k, v in parts.items():
            sums[k] += v
    
    return {k: v / len(loader) for k, v in sums.items()}

def validate_epoch(model, loader, cfg, beta, device):
    """Валидация"""
    model.eval()
    sums = {k: 0.0 for k in ['total','recon','kl','energy','role','dis']}
    
    with torch.no_grad():
        for z_mv, x_flat, energy in loader:
            z_mv = z_mv.to(device)
            x_flat = x_flat.to(device)
            energy = energy.to(device)
            
            out = model(x_flat)
            _, parts = total_loss(out, z_mv, x_flat, energy, cfg, beta)
            
            for k, v in parts.items():
                sums[k] += v
    
    return {k: v / len(loader) for k, v in sums.items()}

def train(model, train_loader, val_loader, cfg, device):
    """Основной цикл обучения"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs, eta_min=1e-5)
    
    history = {k: [] for k in ['train_total','val_total','train_recon','val_recon']}
    best_val = float('inf')
    
    print("Starting training...")
    for epoch in range(1, cfg.n_epochs + 1):
        beta = beta_schedule(epoch, cfg)
        t0 = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, cfg, beta, device)
        val_metrics = validate_epoch(model, val_loader, cfg, beta, device)
        scheduler.step()
        
        # Сохраняем историю
        history['train_total'].append(train_metrics['total'])
        history['val_total'].append(val_metrics['total'])
        history['train_recon'].append(train_metrics['recon'])
        history['val_recon'].append(val_metrics['recon'])
        
        # Сохраняем лучшую модель
        if val_metrics['total'] < best_val:
            best_val = val_metrics['total']
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'best_val': best_val,
                'cfg': cfg.__dict__
            }
            torch.save(checkpoint, cfg.checkpoint_dir / 'best.pt')
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}/{cfg.n_epochs} β={beta:.3f} | "
                  f"tr: {train_metrics['total']:.4f} rec={train_metrics['recon']:.4f} | "
                  f"val: {val_metrics['total']:.4f} | {time.time()-t0:.1f}s")
    
    # Сохраняем историю
    with open(cfg.checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    
    print(f"Best val loss: {best_val:.4f}")
    return history