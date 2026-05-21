import magenta
print('Magenta version:', magenta.__version__)

import json, time, random, glob
from pathlib import Path

import numpy as np
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pretty_midi
import note_seq

# TensorFlow (нужен для Music VAE)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Пути (локальные)
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data' / 'musicvae_data'
CKPT_DIR = PROJECT_ROOT / 'data' / 'checkpoints' / 'structured_controller'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
MIDI_DIR = Path('/home/ubuntu/SSLR/data/raw_midi')

for d in [DATA_DIR, CKPT_DIR, OUTPUT_DIR, MIDI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Гиперпараметры
CFG = {
    'musicvae_z_dim': 512,
    'musicvae_model': 'cat-mel_2bar_big',
    
    # УВЕЛИЧИВАЕМ размерности компонентов
    'z_struct_dim': 128,    # было 64
    'z_energy_dim': 128,     # было 32
    'z_rhythm_dim': 256,    # было 64 ← важно!
    'z_harmony_dim': 256,   # было 64 ← важно!
    'z_melody_dim': 256,    # было 64 ← важно!
    
    'hidden_dim': 256,      # оставляем
    
    # Балансируем веса
    'lambda_recon': 1.0,
    'lambda_kl': 0.005,     # чуть меньше
    'lambda_energy': 3.0,   # УМЕНЬШАЕМ с 2.0 до 1.0
    'lambda_role': 2.5,     # УВЕЛИЧИВАЕМ для ролей
    'lambda_dis': 0.1,
    'kl_warmup': 30,
    
    'batch_size': 256,
    'lr': 1e-4,
    'n_epochs': 300,         # Больше эпох
    
    'seed': 42,
    'max_midi_files': 10000,  # Больше данных!
    'num_workers': 4,
}

CFG['z_total'] = sum([CFG['z_struct_dim'], CFG['z_energy_dim'], 
                      CFG['z_rhythm_dim'], CFG['z_harmony_dim'], CFG['z_melody_dim']])
print(f"Structured z total dim: {CFG['z_total']}")

random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])


# ============================================================
# ФИЛЬТРАЦИЯ MIDI ПО РОЛЯМ
# ============================================================

def count_roles_in_midi(midi_path: str) -> int:
    """
    Подсчитывает количество ролей в MIDI файле:
    - rhythm (drums)
    - harmony (bass или низкие ноты)
    - melody (высокие ноты)
    Возвращает количество ролей (0-3)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return 0

    has_drums = False
    has_bass = False
    has_melody = False

    for inst in pm.instruments:
        if inst.is_drum and len(inst.notes) > 0:
            has_drums = True
        elif inst.program in range(32, 40) and len(inst.notes) > 0:  # Bass instruments
            has_bass = True
        else:
            # Для остальных инструментов проверяем высоту нот
            if len(inst.notes) > 0:
                avg_pitch = np.mean([n.pitch for n in inst.notes])
                if avg_pitch < 55:  # Низкие ноты → гармония
                    has_bass = True
                else:  # Высокие ноты → мелодия
                    has_melody = True

    return sum([has_drums, has_bass, has_melody])


def filter_midi_by_roles(midi_dir: Path, min_roles: int = 2, max_files: int = 5000):
    """
    Фильтрует MIDI файлы, оставляя только те, у которых >= min_roles ролей
    """
    import shutil

    midi_files = list(midi_dir.rglob('*.mid')) + list(midi_dir.rglob('*.midi'))
    print(f"Total MIDI files found: {len(midi_files)}")

    # Создаём папку для отфильтрованных файлов
    filtered_dir = midi_dir.parent / f'filtered_{min_roles}_roles'
    filtered_dir.mkdir(parents=True, exist_ok=True)

    roles_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    filtered_files = []

    print(f"Filtering MIDI files (min_roles={min_roles})...")
    for f in tqdm(midi_files, desc='Checking roles'):
        role_count = count_roles_in_midi(str(f))
        roles_distribution[role_count] += 1
        if role_count >= min_roles:
            filtered_files.append(f)
            # Копируем в отфильтрованную папку
            shutil.copy(f, filtered_dir / f.name)

    print(f"\nRoles distribution:")
    print(f"  0 roles: {roles_distribution[0]} files")
    print(f"  1 role:  {roles_distribution[1]} files")
    print(f"  2 roles: {roles_distribution[2]} files")
    print(f"  3 roles: {roles_distribution[3]} files")
    print(f"\n✅ Kept {len(filtered_files)} files with >= {min_roles} roles")
    print(f"📁 Saved to: {filtered_dir}")

    return filtered_dir, filtered_files[:max_files]

# ============================================================
# ЗАГРУЗКА MUSIC VAE
# ============================================================

from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

MODEL_NAME = CFG['musicvae_model']
MUSICVAE_DIR = PROJECT_ROOT / 'data' / 'checkpoints' / 'music_vae'

ckpt_files = glob.glob(str(MUSICVAE_DIR / f"{MODEL_NAME}.ckpt.index"))
if ckpt_files:
    checkpoint_path = ckpt_files[0].replace('.index', '')
else:
    checkpoint_path = str(MUSICVAE_DIR)

print(f'Loading Music VAE from: {checkpoint_path}')
musicvae_config = configs.CONFIG_MAP[MODEL_NAME]
musicvae = TrainedModel(
    config=musicvae_config,
    batch_size=CFG['batch_size'],
    checkpoint_dir_or_path=checkpoint_path,
)
print('Music VAE loaded successfully.')

# ============================================================
# ФУНКЦИИ ДЛЯ MIDI
# ============================================================

def extract_piano_roll(midi_path: str, fs: int = 8, clip_frames: int = 256) -> np.ndarray:
    """
    Извлекает piano-roll с РАЗНЫМИ каналами для разных ролей:
    - Канал 0: ритм (ударные)
    - Канал 1: гармония (бас и низкие ноты)
    - Канал 2: мелодия (высокие ноты)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return np.zeros((clip_frames, 128, 3), dtype=np.float32)
    
    # Разделяем ноты по ролям
    rhythm_notes = []   # ударные
    harmony_notes = []  # бас и низкие ноты
    melody_notes = []   # высокие ноты
    
    for inst in pm.instruments:
        if inst.is_drum:
            rhythm_notes.extend(inst.notes)
        elif inst.program in range(32, 40):  # Bass instruments (Electric Bass, Acoustic Bass, etc.)
            harmony_notes.extend(inst.notes)
        else:
            # Сортируем по высоте нот
            for note in inst.notes:
                if note.pitch < 55:  # Низкие ноты (ниже E4) → гармония
                    harmony_notes.append(note)
                else:  # Высокие ноты → мелодия
                    melody_notes.append(note)
    
    # Создаём три разных piano-roll
    rolls = {}
    for role_name, notes in [('rhythm', rhythm_notes), ('harmony', harmony_notes), ('melody', melody_notes)]:
        tmp = pretty_midi.PrettyMIDI()
        if notes:
            inst = pretty_midi.Instrument(program=0, is_drum=(role_name == 'rhythm'))
            inst.notes = notes
            tmp.instruments.append(inst)
        
        try:
            roll = tmp.get_piano_roll(fs=fs)
        except Exception:
            roll = np.zeros((128, clip_frames))
        
        # Приводим к нужной длине
        if roll.shape[1] < clip_frames:
            pad = np.zeros((128, clip_frames - roll.shape[1]))
            roll = np.hstack([roll, pad])
        else:
            roll = roll[:, :clip_frames]
        
        rolls[role_name] = (roll > 0).astype(np.float32)
    
    # Собираем в стек (T, 128, 3) - три РАЗНЫХ канала!
    stack = np.stack([rolls['rhythm'], rolls['harmony'], rolls['melody']], axis=-1).transpose(1, 0, 2)
    
    return stack.astype(np.float32)

def compute_energy(roll: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    density = roll.sum(axis=(1, 2)) / (128 * 3)
    return (alpha * density + (1 - alpha) * density).astype(np.float32)

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

# ============================================================
# ПОДГОТОВКА ДАННЫХ С ФИЛЬТРАЦИЕЙ
# ============================================================

Z_CACHE = DATA_DIR / 'musicvae_z_vectors.npy'
ROLL_CACHE = DATA_DIR / 'piano_rolls.npy'
ENERGY_CACHE = DATA_DIR / 'energy_profiles.npy'

if Z_CACHE.exists() and ROLL_CACHE.exists():
    print('Loading cached data...')
    z_musicvae_all = np.load(Z_CACHE)
    rolls_all = np.load(ROLL_CACHE)
    energy_all = np.load(ENERGY_CACHE)
    print(f'Loaded: {len(z_musicvae_all)} samples')
else:
    # Используем отфильтрованную папку если она есть, иначе исходную
    filtered_dir = MIDI_DIR.parent / 'filtered_2_roles'
    if filtered_dir.exists() and len(list(filtered_dir.glob('*.mid'))) > 0:
        midi_dir = filtered_dir
        print(f"Using filtered MIDI directory: {midi_dir}")
    else:
        midi_dir = MIDI_DIR
        print(f"Using original MIDI directory: {midi_dir}")
        print("Consider filtering first: python filter_midi.py")
    
    # Поиск всех MIDI файлов
    midi_files = list(midi_dir.glob('*.mid')) + list(midi_dir.glob('*.midi'))
    random.shuffle(midi_files)
    midi_files = midi_files[:CFG['max_midi_files']]
    
    print(f'Processing {len(midi_files)} MIDI files...')
    
    # Проверяем роли для статистики
    roles_count = []
    for f in tqdm(midi_files, desc='Checking roles'):
        rc = count_roles_in_midi(str(f))
        roles_count.append(rc)
    
    print(f"Roles in selected files: 0:{roles_count.count(0)}, 1:{roles_count.count(1)}, 2:{roles_count.count(2)}, 3:{roles_count.count(3)}")
    
    z_list, roll_list, energy_list = [], [], []
    
    for midi_path in tqdm(midi_files, desc='Encoding MIDI'):
        roll = extract_piano_roll(str(midi_path))
        
        try:
            ns = note_seq.midi_file_to_note_sequence(str(midi_path))
            z_batch = musicvae.encode([ns])
            if isinstance(z_batch, list):
                z = z_batch[0]
            else:
                z = z_batch[0]
        except Exception as e:
            z = np.random.randn(512).astype(np.float32)
        
        z_list.append(z)
        roll_list.append(roll)
        energy_list.append(compute_energy(roll))
    
    print(f'Processed: {len(z_list)} files')
    
    z_musicvae_all = np.stack(z_list, axis=0).astype(np.float32)
    rolls_all = np.stack(roll_list, axis=0).astype(np.float32)
    energy_all = np.stack(energy_list, axis=0).astype(np.float32)
    
    np.save(Z_CACHE, z_musicvae_all)
    np.save(ROLL_CACHE, rolls_all)
    np.save(ENERGY_CACHE, energy_all)
    print(f'Saved to {DATA_DIR}')

print(f'\nDataset shapes:')
print(f'  z_musicvae: {z_musicvae_all.shape}   (N, 512)')
print(f'  rolls:      {rolls_all.shape}   (N, T, 128, 3)')
print(f'  energy:     {energy_all.shape}  (N, T)')

# ============================================================
# ДАТАСЕТ И ДАТАЛОАДЕР
# ============================================================

class ZDataset(Dataset):
    def __init__(self, z_mv, rolls, energy):
        self.z_mv = torch.from_numpy(z_mv)
        self.x_flat = torch.from_numpy(rolls.reshape(len(rolls), -1))
        self.energy = torch.from_numpy(energy)
    def __len__(self):
        return len(self.z_mv)
    def __getitem__(self, idx):
        return self.z_mv[idx], self.x_flat[idx], self.energy[idx]

N = len(z_musicvae_all)
idx = np.random.permutation(N)
n_train = int(0.85 * N)
n_val = int(0.10 * N)

train_ds = ZDataset(z_musicvae_all[idx[:n_train]], rolls_all[idx[:n_train]], energy_all[idx[:n_train]])
val_ds = ZDataset(z_musicvae_all[idx[n_train:n_train + n_val]], rolls_all[idx[n_train:n_train + n_val]], energy_all[idx[n_train:n_train + n_val]])
test_ds = ZDataset(z_musicvae_all[idx[n_train + n_val:]], rolls_all[idx[n_train + n_val:]], energy_all[idx[n_train + n_val:]])

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False)

ROLL_FLAT_DIM = rolls_all.shape[1] * 128 * 3
T_FRAMES = rolls_all.shape[1]

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
print(f'Roll flat dim: {ROLL_FLAT_DIM}')

# ============================================================
# МОДЕЛЬ
# ============================================================

def mlp(dims, activation=nn.GELU, norm=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            if norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(activation())
    return nn.Sequential(*layers)

class ComponentEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden // 2])
        self.mu_head = nn.Linear(hidden // 2, out_dim)
        self.lv_head = nn.Linear(hidden // 2, out_dim)
    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        lv = self.lv_head(h).clamp(-5, 2)  # ← Ограничиваем logvar!
        return mu, lv

class StructuredController(nn.Module):
    def __init__(self, cfg, roll_flat_dim, t_frames):
        super().__init__()
        H = cfg['hidden_dim']
        MVZ = cfg['musicvae_z_dim']
        zs = cfg['z_struct_dim']
        ze = cfg['z_energy_dim']
        zr = cfg['z_rhythm_dim']
        zh = cfg['z_harmony_dim']
        zm = cfg['z_melody_dim']
        role_dim = t_frames * 128
        
        self.struct_enc = ComponentEncoder(roll_flat_dim, H, zs)
        self.energy_enc = ComponentEncoder(roll_flat_dim, H, ze)
        self.rhythm_enc = ComponentEncoder(role_dim, H // 2, zr)
        self.harmony_enc = ComponentEncoder(role_dim, H // 2, zh)
        self.melody_enc = ComponentEncoder(role_dim, H // 2, zm)
        
        total_z = zs + ze + zr + zh + zm
        self.decoder = mlp([total_z, H, H, H, MVZ])
        
        self.energy_head = nn.Sequential(mlp([ze, H // 4, t_frames], norm=False), nn.Sigmoid())
        self.rhythm_head = mlp([zr + zs, H // 2, role_dim], norm=False)
        self.harmony_head = mlp([zh + zs, H // 2, role_dim], norm=False)
        self.melody_head = mlp([zm + zs, H // 2, role_dim], norm=False)
        
        self.dims = dict(struct=zs, energy=ze, rhythm=zr, harmony=zh, melody=zm)
        self.t = t_frames
    
    def reparam(self, mu, lv):
        if self.training:
            return mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return mu
    
    def encode(self, x_flat):
        B = x_flat.size(0)
        T = self.t
        x3d = x_flat.view(B, T, 128, 3)
    
        # Только ролевые энкодеры!
        xr = x3d[:, :, :, 0].reshape(B, -1)  # rhythm
        xh = x3d[:, :, :, 1].reshape(B, -1)  # harmony
        xm = x3d[:, :, :, 2].reshape(B, -1)  # melody
    
        mr, lr = self.rhythm_enc(xr)
        mh, lh = self.harmony_enc(xh)
        mm, lm = self.melody_enc(xm)
    
        # struct и energy генерируем из объединённых ролей
        x_combined = torch.cat([xr, xh, xm], dim=1)
        ms, ls = self.struct_enc(x_combined)
        me, le = self.energy_enc(x_combined)
        
        return dict(mu=dict(struct=ms, energy=me, rhythm=mr, harmony=mh, melody=mm),
                   lv=dict(struct=ls, energy=le, rhythm=lr, harmony=lh, melody=lm))
    
    def sample_z(self, enc_out):
        return {k: self.reparam(enc_out['mu'][k], enc_out['lv'][k]) for k in enc_out['mu']}
    
    def decode_to_musicvae_z(self, z_parts):
        z_cat = torch.cat([z_parts['struct'], z_parts['energy'], z_parts['rhythm'],
                          z_parts['harmony'], z_parts['melody']], dim=1)
        return self.decoder(z_cat)
    
    def forward(self, x_flat):
        enc = self.encode(x_flat)
        z = self.sample_z(enc)
        z_mv_hat = self.decode_to_musicvae_z(z)
        e_pred = self.energy_head(z['energy'])
        r_pred = self.rhythm_head(torch.cat([z['rhythm'], z['struct']], 1))
        h_pred = self.harmony_head(torch.cat([z['harmony'], z['struct']], 1))
        m_pred = self.melody_head(torch.cat([z['melody'], z['struct']], 1))
        return dict(z_mv_hat=z_mv_hat, z_parts=z, enc=enc, e_pred=e_pred,
                   r_pred=r_pred, h_pred=h_pred, m_pred=m_pred)
    
    def sample_structured_z(self, n, device):
        return {k: torch.randn(n, d, device=device) for k, d in self.dims.items()}

# ============================================================
# LOSS ФУНКЦИИ
# ============================================================

def tc_penalty(z_parts):
    z_cat = torch.cat(list(z_parts.values()), dim=1)
    z_norm = (z_cat - z_cat.mean(0)) / (z_cat.std(0) + 1e-8)
    B = z_cat.size(0)
    corr = (z_norm.T @ z_norm) / B
    D = corr.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=corr.device)
    return corr[mask].abs().mean()

def kl_div(enc):
    kl = 0.0
    for k in enc['mu']:
        mu, lv = enc['mu'][k], enc['lv'][k]
        # Ограничиваем logvar, чтобы не уходил в -∞
        lv = lv.clamp(-5, 2)
        kl += -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1).mean()
    return kl

def total_loss(out, z_mv_target, x_flat, energy, cfg, beta):
    B, T = x_flat.size(0), energy.size(1)
    recon_loss = F.mse_loss(out['z_mv_hat'], z_mv_target)
    kl = kl_div(out['enc'])
    energy_loss = F.mse_loss(out['e_pred'], energy)
    
    x3d = x_flat.view(B, T, 128, 3)
    xr = x3d[:, :, :, 0].reshape(B, -1)
    xh = x3d[:, :, :, 1].reshape(B, -1)
    xm = x3d[:, :, :, 2].reshape(B, -1)
    role_loss = (F.binary_cross_entropy_with_logits(out['r_pred'], xr) +
                 F.binary_cross_entropy_with_logits(out['h_pred'], xh) +
                 F.binary_cross_entropy_with_logits(out['m_pred'], xm)) / 3.0
    
    dis_loss = tc_penalty(out['z_parts'])
    total = (cfg['lambda_recon'] * recon_loss + beta * kl +
             cfg['lambda_energy'] * energy_loss + cfg['lambda_role'] * role_loss +
             cfg['lambda_dis'] * dis_loss)
    return total, {'recon': recon_loss.item(), 'kl': kl.item(), 'energy': energy_loss.item(),
                   'role': role_loss.item(), 'dis': dis_loss.item(), 'total': total.item()}

def beta_schedule(epoch, cfg):
    if epoch < 5:  # Всего 5 эпох без KL
        return 0.0
    return min(cfg['lambda_kl'], cfg['lambda_kl'] * (epoch - 5) / cfg['kl_warmup'])

# ============================================================
# ОБУЧЕНИЕ С СОХРАНЕНИЕМ ИСТОРИИ
# ============================================================

controller = StructuredController(CFG, ROLL_FLAT_DIM, T_FRAMES).to(DEVICE)
print(f'Controller parameters: {sum(p.numel() for p in controller.parameters()):,}')

optimizer = torch.optim.AdamW(controller.parameters(), lr=CFG['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['n_epochs'], eta_min=1e-5)

# Расширенная история для всех компонентов loss
history = {k: [] for k in ['train_total', 'train_recon', 'train_kl', 'train_energy', 'train_role', 'train_dis',
                           'val_total', 'val_recon', 'val_kl', 'val_energy', 'val_role', 'val_dis']}
best_val = float('inf')

print('\nStarting training...')
for epoch in range(1, CFG['n_epochs'] + 1):
    beta = beta_schedule(epoch, CFG)
    t0 = time.time()
    
    # Train
    controller.train()
    train_sums = {k: 0.0 for k in ['total', 'recon', 'kl', 'energy', 'role', 'dis']}
    for z_mv, x_flat, energy in train_loader:
        z_mv, x_flat, energy = z_mv.to(DEVICE), x_flat.to(DEVICE), energy.to(DEVICE)
        optimizer.zero_grad()
        out = controller(x_flat)
        loss, parts = total_loss(out, z_mv, x_flat, energy, CFG, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()
        for k, v in parts.items():
            train_sums[k] += v
    train_metrics = {k: v / len(train_loader) for k, v in train_sums.items()}
    
    # Validation
    controller.eval()
    val_sums = {k: 0.0 for k in ['total', 'recon', 'kl', 'energy', 'role', 'dis']}
    with torch.no_grad():
        for z_mv, x_flat, energy in val_loader:
            z_mv, x_flat, energy = z_mv.to(DEVICE), x_flat.to(DEVICE), energy.to(DEVICE)
            out = controller(x_flat)
            _, parts = total_loss(out, z_mv, x_flat, energy, CFG, beta)
            for k, v in parts.items():
                val_sums[k] += v
    val_metrics = {k: v / len(val_loader) for k, v in val_sums.items()}
    
    scheduler.step()
    
    # Сохраняем историю
    for k, v in train_metrics.items():
        history[f'train_{k}'].append(v)
    for k, v in val_metrics.items():
        history[f'val_{k}'].append(v)
    
    if val_metrics['total'] < best_val:
        best_val = val_metrics['total']
        torch.save({'epoch': epoch, 'model_state': controller.state_dict(), 'best_val': best_val, 'cfg': CFG}, 
                  CKPT_DIR / 'best.pt')
    
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{CFG["n_epochs"]} β={beta:.3f} | '
              f'tr: {train_metrics["total"]:.4f} (rec={train_metrics["recon"]:.4f} '
              f'kl={train_metrics["kl"]:.3f} role={train_metrics["role"]:.4f}) | '
              f'val: {val_metrics["total"]:.4f} | {time.time()-t0:.1f}s')

with open(CKPT_DIR / 'history.json', 'w') as f:
    json.dump(history, f)
print(f'\nBest val loss: {best_val:.4f}')

# Загрузка лучшей модели
ckpt = torch.load(CKPT_DIR / 'best.pt', map_location=DEVICE)
controller.load_state_dict(ckpt['model_state'])
controller.eval()
print(f'Loaded best checkpoint: epoch={ckpt["epoch"]}, val={ckpt["best_val"]:.4f}')

# ============================================================
# МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================================

print('\n' + '='*55)
print('Computing test metrics...')
print('='*55)

controller.eval()
metrics = {'z_recon_mse': [], 'energy_corr': [], 'inter_track_corr': [], 'tc_score': []}
kl_per_comp = {k: [] for k in ['struct', 'energy', 'rhythm', 'harmony', 'melody']}

with torch.no_grad():
    for z_mv, x_flat, energy in test_loader:
        z_mv = z_mv.to(DEVICE)
        x_flat = x_flat.to(DEVICE)
        energy = energy.to(DEVICE)
        
        out = controller(x_flat)
        
        mse = F.mse_loss(out['z_mv_hat'], z_mv).item()
        metrics['z_recon_mse'].append(mse)
        
        e_pred = out['e_pred'].cpu().numpy()
        e_true = energy.cpu().numpy()
        for b in range(e_pred.shape[0]):
            if e_true[b].std() > 1e-6 and e_pred[b].std() > 1e-6:
                metrics['energy_corr'].append(float(np.corrcoef(e_true[b], e_pred[b])[0, 1]))
        
        B = x_flat.size(0)
        h_out = torch.sigmoid(out['h_pred']).cpu().numpy().reshape(B, T_FRAMES, 128)
        m_out = torch.sigmoid(out['m_pred']).cpu().numpy().reshape(B, T_FRAMES, 128)
        for b in range(B):
            hd = h_out[b].sum(axis=1)
            md = m_out[b].sum(axis=1)
            if hd.std() > 1e-6 and md.std() > 1e-6:
                metrics['inter_track_corr'].append(float(np.corrcoef(hd, md)[0, 1]))
        
        tc = tc_penalty(out['z_parts']).item()
        metrics['tc_score'].append(tc)
        
        for comp in kl_per_comp:
            mu = out['enc']['mu'][comp]
            lv = out['enc']['lv'][comp]
            kl_c = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean().item()
            kl_per_comp[comp].append(kl_c)

final_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
final_metrics['kl_per_component'] = {k: float(np.mean(v)) for k, v in kl_per_comp.items()}

print(f'\n  z_recon_mse     : {final_metrics["z_recon_mse"]:.4f}  (↓ лучше)')
print(f'  energy_corr     : {final_metrics["energy_corr"]:.4f}  (↑ лучше)')
print(f'  inter_track_corr: {final_metrics["inter_track_corr"]:.4f}  (↑ лучше)')
print(f'  tc_score        : {final_metrics["tc_score"]:.4f}  (↓ лучше)')
print(f'\n  KL per component (> 0.1 = not collapsed):')
for comp, kl_v in final_metrics['kl_per_component'].items():
    bar = '█' * int(kl_v * 10)
    print(f'    {comp:<10}: {kl_v:.3f}  {bar}')

with open(CKPT_DIR / 'test_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)
print('Saved: test_metrics.json')

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

print('\nGenerating plots...')

# 1. Training curves
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
keys = ['total', 'recon', 'kl', 'energy', 'role', 'dis']
titles = ['Total Loss', 'z Recon (MSE)', 'KL Divergence', 'Energy Loss', 'Role Aux Loss', 'TC Penalty']

for ax, key, title in zip(axes.flat, keys, titles):
    if f'train_{key}' in history:
        ax.plot(history[f'train_{key}'], label='train', color='#3498db', lw=1.5)
        ax.plot(history[f'val_{key}'], label='val', color='#e74c3c', lw=1.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)

plt.suptitle('Structured Controller — Training Curves', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150)
plt.close()
print(f'  Saved: {OUTPUT_DIR / "training_curves.png"}')

# 2. KL per component bar chart
comp_kls = final_metrics['kl_per_component']
colors = ['#9b59b6', '#f39c12', '#e74c3c', '#2ecc71', '#3498db']

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(comp_kls.keys(), comp_kls.values(), color=colors, edgecolor='white', width=0.5)
ax.axhline(0.1, color='gray', linestyle='--', lw=1, label='collapse threshold')
ax.set_title('KL Divergence per Latent Component', fontsize=11)
ax.set_ylabel('Mean KL')
ax.legend()
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kl_per_component.png', dpi=150)
plt.close()
print(f'  Saved: {OUTPUT_DIR / "kl_per_component.png"}')

# 3. PCA latent space visualization
from sklearn.decomposition import PCA

z_all_parts = {k: [] for k in ['struct', 'energy', 'rhythm', 'harmony', 'melody']}
with torch.no_grad():
    for _, x_flat, _ in test_loader:
        enc = controller.encode(x_flat.to(DEVICE))
        for k in z_all_parts:
            z_all_parts[k].append(enc['mu'][k].cpu().numpy())
        if sum(len(v) for v in z_all_parts.values()) > 5 * 1000:
            break

z_concat = np.concatenate([np.concatenate(z_all_parts[k], 0) for k in z_all_parts], axis=1)[:1000]
pca = PCA(n_components=2)
z_2d = pca.fit_transform(z_concat)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=np.arange(len(z_2d)), cmap='viridis', alpha=0.4, s=8)
plt.colorbar(sc, label='Sample index')
ax.set_title('Structured Latent Space (PCA)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_pca.png', dpi=150)
plt.close()
print(f'  Saved: {OUTPUT_DIR / "latent_pca.png"}')

# ============================================================
# ГЕНЕРАЦИЯ MIDI
# ============================================================

def decode_z_to_midi(z_parts, temperature=0.5, length=32):
    with torch.no_grad():
        z_mv_hat = controller.decode_to_musicvae_z(z_parts)
    z_np = z_mv_hat.cpu().numpy()
    return musicvae.decode(z_np, length=length, temperature=temperature)

# Random generation
print('\nGenerating random MIDI files...')
for i in range(6):
    z_random = controller.sample_structured_z(1, DEVICE)
    seqs = decode_z_to_midi(z_random)
    out_path = OUTPUT_DIR / f'generated_{i+1:02d}.mid'
    note_seq.sequence_proto_to_midi_file(seqs[0], str(out_path))
    print(f'  Saved: {out_path.name}')

print(f'\n✅ All done! Results in {OUTPUT_DIR}')
print(f'  - MIDI files: {OUTPUT_DIR}/*.mid')
print(f'  - Training curves: {OUTPUT_DIR}/training_curves.png')
print(f'  - KL per component: {OUTPUT_DIR}/kl_per_component.png')
print(f'  - PCA visualization: {OUTPUT_DIR}/latent_pca.png')
print(f'  - Metrics JSON: {CKPT_DIR}/test_metrics.json')
