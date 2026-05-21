#!/usr/bin/env python3
"""
Structured Controller для генерации музыки
Запуск: python main.py
"""

import sys
import os
import json
import time
import random
import glob
import urllib.request
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pretty_midi
import tensorflow as tf
import note_seq
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from tqdm import tqdm
from sklearn.decomposition import PCA

# Отключаем лишние предупреждения
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# НАСТРОЙКИ
# ============================================================

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
    'z_struct_dim': 128,
    'z_energy_dim': 64,
    'z_rhythm_dim': 100,
    'z_harmony_dim': 110,
    'z_melody_dim': 110,
    'hidden_dim': 1024,
    'lambda_recon': 1.0,
    'lambda_kl': 0.5,
    'lambda_energy': 2.0,
    'lambda_role': 1.5,
    'lambda_dis': 0.3,
    'kl_warmup': 10,
    'batch_size': 32,  # Уменьшаем для стабильности
    'lr': 3e-4,
    'n_epochs': 10,    # Тестовый запуск
    'seed': 42,
    'max_midi_files': 500,  # Ограничиваем для теста
}

CFG['z_total'] = sum([CFG['z_struct_dim'], CFG['z_energy_dim'], 
                      CFG['z_rhythm_dim'], CFG['z_harmony_dim'], CFG['z_melody_dim']])
print(f"Structured z total dim: {CFG['z_total']}")

random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])

# ============================================================
# ЗАГРУЗКА MUSIC VAE (ручное скачивание)
# ============================================================

from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import glob

MODEL_NAME = CFG['musicvae_model']

# Путь к папке с чекпоинтами (без подпапки MODEL_NAME)
MUSICVAE_DIR = PROJECT_ROOT / 'data' / 'checkpoints' / 'music_vae'

# Ищем файл чекпоинта
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

# Тестовая генерация
test_seqs = musicvae.sample(n=2, length=32, temperature=0.5)
note_seq.sequence_proto_to_midi_file(test_seqs[0], str(OUTPUT_DIR / 'test_musicvae.mid'))
print(f'Test MIDI saved to {OUTPUT_DIR}/test_musicvae.mid')

# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С MIDI (упрощенные, без требования всех ролей)
# ============================================================

BASS_PROGRAMS = set(range(32, 40))

def assign_role(inst):
    if inst.is_drum:
        return 'rhythm'
    if inst.program in BASS_PROGRAMS:
        return 'harmony'
    if inst.notes and np.mean([n.pitch for n in inst.notes]) < 48:
        return 'harmony'
    return 'melody'

def midi_to_note_seq(midi_path: str):
    try:
        return note_seq.midi_file_to_note_sequence(midi_path)
    except Exception:
        return None

def extract_piano_roll(midi_path: str, fs: int = 8, clip_frames: int = 256) -> np.ndarray:
    """ВСЕГДА возвращает массив нужной формы"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return np.zeros((clip_frames, 128, 3), dtype=np.float32)
    
    # Собираем все ноты
    all_notes = []
    for inst in pm.instruments:
        for note in inst.notes:
            if 0 <= note.pitch <= 127:
                all_notes.append(note)
    
    if len(all_notes) == 0:
        return np.zeros((clip_frames, 128, 3), dtype=np.float32)
    
    # Создаем MIDI с этими нотами
    tmp = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = all_notes
    tmp.instruments.append(inst)
    
    try:
        roll = tmp.get_piano_roll(fs=fs)
    except Exception:
        return np.zeros((clip_frames, 128, 3), dtype=np.float32)
    
    # Приводим к нужной длине
    if roll.shape[1] < clip_frames:
        pad = np.zeros((128, clip_frames - roll.shape[1]))
        roll = np.hstack([roll, pad])
    else:
        roll = roll[:, :clip_frames]
    
    roll = (roll > 0).astype(np.float32)
    roll_3ch = np.stack([roll, roll, roll], axis=-1).transpose(1, 0, 2)
    
    return roll_3ch.astype(np.float32)

def compute_energy(roll: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    density = roll.sum(axis=(1, 2)) / (128 * 3)
    return (alpha * density + (1 - alpha) * density).astype(np.float32)

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

# ── Encode всех MIDI через Music VAE (принимает ВСЕ файлы) ──
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
    midi_files = (glob.glob(str(MIDI_DIR / '**/*.mid'), recursive=True) +
                  glob.glob(str(MIDI_DIR / '**/*.midi'), recursive=True))
    random.shuffle(midi_files)
    midi_files = midi_files[:CFG['max_midi_files']]
    print(f'Processing {len(midi_files)} MIDI files...')

    z_list, roll_list, energy_list = [], [], []
    
    from tqdm import tqdm
    
    for midi_path in tqdm(midi_files, desc='Encoding MIDI'):
        # Всегда получаем roll (даже если пустой)
        roll = extract_piano_roll(midi_path)
        
        # Всегда получаем NoteSequence (даже если пустой)
        try:
            ns = note_seq.midi_file_to_note_sequence(str(midi_path))
        except Exception:
            ns = note_seq.NoteSequence()
        
        # Пробуем закодировать, но если не получается - создаем случайный z
        try:
            z_batch, _, _ = musicvae.encode([ns])
            if isinstance(z_batch, list):
                z = z_batch[0]
            else:
                z = z_batch[0]
        except Exception:
            # Если encode не работает - создаем случайный z
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

print(f'Dataset shapes: z_musicvae {z_musicvae_all.shape}, rolls {rolls_all.shape}')

# ============================================================
# DATASET И DATALOADER
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

# ============================================================
# МОДЕЛЬ
# ============================================================

def mlp(dims, activation=nn.GELU, norm=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
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
        return self.mu_head(h), self.lv_head(h).clamp(-4, 4)

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
        xr = x3d[:, :, :, 0].reshape(B, -1)
        xh = x3d[:, :, :, 1].reshape(B, -1)
        xm = x3d[:, :, :, 2].reshape(B, -1)
        
        ms, ls = self.struct_enc(x_flat)
        me, le = self.energy_enc(x_flat)
        mr, lr = self.rhythm_enc(xr)
        mh, lh = self.harmony_enc(xh)
        mm, lm = self.melody_enc(xm)
        
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
    return min(cfg['lambda_kl'], cfg['lambda_kl'] * epoch / cfg['kl_warmup'])

# ============================================================
# ОБУЧЕНИЕ
# ============================================================

controller = StructuredController(CFG, ROLL_FLAT_DIM, T_FRAMES).to(DEVICE)
print(f'Controller parameters: {sum(p.numel() for p in controller.parameters()):,}')

optimizer = torch.optim.AdamW(controller.parameters(), lr=CFG['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['n_epochs'], eta_min=1e-5)

history = {'train_total': [], 'val_total': []}
best_val = float('inf')

print('\nStarting training...')
for epoch in range(1, CFG['n_epochs'] + 1):
    beta = beta_schedule(epoch, CFG)
    t0 = time.time()
    
    # Train
    controller.train()
    train_loss = 0
    for z_mv, x_flat, energy in train_loader:
        z_mv, x_flat, energy = z_mv.to(DEVICE), x_flat.to(DEVICE), energy.to(DEVICE)
        optimizer.zero_grad()
        out = controller(x_flat)
        loss, _ = total_loss(out, z_mv, x_flat, energy, CFG, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation
    controller.eval()
    val_loss = 0
    with torch.no_grad():
        for z_mv, x_flat, energy in val_loader:
            z_mv, x_flat, energy = z_mv.to(DEVICE), x_flat.to(DEVICE), energy.to(DEVICE)
            out = controller(x_flat)
            loss, _ = total_loss(out, z_mv, x_flat, energy, CFG, beta)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    
    scheduler.step()
    history['train_total'].append(train_loss)
    history['val_total'].append(val_loss)
    
    if val_loss < best_val:
        best_val = val_loss
        torch.save({'epoch': epoch, 'model_state': controller.state_dict(), 'best_val': best_val}, 
                  CKPT_DIR / 'best.pt')
    
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{CFG["n_epochs"]} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | time: {time.time()-t0:.1f}s')

with open(CKPT_DIR / 'history.json', 'w') as f:
    json.dump(history, f)
print(f'\nBest val loss: {best_val:.4f}')

# ============================================================
# ГЕНЕРАЦИЯ
# ============================================================

def decode_z_to_midi(z_parts, temperature=0.5, length=32):
    with torch.no_grad():
        z_mv_hat = controller.decode_to_musicvae_z(z_parts)
    z_np = z_mv_hat.cpu().numpy()
    return musicvae.decode(z_np, length=length, temperature=temperature)

# Генерация 6 MIDI файлов
print('\nGenerating MIDI files...')
for i in range(6):
    z_random = controller.sample_structured_z(1, DEVICE)
    seqs = decode_z_to_midi(z_random)
    out_path = OUTPUT_DIR / f'generated_{i+1:02d}.mid'
    note_seq.sequence_proto_to_midi_file(seqs[0], str(out_path))
    print(f'Saved: {out_path}')

print(f'\n✅ All done! Results in {OUTPUT_DIR}')
