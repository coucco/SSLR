import magenta
print('Magenta version:', magenta.__version__)

import os, json, time, random, glob
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pretty_midi

# TensorFlow (нужен для Music VAE)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

DEVICE = torch.device('cuda')
print(f'PyTorch device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Пути ──────────────────────────────────────────────────────
from pathlib import Path

# Корень проекта (текущая директория)
PROJECT_ROOT = Path.cwd()

# Директории проекта
DATA_DIR    = PROJECT_ROOT / 'data' / 'musicvae_data'
CKPT_DIR    = PROJECT_ROOT / 'data' / 'checkpoints' / 'structured_controller'
OUTPUT_DIR  = PROJECT_ROOT / 'outputs'
MIDI_DIR    = PROJECT_ROOT / 'data' / 'raw_midi'

# Создаем директории
for d in [DATA_DIR, CKPT_DIR, OUTPUT_DIR, MIDI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Гиперпараметры ────────────────────────────────────────────
CFG = {
    # Music VAE
    'musicvae_z_dim': 512,       # размерность z в Music VAE (фиксировано)
    'musicvae_model': 'cat-mel_2bar_big',  # какую модель используем

    # Structured latent (ваши компоненты)
    # Сумма = 512 чтобы маппинг был равномерным
    'z_struct_dim':  128,
    'z_energy_dim':   64,
    'z_rhythm_dim':  100,
    'z_harmony_dim': 110,
    'z_melody_dim':  110,

    # Encoder/Decoder MLP
    'hidden_dim': 1024,

    # Loss weights
    'lambda_recon':   1.0,
    'lambda_kl':      0.5,
    'lambda_energy':  2.0,
    'lambda_role':    1.5,
    'lambda_dis':     0.3,
    'kl_warmup':      10,

    # Training
    'batch_size':    128,
    'lr':           3e-4,
    'n_epochs':      60,
    'seed':          42,

    # Data
    'max_midi_files': 3000,
}

CFG['z_total'] = (CFG['z_struct_dim'] + CFG['z_energy_dim'] +
                   CFG['z_rhythm_dim'] + CFG['z_harmony_dim'] + CFG['z_melody_dim'])
print(f"Structured z total dim: {CFG['z_total']} (Music VAE z: {CFG['musicvae_z_dim']})")

random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])

print('Config ready.')

from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import note_seq

# ── Скачать веса Music VAE ────────────────────────────────────
# cat-mel_2bar_big: мелодии 2-bar, хорошо генерирует, быстрый энкодинг
# Альтернативы:
#   'hierdec-mel_16bar'  — 16 тактов, более структурированная музыка
#   'groovae_2bar_humanize' — только ритм

MODEL_NAME = CFG['musicvae_model']   # 'cat-mel_2bar_big'
CKPT_PATH  = f'/content/{MODEL_NAME}.tar'

if not os.path.exists(CKPT_PATH):
    print(f'Downloading Music VAE weights ({MODEL_NAME})...')
    !wget -q --show-progress -O {CKPT_PATH} \
        https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/{MODEL_NAME}.tar
    !tar -xf {CKPT_PATH} -C /content/
    print('Downloaded.')
else:
    print('Already downloaded.')

# Найдём папку с чекпоинтом
import glob as glib
ckpt_dirs = glib.glob(f'/content/{MODEL_NAME}*/')
print('Checkpoint dirs found:', ckpt_dirs)

MUSICVAE_CKPT = ckpt_dirs[0] if ckpt_dirs else f'/content/{MODEL_NAME}'

# ── Инициализация модели ──────────────────────────────────────
print('Loading Music VAE...')
musicvae_config = configs.CONFIG_MAP[MODEL_NAME]
musicvae = TrainedModel(
    config=musicvae_config,
    batch_size=CFG['batch_size'],
    checkpoint_dir_or_path=MUSICVAE_CKPT,
)
print('Music VAE loaded successfully.')

# ── Быстрая проверка: сгенерируем случайные клипы ────────────
print('\nGenerating test samples from Music VAE prior...')
test_seqs = musicvae.sample(n=4, length=32, temperature=0.5)
print(f'Generated {len(test_seqs)} sequences.')

# Сохранить один как MIDI для проверки
note_seq.sequence_proto_to_midi_file(
    test_seqs[0], str(OUTPUT_DIR / 'musicvae_raw_sample.mid')
)
print(f'Saved test MIDI: {OUTPUT_DIR}/musicvae_raw_sample.mid')

# ════════════════════════════════════════════════════════════
# Идея: прогнать все MIDI файлы через Music VAE encoder,
# получить z-векторы (512-dim) и сохранить их.
# Это датасет для обучения нашего structured controller.
# Структурированный энкодер учится: MIDI → наши z-компоненты
# Структурированный декодер учится: наши z → z_musicvae
# ════════════════════════════════════════════════════════════

BASS_PROGRAMS = set(range(32, 40))

def assign_role(inst):
    if inst.is_drum: return 'rhythm'
    if inst.program in BASS_PROGRAMS: return 'harmony'
    if inst.notes and np.mean([n.pitch for n in inst.notes]) < 48:
        return 'harmony'
    return 'melody'


def midi_to_note_seq(midi_path: str):
    """MIDI файл → NoteSequence для Music VAE."""
    try:
        return note_seq.midi_file_to_note_sequence(midi_path)
    except Exception:
        return None


def extract_piano_roll(midi_path: str, fs: int = 8,
                       clip_frames: int = 256) -> Optional[np.ndarray]:
    """
    Извлекает piano-roll (clip_frames, 128, 3) для вычисления
    вспомогательных признаков (energy, роли).
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None

    role_notes = {'rhythm': [], 'harmony': [], 'melody': []}
    for inst in pm.instruments:
        role_notes[assign_role(inst)].extend(inst.notes)

    if any(len(v) == 0 for v in role_notes.values()):
        return None

    rolls = {}
    for role in ['rhythm', 'harmony', 'melody']:
        tmp = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=(role=='rhythm'))
        inst.notes = role_notes[role]
        tmp.instruments.append(inst)
        try:
            roll = tmp.get_piano_roll(fs=fs)  # (128, T)
        except Exception:
            return None
        rolls[role] = (roll > 0).astype(np.float32)

    min_len = min(r.shape[1] for r in rolls.values())
    if min_len < clip_frames:
        return None

    stack = np.stack([
        rolls['rhythm'][:, :clip_frames],
        rolls['harmony'][:, :clip_frames],
        rolls['melody'][:, :clip_frames],
    ], axis=-1).transpose(1, 0, 2)  # (T, 128, 3)

    return stack.astype(np.float32)


def compute_energy(roll: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """roll: (T, 128, 3) → energy (T,)"""
    density = roll.sum(axis=(1, 2)) / (128 * 3)
    return (alpha * density + (1 - alpha) * density).astype(np.float32)


# ── Encode всех MIDI через Music VAE ─────────────────────────
Z_CACHE = DATA_DIR / 'musicvae_z_vectors.npy'
ROLL_CACHE = DATA_DIR / 'piano_rolls.npy'
ENERGY_CACHE = DATA_DIR / 'energy_profiles.npy'

if Z_CACHE.exists() and ROLL_CACHE.exists():
    print('Loading cached data...')
    z_musicvae_all = np.load(Z_CACHE)
    rolls_all      = np.load(ROLL_CACHE)
    energy_all     = np.load(ENERGY_CACHE)
    print(f'Loaded: {len(z_musicvae_all)} samples')
else:
    midi_files = (glob.glob(str(MIDI_DIR / '**/*.mid'),  recursive=True) +
                  glob.glob(str(MIDI_DIR / '**/*.midi'), recursive=True))
    random.shuffle(midi_files)
    midi_files = midi_files[:CFG['max_midi_files']]
    print(f'Processing {len(midi_files)} MIDI files...')

    z_list, roll_list, energy_list = [], [], []
    skipped = 0
    BATCH_ENCODE = 32   # кодируем батчами для скорости

    batch_seqs, batch_rolls, batch_paths = [], [], []

    def flush_batch():
        nonlocal batch_seqs, batch_rolls, batch_paths
        if not batch_seqs:
            return
        try:
            # Music VAE encode: NoteSequence → z (numpy, float32)
            z_batch, _, _ = musicvae.encode(batch_seqs)
            # z_batch: list of numpy arrays (512,) or ndarray (B, 512)
            if isinstance(z_batch, list):
                z_batch = np.stack(z_batch, axis=0)
            for i in range(len(batch_seqs)):
                z_list.append(z_batch[i])
                roll_list.append(batch_rolls[i])
                energy_list.append(compute_energy(batch_rolls[i]))
        except Exception as e:
            pass  # пропускаем битые батчи
        batch_seqs.clear(); batch_rolls.clear(); batch_paths.clear()

    from tqdm import tqdm
    for midi_path in tqdm(midi_files, desc='Encoding MIDI → z_musicvae'):
        ns   = midi_to_note_seq(midi_path)
        roll = extract_piano_roll(midi_path)
        if ns is None or roll is None:
            skipped += 1
            continue

        batch_seqs.append(ns)
        batch_rolls.append(roll)
        batch_paths.append(midi_path)

        if len(batch_seqs) >= BATCH_ENCODE:
            flush_batch()

    flush_batch()  # остаток

    print(f'Done. Accepted: {len(z_list)}, Skipped: {skipped}')

    z_musicvae_all = np.stack(z_list,   axis=0).astype(np.float32)
    rolls_all      = np.stack(roll_list, axis=0).astype(np.float32)
    energy_all     = np.stack(energy_list, axis=0).astype(np.float32)

    np.save(Z_CACHE,      z_musicvae_all)
    np.save(ROLL_CACHE,   rolls_all)
    np.save(ENERGY_CACHE, energy_all)
    print(f'Saved to {DATA_DIR}')

print(f'\nDataset shapes:')
print(f'  z_musicvae: {z_musicvae_all.shape}   (N, 512)')
print(f'  rolls:      {rolls_all.shape}   (N, T, 128, 3)')
print(f'  energy:     {energy_all.shape}  (N, T)')

class ZDataset(Dataset):
    """
    Возвращает тройку:
      z_mv   — z-вектор Music VAE (512,)  — цель для реконструкции
      x_flat — piano-roll (T*128*3,)       — вход structured encoder
      energy — энергия (T,)                — цель для energy head
    """
    def __init__(self, z_mv, rolls, energy):
        self.z_mv   = torch.from_numpy(z_mv)
        self.x_flat = torch.from_numpy(
            rolls.reshape(len(rolls), -1)   # (N, T*128*3)
        )
        self.energy = torch.from_numpy(energy)

    def __len__(self): return len(self.z_mv)

    def __getitem__(self, idx):
        return self.z_mv[idx], self.x_flat[idx], self.energy[idx]


N = len(z_musicvae_all)
idx = np.random.permutation(N)
n_train = int(0.85 * N)
n_val   = int(0.10 * N)

tr_idx = idx[:n_train]
vl_idx = idx[n_train:n_train + n_val]
te_idx = idx[n_train + n_val:]

train_ds = ZDataset(z_musicvae_all[tr_idx], rolls_all[tr_idx], energy_all[tr_idx])
val_ds   = ZDataset(z_musicvae_all[vl_idx], rolls_all[vl_idx], energy_all[vl_idx])
test_ds  = ZDataset(z_musicvae_all[te_idx], rolls_all[te_idx], energy_all[te_idx])

kw = dict(batch_size=CFG['batch_size'], num_workers=2, pin_memory=True)
train_loader = DataLoader(train_ds, shuffle=True,  **kw)
val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

ROLL_FLAT_DIM = rolls_all.shape[1] * 128 * 3
T_FRAMES      = rolls_all.shape[1]

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
print(f'Roll flat dim: {ROLL_FLAT_DIM}')

# ════════════════════════════════════════════════════════════
# Structured Controller:
#
#  StructuredEncoder:
#    piano-roll (T*128*3) → z_struct, z_energy,
#                           z_rhythm, z_harmony, z_melody
#
#  StructuredDecoder:
#    [z_struct|z_energy|z_rhythm|z_harmony|z_melody] → z_musicvae_hat
#
#  EnergyHead:
#    z_energy → energy profile (T,)
#
#  RoleHeads (auxiliary):
#    z_rhythm  + z_struct → rhythm  piano-roll (T*128)
#    z_harmony + z_struct → harmony piano-roll (T*128)
#    z_melody  + z_struct → melody  piano-roll (T*128)
# ════════════════════════════════════════════════════════════

def mlp(dims, activation=nn.GELU, norm=True):
    """Helper: строит MLP по списку размерностей."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:  # не после последнего
            if norm: layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(activation())
    return nn.Sequential(*layers)


class ComponentEncoder(nn.Module):
    """Кодирует произвольный вход в Gaussian (mu, logvar)."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net     = mlp([in_dim, hidden, hidden // 2])
        self.mu_head = nn.Linear(hidden // 2, out_dim)
        self.lv_head = nn.Linear(hidden // 2, out_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h).clamp(-4, 4)


class StructuredController(nn.Module):
    def __init__(self, cfg, roll_flat_dim, t_frames):
        super().__init__()
        H   = cfg['hidden_dim']
        MVZ = cfg['musicvae_z_dim']
        zs  = cfg['z_struct_dim']
        ze  = cfg['z_energy_dim']
        zr  = cfg['z_rhythm_dim']
        zh  = cfg['z_harmony_dim']
        zm  = cfg['z_melody_dim']
        role_dim = t_frames * 128

        # ── Encoders ──────────────────────────────────────────
        # Глобальные (видят весь piano-roll)
        self.struct_enc = ComponentEncoder(roll_flat_dim, H, zs)
        self.energy_enc = ComponentEncoder(roll_flat_dim, H, ze)
        # Поролевые (видят один канал)
        self.rhythm_enc  = ComponentEncoder(role_dim, H // 2, zr)
        self.harmony_enc = ComponentEncoder(role_dim, H // 2, zh)
        self.melody_enc  = ComponentEncoder(role_dim, H // 2, zm)

        # ── Main decoder: structured z → z_musicvae ───────────
        total_z = zs + ze + zr + zh + zm
        self.decoder = mlp([total_z, H, H, H, MVZ])

        # ── Auxiliary heads ───────────────────────────────────
        self.energy_head = nn.Sequential(
            mlp([ze, H // 4, t_frames], norm=False), nn.Sigmoid()
        )
        self.rhythm_head  = mlp([zr + zs, H // 2, role_dim], norm=False)
        self.harmony_head = mlp([zh + zs, H // 2, role_dim], norm=False)
        self.melody_head  = mlp([zm + zs, H // 2, role_dim], norm=False)

        self.dims = dict(struct=zs, energy=ze, rhythm=zr,
                         harmony=zh, melody=zm, total=total_z)
        self.cfg  = cfg
        self.t    = t_frames

    # ── Reparameterization ────────────────────────────────────
    def reparam(self, mu, lv):
        if self.training:
            return mu + (0.5 * lv).exp() * torch.randn_like(mu)
        return mu

    # ── Encode piano-roll → structured z parts ────────────────
    def encode(self, x_flat):
        """
        Вход: x_flat (B, T*128*3)
        Выход: dict с mu и logvar каждого компонента
        """
        B = x_flat.size(0)
        T = self.t
        x3d = x_flat.view(B, T, 128, 3)
        xr  = x3d[:, :, :, 0].reshape(B, -1)
        xh  = x3d[:, :, :, 1].reshape(B, -1)
        xm  = x3d[:, :, :, 2].reshape(B, -1)

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
        """Reparameterize каждый компонент."""
        return {k: self.reparam(enc_out['mu'][k], enc_out['lv'][k])
                for k in enc_out['mu']}

    # ── Decode structured z → z_musicvae ─────────────────────
    def decode_to_musicvae_z(self, z_parts):
        """
        Вход: dict z_parts с ключами struct/energy/rhythm/harmony/melody
        Выход: z_musicvae_hat (B, 512)
        """
        z_cat = torch.cat([
            z_parts['struct'], z_parts['energy'],
            z_parts['rhythm'], z_parts['harmony'], z_parts['melody']
        ], dim=1)
        return self.decoder(z_cat)

    def forward(self, x_flat):
        enc    = self.encode(x_flat)
        z      = self.sample_z(enc)
        z_mv_hat = self.decode_to_musicvae_z(z)

        # Auxiliary predictions
        e_pred = self.energy_head(z['energy'])
        r_pred = self.rhythm_head(torch.cat([z['rhythm'],  z['struct']], 1))
        h_pred = self.harmony_head(torch.cat([z['harmony'], z['struct']], 1))
        m_pred = self.melody_head(torch.cat([z['melody'],  z['struct']], 1))

        return dict(
            z_mv_hat=z_mv_hat,
            z_parts=z, enc=enc,
            e_pred=e_pred,
            r_pred=r_pred, h_pred=h_pred, m_pred=m_pred,
        )

    # ── Генерация через Music VAE ──────────────────────────────
    def sample_structured_z(self, n, device):
        """Сэмплируем структурированный z из prior N(0,I)."""
        return {k: torch.randn(n, d, device=device)
                for k, d in self.dims.items() if k != 'total'}

    def swap(self, z_a, z_b, component):
        """Подменяем один компонент из z_b в z_a."""
        z_new = {k: v.clone() for k, v in z_a.items()}
        z_new[component] = z_b[component].clone()
        return z_new

    def interpolate(self, z_a, z_b, alpha):
        """Линейная интерполяция между двумя z."""
        return {k: (1 - alpha) * z_a[k] + alpha * z_b[k] for k in z_a}


controller = StructuredController(CFG, ROLL_FLAT_DIM, T_FRAMES).to(DEVICE)

n_params = sum(p.numel() for p in controller.parameters())
print(f'Controller parameters: {n_params:,}')
print(f'Latent components: {controller.dims}')

def tc_penalty(z_parts):
    """Total Correlation proxy: off-diagonal |correlation| между компонентами."""
    # Конкатенируем все компоненты
    z_cat = torch.cat(list(z_parts.values()), dim=1)
    z_norm = (z_cat - z_cat.mean(0)) / (z_cat.std(0) + 1e-8)
    B = z_cat.size(0)
    corr = (z_norm.T @ z_norm) / B
    D = corr.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=corr.device)
    return corr[mask].abs().mean()


def kl_div(enc):
    """KL divergence для всех компонентов суммарно."""
    kl = 0.0
    for k in enc['mu']:
        mu, lv = enc['mu'][k], enc['lv'][k]
        kl += -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1).mean()
    return kl


def total_loss(out, z_mv_target, x_flat, energy, cfg, beta):
    """
    Главная loss функция.

    Компоненты:
    1. recon_loss  — MSE между z_mv_hat и z_mv_target
                     (учим воспроизводить z Music VAE)
    2. kl_loss     — KL от prior N(0,I)
    3. energy_loss — MSE энергетического профиля
    4. role_loss   — BCE для каждого ролевого трека
    5. dis_loss    — TC penalty (диsentanglement)
    """
    B  = x_flat.size(0)
    T  = energy.size(1)

    # 1. Recon: MSE в пространстве Music VAE z
    recon_loss = F.mse_loss(out['z_mv_hat'], z_mv_target)

    # 2. KL
    kl = kl_div(out['enc'])

    # 3. Energy
    energy_loss = F.mse_loss(out['e_pred'], energy)

    # 4. Role losses (auxiliary piano-roll reconstruction)
    x3d = x_flat.view(B, T, 128, 3)
    xr  = x3d[:, :, :, 0].reshape(B, -1)
    xh  = x3d[:, :, :, 1].reshape(B, -1)
    xm  = x3d[:, :, :, 2].reshape(B, -1)
    role_loss = (
        F.binary_cross_entropy_with_logits(out['r_pred'], xr, reduction='mean') +
        F.binary_cross_entropy_with_logits(out['h_pred'], xh, reduction='mean') +
        F.binary_cross_entropy_with_logits(out['m_pred'], xm, reduction='mean')
    ) / 3.0

    # 5. TC disentanglement penalty
    dis_loss = tc_penalty(out['z_parts'])

    total = (
        cfg['lambda_recon']  * recon_loss  +
        beta                 * kl          +
        cfg['lambda_energy'] * energy_loss +
        cfg['lambda_role']   * role_loss   +
        cfg['lambda_dis']    * dis_loss
    )

    parts = dict(
        recon=recon_loss.item(), kl=kl.item(),
        energy=energy_loss.item(), role=role_loss.item(),
        dis=dis_loss.item(), total=total.item(),
    )
    return total, parts


def beta_schedule(epoch, cfg):
    w = cfg['kl_warmup']
    return min(cfg['lambda_kl'], cfg['lambda_kl'] * epoch / w)


print('Loss functions defined.')

optimizer = torch.optim.AdamW(controller.parameters(),
                               lr=CFG['lr'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['n_epochs'], eta_min=1e-5
)

history   = {k: [] for k in ['train_total','train_recon','train_kl',
                               'train_energy','train_role','train_dis',
                               'val_total','val_recon','val_kl',
                               'val_energy','val_role','val_dis']}
best_val  = float('inf')


def run_epoch(loader, train=True):
    controller.train() if train else controller.eval()
    sums = {k: 0.0 for k in ['total','recon','kl','energy','role','dis']}
    ctx  = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for z_mv, x_flat, energy in loader:
            z_mv   = z_mv.to(DEVICE)
            x_flat = x_flat.to(DEVICE)
            energy = energy.to(DEVICE)

            out    = controller(x_flat)
            loss, parts = total_loss(out, z_mv, x_flat, energy, CFG, beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                optimizer.step()

            for k, v in parts.items():
                sums[k] += v

    n = len(loader)
    return {k: v / n for k, v in sums.items()}


print('Starting training...')
for epoch in range(1, CFG['n_epochs'] + 1):
    beta  = beta_schedule(epoch, CFG)
    t0    = time.time()
    tr    = run_epoch(train_loader, train=True)
    vl    = run_epoch(val_loader,   train=False)
    scheduler.step()

    for k, v in tr.items(): history[f'train_{k}'].append(v)
    for k, v in vl.items(): history[f'val_{k}'].append(v)

    if vl['total'] < best_val:
        best_val = vl['total']
        torch.save(dict(
            epoch=epoch, cfg=CFG,
            model_state=controller.state_dict(),
            best_val=best_val,
        ), CKPT_DIR / 'best.pt')

    if epoch % 5 == 0 or epoch == 1:
        print(f'Ep {epoch:3d}/{CFG["n_epochs"]} β={beta:.3f} | '
              f'tr: {tr["total"]:.4f} (rec={tr["recon"]:.4f} '
              f'kl={tr["kl"]:.3f} role={tr["role"]:.4f}) | '
              f'val: {vl["total"]:.4f} | {time.time()-t0:.1f}s')

with open(CKPT_DIR / 'history.json', 'w') as f:
    json.dump(history, f)
print(f'\nBest val loss: {best_val:.4f}')

ckpt = torch.load(CKPT_DIR / 'best.pt', map_location=DEVICE)
controller.load_state_dict(ckpt['model_state'])
controller.eval()
print(f'Loaded best checkpoint: epoch={ckpt["epoch"]}, val={ckpt["best_val"]:.4f}')


def decode_z_to_midi(z_parts: dict, temperature: float = 0.5,
                     length: int = 32) -> list:
    """
    Главная функция генерации.

    Вход:
        z_parts    — dict с компонентами (torch tensors, device)
        temperature — "креативность" Music VAE (0.1=консервативно, 1.0=хаос)
        length      — длина в шагах (32 = 2 такта при 120 BPM)

    Выход:
        list[NoteSequence] — готовые для сохранения в MIDI
    """
    with torch.no_grad():
        z_mv_hat = controller.decode_to_musicvae_z(z_parts)  # (B, 512)

    # Переводим в numpy для Music VAE
    z_np = z_mv_hat.cpu().numpy()  # (B, 512)

    # Music VAE decode: z_numpy → NoteSequences
    sequences = musicvae.decode(
        z_np,
        length=length,
        temperature=temperature,
    )
    return sequences


# Быстрый тест: сгенерировать из prior
print('Test generation from structured prior...')
z_test = controller.sample_structured_z(2, DEVICE)
seqs   = decode_z_to_midi(z_test, temperature=0.5)
print(f'Generated {len(seqs)} sequences OK')

# ════════════════════════════════════════════════════════════
# Сэмплируем z из prior N(0,I) для каждого компонента
# отдельно → декодируем через Music VAE → настоящий MIDI
# ════════════════════════════════════════════════════════════
N_GEN = 6
z_random = controller.sample_structured_z(N_GEN, DEVICE)
generated_seqs = decode_z_to_midi(z_random, temperature=0.5)

print(f'Saving {N_GEN} generated MIDI files...')
for i, seq in enumerate(generated_seqs):
    out_path = OUTPUT_DIR / f'generated_{i+1:02d}.mid'
    note_seq.sequence_proto_to_midi_file(seq, str(out_path))
    print(f'  Saved: {out_path.name}')

# Прослушать в Colab
try:
    from IPython.display import display
    import note_seq
    print('\nPlaying first generated clip:')
    display(note_seq.plot_sequence(generated_seqs[0]))
    note_seq.play_sequence(generated_seqs[0], synth=note_seq.fluidsynth)
except Exception as e:
    print(f'(Playback unavailable: {e} — download MIDI files from Drive)')

# ════════════════════════════════════════════════════════════
# Фиксируем базовый z, варьируем один компонент.
# Это показывает что каждый компонент контролирует
# свой аспект музыки независимо.
# ════════════════════════════════════════════════════════════
N_VARIATIONS = 4
components   = ['struct', 'energy', 'rhythm', 'harmony', 'melody']

# Базовый z (фиксированный сид)
torch.manual_seed(7)
z_base = controller.sample_structured_z(1, DEVICE)

all_seqs    = {}   # component → list of NoteSequence
all_z_hats  = {}   # component → list of z_musicvae_hat tensors

for comp in components:
    comp_seqs = []
    for var_idx in range(N_VARIATIONS):
        # Новый случайный вектор только для одного компонента
        z_varied = {k: v.clone() for k, v in z_base.items()}
        z_varied[comp] = torch.randn_like(z_base[comp]) * 1.5

        seqs = decode_z_to_midi(z_varied, temperature=0.4)
        comp_seqs.append(seqs[0])

        # Сохранить MIDI
        out_path = OUTPUT_DIR / f'control_{comp}_var{var_idx+1}.mid'
        note_seq.sequence_proto_to_midi_file(seqs[0], str(out_path))

    all_seqs[comp] = comp_seqs
    print(f'  {comp}: {N_VARIATIONS} variations saved')

print('Component control experiment complete.')

# ════════════════════════════════════════════════════════════
# Берём два реальных MIDI файла A и B,
# кодируем их через наш controller,
# комбинируем компоненты → генерируем гибриды.
# ════════════════════════════════════════════════════════════
from tqdm import tqdm

def find_valid_midis(n=2):
    midi_files = glob.glob(str(MIDI_DIR / '**/*.mid'), recursive=True)
    random.shuffle(midi_files)
    found = []
    for f in midi_files:
        roll = extract_piano_roll(f)
        if roll is not None:
            found.append((f, roll))
        if len(found) == n:
            break
    return found

print('Finding valid MIDI files...')
valid = find_valid_midis(2)

if len(valid) < 2:
    print('Not enough valid MIDI files found. Falling back to random z.')
    z_A = controller.sample_structured_z(1, DEVICE)
    z_B = controller.sample_structured_z(1, DEVICE)
    name_A, name_B = 'random_A', 'random_B'
else:
    path_A, roll_A = valid[0]
    path_B, roll_B = valid[1]
    name_A = Path(path_A).stem[:20]
    name_B = Path(path_B).stem[:20]

    print(f'Track A: {name_A}')
    print(f'Track B: {name_B}')

    # Кодируем через controller
    def roll_to_z(roll_np):
        x = torch.from_numpy(roll_np.reshape(1, -1)).to(DEVICE)
        with torch.no_grad():
            enc = controller.encode(x)
        # Используем posterior mean (детерминированное кодирование)
        return {k: enc['mu'][k] for k in enc['mu']}

    z_A = roll_to_z(roll_A)
    z_B = roll_to_z(roll_B)

# ── Сценарии комбинирования ────────────────────────────────
scenarios = {
    'A_melody__B_harmony': controller.swap(z_A, z_B, 'harmony'),
    'A_harmony__B_melody': controller.swap(z_A, z_B, 'melody'),
    'A_struct__B_roles':   controller.swap(z_B, z_A, 'struct'),
    'A_energy__B_music':   controller.swap(z_B, z_A, 'energy'),
    'pure_A': z_A,
    'pure_B': z_B,
}

print('\nGenerating swap scenarios...')
for scenario_name, z_mixed in scenarios.items():
    seqs = decode_z_to_midi(z_mixed, temperature=0.4)
    out  = OUTPUT_DIR / f'swap_{scenario_name}.mid'
    note_seq.sequence_proto_to_midi_file(seqs[0], str(out))
    print(f'  Saved: {out.name}')

N_STEPS = 8
alphas  = np.linspace(0, 1, N_STEPS)

print(f'Interpolating A → B in {N_STEPS} steps...')
interp_seqs = []
for alpha in alphas:
    z_interp = controller.interpolate(z_A, z_B, float(alpha))
    seqs     = decode_z_to_midi(z_interp, temperature=0.4)
    interp_seqs.append(seqs[0])
    out = OUTPUT_DIR / f'interp_alpha{alpha:.2f}.mid'
    note_seq.sequence_proto_to_midi_file(seqs[0], str(out))

print(f'Saved {N_STEPS} interpolation steps.')

# Визуализация piano-roll всех шагов
def noteseq_to_pianoroll(ns, fs=8, n_frames=256):
    pm = note_seq.note_sequence_to_pretty_midi(ns)
    roll = pm.get_piano_roll(fs=fs)  # (128, T)
    T = min(roll.shape[1], n_frames)
    out = np.zeros((n_frames, 128))
    out[:T] = roll[:, :T].T
    return (out > 0).astype(np.float32)

fig, axes = plt.subplots(1, N_STEPS, figsize=(N_STEPS * 2.5, 3))
for i, (seq, alpha) in enumerate(zip(interp_seqs, alphas)):
    roll = noteseq_to_pianoroll(seq)
    axes[i].imshow(roll.T, aspect='auto', origin='lower',
                   cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest')
    axes[i].set_title(f'α={alpha:.2f}', fontsize=8)
    axes[i].axis('off')

plt.suptitle(f'Interpolation: {name_A} → {name_B}', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'interpolation.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: interpolation.png')

# ════════════════════════════════════════════════════════════
# Метрики:
#
# 1. z_recon_mse     — насколько точно мы воспроизводим z_musicvae
#                      (качество маппинга)
# 2. energy_corr     — корреляция предсказанной и реальной энергии
# 3. inter_track_corr— согласованность между ролевыми предсказаниями
# 4. kl_per_component— KL каждого компонента (проверка на collapse)
# 5. tc_score        — total correlation (диentanglement)
# ════════════════════════════════════════════════════════════

controller.eval()
metrics = dict(
    z_recon_mse=[], energy_corr=[],
    inter_track_corr=[], tc_score=[],
)
kl_per_comp = {k: [] for k in ['struct','energy','rhythm','harmony','melody']}

with torch.no_grad():
    for z_mv, x_flat, energy in test_loader:
        z_mv   = z_mv.to(DEVICE)
        x_flat = x_flat.to(DEVICE)
        energy = energy.to(DEVICE)

        out = controller(x_flat)

        # 1. z recon MSE
        mse = F.mse_loss(out['z_mv_hat'], z_mv).item()
        metrics['z_recon_mse'].append(mse)

        # 2. Energy correlation
        e_pred = out['e_pred'].cpu().numpy()   # (B, T)
        e_true = energy.cpu().numpy()          # (B, T)
        for b in range(e_pred.shape[0]):
            if e_true[b].std() > 1e-6 and e_pred[b].std() > 1e-6:
                metrics['energy_corr'].append(
                    float(np.corrcoef(e_true[b], e_pred[b])[0,1])
                )

        # 3. Inter-track correlation (harmony vs melody role heads)
        B = x_flat.size(0)
        h_out = torch.sigmoid(out['h_pred']).cpu().numpy().reshape(B, T_FRAMES, 128)
        m_out = torch.sigmoid(out['m_pred']).cpu().numpy().reshape(B, T_FRAMES, 128)
        for b in range(B):
            hd = h_out[b].sum(axis=1)  # onset density (T,)
            md = m_out[b].sum(axis=1)
            if hd.std() > 1e-6 and md.std() > 1e-6:
                metrics['inter_track_corr'].append(
                    float(np.corrcoef(hd, md)[0,1])
                )

        # 4. TC score
        tc = tc_penalty(out['z_parts']).item()
        metrics['tc_score'].append(tc)

        # 5. KL per component
        for comp in kl_per_comp:
            mu = out['enc']['mu'][comp]
            lv = out['enc']['lv'][comp]
            kl_c = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean().item()
            kl_per_comp[comp].append(kl_c)

# ── Финальные метрики ─────────────────────────────────────────
final = {k: float(np.mean(v)) for k, v in metrics.items()}
final['kl_per_component'] = {k: float(np.mean(v)) for k, v in kl_per_comp.items()}

print('\n' + '='*55)
print('  Structured Controller — Test Metrics')
print('='*55)
print(f'  z_recon_mse     : {final["z_recon_mse"]:.4f}  (↓ лучше)')
print(f'  energy_corr     : {final["energy_corr"]:.4f}  (↑ лучше)')
print(f'  inter_track_corr: {final["inter_track_corr"]:.4f}  (↑ лучше)')
print(f'  tc_score        : {final["tc_score"]:.4f}  (↓ лучше, меньше корреляции)')
print(f'\n  KL per component (> 0.1 = not collapsed):')
for comp, kl_v in final['kl_per_component'].items():
    bar = '█' * int(kl_v * 10)
    print(f'    z_{comp:<8}: {kl_v:.3f}  {bar}')
print('='*55)

with open(CKPT_DIR / 'test_metrics.json', 'w') as f:
    json.dump(final, f, indent=2)
print('Saved: test_metrics.json')

# ── Training curves ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
keys   = ['total','recon','kl','energy','role','dis']
titles = ['Total Loss','z Recon (MSE)','KL Divergence',
          'Energy Loss','Role Aux Loss','TC Penalty']

for ax, key, title in zip(axes.flat, keys, titles):
    ax.plot(history[f'train_{key}'], label='train', color='#3498db', lw=1.5)
    ax.plot(history[f'val_{key}'],   label='val',   color='#e74c3c', lw=1.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)

plt.suptitle('Structured Controller — Training Curves', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150)
plt.show()

# ── KL per component bar chart ────────────────────────────────
comp_kls = final['kl_per_component']
colors   = ['#9b59b6','#f39c12','#e74c3c','#2ecc71','#3498db']

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(comp_kls.keys(), comp_kls.values(),
              color=colors, edgecolor='white', width=0.5)
ax.axhline(0.1, color='gray', linestyle='--', lw=1, label='collapse threshold')
ax.set_title('KL Divergence per Latent Component\n'
             '(values > 0.1 indicate active, non-collapsed components)',
             fontsize=11)
ax.set_ylabel('Mean KL')
ax.legend()
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kl_per_component.png', dpi=150)
plt.show()

# ── PCA latent space ──────────────────────────────────────────
from sklearn.decomposition import PCA

z_all_parts = {k: [] for k in ['struct','energy','rhythm','harmony','melody']}
controller.eval()
with torch.no_grad():
    for _, x_flat, _ in test_loader:
        enc = controller.encode(x_flat.to(DEVICE))
        for k in z_all_parts:
            z_all_parts[k].append(enc['mu'][k].cpu().numpy())
        if sum(len(v) for v in z_all_parts.values()) > 5 * 1000:
            break

z_concat = np.concatenate(
    [np.concatenate(z_all_parts[k], 0) for k in z_all_parts], axis=1
)[:1000]
pca  = PCA(n_components=2)
z_2d = pca.fit_transform(z_concat)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(z_2d[:,0], z_2d[:,1],
                c=np.arange(len(z_2d)), cmap='viridis', alpha=0.4, s=8)
plt.colorbar(sc, label='Sample index')
ax.set_title('Structured Latent Space (PCA)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_pca.png', dpi=150)
plt.show()

print('All plots saved.')

print('='*60)
print('  Generated files')
print('='*60)

groups = {
    'Random generation':    list(OUTPUT_DIR.glob('generated_*.mid')),
    'Component control':    list(OUTPUT_DIR.glob('control_*.mid')),
    'Track swap':           list(OUTPUT_DIR.glob('swap_*.mid')),
    'Interpolation (MIDI)': list(OUTPUT_DIR.glob('interp_*.mid')),
    'Plots (PNG)':          list(OUTPUT_DIR.glob('*.png')),
    'Metrics (JSON)':       list(CKPT_DIR.glob('*.json')),
}

for group, files in groups.items():
    if not files: continue
    print(f'\n  {group}:')
    for f in sorted(files):
        kb = f.stat().st_size / 1024
        print(f'    {f.name:<45} {kb:5.1f} KB')

print('\n' + '='*60)
print('  Quick reference: how to use the model')
print('='*60)
print('''
  # 1. Generate random music
  z = controller.sample_structured_z(n=4, device=DEVICE)
  seqs = decode_z_to_midi(z, temperature=0.5)

  # 2. Encode your MIDI
  roll = extract_piano_roll('/path/to/your.mid')
  x = torch.from_numpy(roll.reshape(1,-1)).to(DEVICE)
  enc = controller.encode(x)
  z   = {k: enc['mu'][k] for k in enc['mu']}  # posterior mean

  # 3. Swap one component from another track
  z_mixed = controller.swap(z_A, z_B, component='melody')
  seqs = decode_z_to_midi(z_mixed)

  # 4. Smooth interpolation
  z_mid = controller.interpolate(z_A, z_B, alpha=0.5)
  seqs  = decode_z_to_midi(z_mid)

  # 5. Save result
  note_seq.sequence_proto_to_midi_file(seqs[0], 'output.mid')
''')

print(f'All files in: {OUTPUT_DIR}')