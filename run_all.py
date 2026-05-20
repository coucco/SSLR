#!/usr/bin/env python3
"""
Главный скрипт для запуска всего пайплайна Structured Controller
Использование: 
    python run_all.py --step prepare   # Только подготовка данных
    python run_all.py --step train     # Только обучение
    python run_all.py --step generate  # Только генерация
    python run_all.py --step evaluate  # Только оценка
    python run_all.py --step all       # Полный пайплайн
"""

import argparse
import sys
import json
import random
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import tensorflow as tf
import note_seq

from src.data_utils import (
    prepare_datasets, ZDataset, extract_piano_roll,
    midi_to_note_seq, compute_energy
)
from src.model import StructuredController
from src.train import train
from src.generate import decode_z_to_midi, save_midi
from src.metrics import compute_all_metrics, plot_training_curves, plot_kl_per_component

# Отключаем лишние предупреждения
tf.get_logger().setLevel('ERROR')

# В начале файла, после импортов
class ConfigWrapper:
    def __init__(self, cfg_dict, processed_dir):
        self.__dict__.update(cfg_dict)
        self.processed_dir = processed_dir

# ============================================================
#  НАСТРОЙКИ
# ============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Пути
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data' / 'musicvae_data'
CKPT_DIR = PROJECT_ROOT / 'data' / 'checkpoints' / 'structured_controller'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
MIDI_DIR = PROJECT_ROOT / 'data' / 'raw_midi'

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
    'batch_size': 128,
    'lr': 3e-4,
    'n_epochs': 60,
    'seed': 42,
    'max_midi_files': 3000,
    'num_workers': 4,
    'checkpoint_dir': CKPT_DIR,
    't_frames': None,      # Заполнится при загрузке данных
    'roll_flat_dim': None, # Заполнится при загрузке данных
}

CFG['z_total'] = (CFG['z_struct_dim'] + CFG['z_energy_dim'] +
                  CFG['z_rhythm_dim'] + CFG['z_harmony_dim'] + CFG['z_melody_dim'])

# Устанавливаем seed
random.seed(CFG['seed'])
np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])

# ============================================================
#  Music VAE ЗАГРУЗКА (остается здесь, так как это внешняя зависимость)
# ============================================================

def load_music_vae():
    """Загружает Music VAE модель"""
    from magenta.models.music_vae import configs
    from magenta.models.music_vae.trained_model import TrainedModel
    import tarfile
    import urllib.request
    
    MODEL_NAME = CFG['musicvae_model']
    musicvae_checkpoint_dir = PROJECT_ROOT / 'data' / 'checkpoints' / 'music_vae'
    musicvae_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = musicvae_checkpoint_dir / f"{MODEL_NAME}.tar"
    if not ckpt_path.exists():
        print(f'Downloading Music VAE weights ({MODEL_NAME})...')
        url = f"https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/{MODEL_NAME}.tar"
        urllib.request.urlretrieve(url, ckpt_path)
        with tarfile.open(ckpt_path, 'r') as tar:
            tar.extractall(musicvae_checkpoint_dir)

    ckpt_dirs = glob.glob(str(musicvae_checkpoint_dir / f'{MODEL_NAME}*/'))
    MUSICVAE_CKPT = ckpt_dirs[0] if ckpt_dirs else str(musicvae_checkpoint_dir / MODEL_NAME)

    print('Loading Music VAE...')
    musicvae_config = configs.CONFIG_MAP[MODEL_NAME]
    musicvae = TrainedModel(
        config=musicvae_config,
        batch_size=CFG['batch_size'],
        checkpoint_dir_or_path=MUSICVAE_CKPT,
    )
    print('Music VAE loaded successfully.')
    return musicvae

# ============================================================
#  ШАГИ ПАЙПЛАЙНА
# ============================================================

def step_prepare():
    """Шаг 1: Подготовка данных (кодирование MIDI → z-векторы)"""
    print("\n" + "=" * 60)
    print("STEP 1: Preparing data (encoding MIDI → z_musicvae)")
    print("=" * 60)

    musicvae = load_music_vae()
    
    # В функции step_prepare(), перед вызовом prepare_datasets:
    cfg_obj = ConfigWrapper(CFG, DATA_DIR)
    train_ds, val_ds, test_ds = prepare_datasets(MIDI_DIR, musicvae, cfg_obj)
    
    # Сохраняем размерности
    dataset_info = {
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'roll_flat_dim': CFG['roll_flat_dim'],
        't_frames': CFG['t_frames'],
    }
    with open(DATA_DIR / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f)

    print(f'\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    print(f'Roll flat dim: {CFG["roll_flat_dim"]}')
    print(f'T frames: {CFG["t_frames"]}')

def step_train():
    """Шаг 2: Обучение StructuredController (использует train.py)"""
    print("\n" + "=" * 60)
    print("STEP 2: Training StructuredController")
    print("=" * 60)

    # Загружаем информацию о датасете
    with open(DATA_DIR / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    CFG['roll_flat_dim'] = dataset_info['roll_flat_dim']
    CFG['t_frames'] = dataset_info['t_frames']

    # Загружаем данные из кэша
    z_musicvae_all = np.load(DATA_DIR / 'musicvae_z_vectors.npy')
    rolls_all = np.load(DATA_DIR / 'piano_rolls.npy')
    energy_all = np.load(DATA_DIR / 'energy_profiles.npy')

    train_ds = ZDataset(
        z_musicvae_all[:dataset_info['train_size']],
        rolls_all[:dataset_info['train_size']],
        energy_all[:dataset_info['train_size']]
    )
    val_ds = ZDataset(
        z_musicvae_all[dataset_info['train_size']:dataset_info['train_size'] + dataset_info['val_size']],
        rolls_all[dataset_info['train_size']:dataset_info['train_size'] + dataset_info['val_size']],
        energy_all[dataset_info['train_size']:dataset_info['train_size'] + dataset_info['val_size']]
    )

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'])

    # Создаем модель
    model = StructuredController(CFG, CFG['roll_flat_dim'], CFG['t_frames']).to(DEVICE)
    print(f'Controller parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Обучаем (функция из train.py)
    history = train(model, train_loader, val_loader, CFG, DEVICE)

    print(f'\n✅ Training completed!')
    print(f'Best model saved to: {CKPT_DIR / "best.pt"}')

def step_generate():
    """Шаг 3: Генерация музыки (использует generate.py)"""
    print("\n" + "=" * 60)
    print("STEP 3: Generating music")
    print("=" * 60)

    # Загружаем Music VAE
    musicvae = load_music_vae()

    # Загружаем информацию о датасете
    with open(DATA_DIR / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    CFG['roll_flat_dim'] = dataset_info['roll_flat_dim']
    CFG['t_frames'] = dataset_info['t_frames']

    # Загружаем обученную модель
    model = StructuredController(CFG, CFG['roll_flat_dim'], CFG['t_frames']).to(DEVICE)
    checkpoint = torch.load(CKPT_DIR / 'best.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f'Loaded checkpoint: epoch={checkpoint["epoch"]}, val_loss={checkpoint["best_val"]:.4f}')

    # Генерация из prior
    print('\nGenerating from prior...')
    for i in range(6):
        z_random = model.sample_structured_z(1, DEVICE)
        seqs = decode_z_to_midi(z_random, musicvae, model, temperature=0.5)
        save_midi(seqs[0], OUTPUT_DIR / 'generated' / f'generated_{i+1:02d}.mid')

    print(f'\n✅ Generation completed!')
    print(f'Outputs saved to: {OUTPUT_DIR}')

def step_evaluate():
    """Шаг 4: Оценка модели (использует metrics.py)"""
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating model")
    print("=" * 60)

    # Загружаем информацию о датасете
    with open(DATA_DIR / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    CFG['roll_flat_dim'] = dataset_info['roll_flat_dim']
    CFG['t_frames'] = dataset_info['t_frames']

    # Загружаем модель
    model = StructuredController(CFG, CFG['roll_flat_dim'], CFG['t_frames']).to(DEVICE)
    checkpoint = torch.load(CKPT_DIR / 'best.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Загружаем тестовые данные
    z_musicvae_all = np.load(DATA_DIR / 'musicvae_z_vectors.npy')
    rolls_all = np.load(DATA_DIR / 'piano_rolls.npy')
    energy_all = np.load(DATA_DIR / 'energy_profiles.npy')

    test_ds = ZDataset(
        z_musicvae_all[-dataset_info['test_size']:],
        rolls_all[-dataset_info['test_size']:],
        energy_all[-dataset_info['test_size']:]
    )
    test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False)

    # Вычисляем метрики (функция из metrics.py)
    metrics = compute_all_metrics(model, test_loader, CFG, DEVICE)

    print('\n' + '=' * 55)
    print('  Structured Controller — Test Metrics')
    print('=' * 55)
    print(f'  z_recon_mse     : {metrics["z_recon_mse"]:.4f}  (↓ лучше)')
    print(f'  energy_corr     : {metrics["energy_corr"]:.4f}  (↑ лучше)')
    print(f'  inter_track_corr: {metrics["inter_track_corr"]:.4f}  (↑ лучше)')
    print(f'  tc_score        : {metrics["tc_score"]:.4f}  (↓ лучше)')

    print(f'\n  KL per component:')
    for comp, kl_v in metrics['kl_per_component'].items():
        bar = '█' * int(kl_v * 10)
        print(f'    {comp:<10}: {kl_v:.3f}  {bar}')

    # Сохраняем метрики
    with open(CKPT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Рисуем графики (функции из metrics.py)
    if (CKPT_DIR / 'history.json').exists():
        plot_training_curves(CKPT_DIR / 'history.json', OUTPUT_DIR / 'plots')
        plot_kl_per_component(metrics['kl_per_component'], OUTPUT_DIR / 'plots')
        print(f'\n✅ Plots saved to: {OUTPUT_DIR / "plots"}')

    print(f'\n✅ Evaluation completed!')

def step_all():
    """Шаг 5: Полный пайплайн"""
    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE")
    print("=" * 60)

    step_prepare()
    step_train()
    step_generate()
    step_evaluate()

    print("\n" + "=" * 60)
    print("✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Structured Controller Pipeline')
    parser.add_argument('--step', type=str, default='all',
                       choices=['prepare', 'train', 'generate', 'evaluate', 'all'],
                       help='Step to run')
    args = parser.parse_args()

    if args.step == 'prepare':
        step_prepare()
    elif args.step == 'train':
        step_train()
    elif args.step == 'generate':
        step_generate()
    elif args.step == 'evaluate':
        step_evaluate()
    elif args.step == 'all':
        step_all()

if __name__ == "__main__":
    main()
