#!/usr/bin/env python3
"""
Главный скрипт для запуска всего пайплайна Structured Controller
Использование: python run_all.py [--step prepare|train|generate|evaluate|all]
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import random

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from src.music_vae_wrapper import MusicVAEWrapper
from src.data_utils import prepare_datasets, ZDataset
from src.model import StructuredController
from src.train import train
from src.generate import generate_samples, save_midi
from src.metrics import compute_all_metrics

def setup_environment():
    """Настройка окружения"""
    # Создаем директории
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Устанавливаем seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def step_prepare():
    """Шаг 1: Подготовка данных"""
    print("\n" + "="*60)
    print("STEP 1: Preparing data")
    print("="*60)
    
    device = setup_environment()
    
    # Загружаем Music VAE
    musicvae = MusicVAEWrapper(cfg.musicvae_model, cfg.processed_dir, cfg.batch_size)
    musicvae.load()
    
    # Подготавливаем датасеты
    train_ds, val_ds, test_ds = prepare_datasets(cfg.data_dir, musicvae, cfg)
    
    # Создаем DataLoaders
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **kw)
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **kw)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Roll flat dim: {cfg.roll_flat_dim}")
    
    # Сохраняем информацию о датасете
    dataset_info = {
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'roll_flat_dim': cfg.roll_flat_dim,
        't_frames': cfg.t_frames
    }
    import json
    with open(cfg.processed_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f)
    
    return train_loader, val_loader, test_loader

def step_train():
    """Шаг 2: Обучение модели"""
    print("\n" + "="*60)
    print("STEP 2: Training model")
    print("="*60)
    
    device = setup_environment()
    
    # Загружаем информацию о датасете
    import json
    with open(cfg.processed_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    cfg.roll_flat_dim = dataset_info['roll_flat_dim']
    cfg.t_frames = dataset_info['t_frames']
    
    # Создаем модель
    model = StructuredController(cfg, cfg.roll_flat_dim, cfg.t_frames).to(device)
    print(f"Controller parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Загружаем данные
    train_ds = ZDataset(
        np.load(cfg.processed_dir / 'musicvae_z_vectors.npy')[:dataset_info['train_size']],
        np.load(cfg.processed_dir / 'piano_rolls.npy')[:dataset_info['train_size']],
        np.load(cfg.processed_dir / 'energy_profiles.npy')[:dataset_info['train_size']]
    )
    val_ds = ZDataset(
        np.load(cfg.processed_dir / 'musicvae_z_vectors.npy')[dataset_info['train_size']:dataset_info['train_size']+dataset_info['val_size']],
        np.load(cfg.processed_dir / 'piano_rolls.npy')[dataset_info['train_size']:dataset_info['train_size']+dataset_info['val_size']],
        np.load(cfg.processed_dir / 'energy_profiles.npy')[dataset_info['train_size']:dataset_info['train_size']+dataset_info['val_size']]
    )
    
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **kw)
    
    # Обучаем
    history = train(model, train_loader, val_loader, cfg, device)
    
    print("Training completed!")
    return model, history

def step_generate():
    """Шаг 3: Генерация музыки"""
    print("\n" + "="*60)
    print("STEP 3: Generating music")
    print("="*60)
    
    device = setup_environment()
    
    # Загружаем информацию
    import json
    with open(cfg.processed_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    cfg.roll_flat_dim = dataset_info['roll_flat_dim']
    cfg.t_frames = dataset_info['t_frames']
    
    # Загружаем модель
    model = StructuredController(cfg, cfg.roll_flat_dim, cfg.t_frames).to(device)
    checkpoint = torch.load(cfg.checkpoint_dir / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded checkpoint: epoch={checkpoint['epoch']}, val={checkpoint['best_val']:.4f}")
    
    # Загружаем Music VAE
    musicvae = MusicVAEWrapper(cfg.musicvae_model, cfg.processed_dir, cfg.batch_size)
    musicvae.load()
    
    # Функция декодирования для нашей модели
    def decode_z_to_midi(z_parts, temperature=0.5, length=32):
        with torch.no_grad():
            z_mv_hat = model.decode_to_musicvae_z(z_parts)
        z_np = z_mv_hat.cpu().numpy()
        return musicvae.decode(z_np, length=length, temperature=temperature)
    
    # 1. Генерация из prior
    print("\nGenerating from prior...")
    for i in range(6):
        z_random = model.sample_structured_z(1, device)
        seqs = decode_z_to_midi(z_random, temperature=0.5)
        save_midi(seqs[0], cfg.output_dir / 'generated' / f'generated_{i+1:02d}.mid')
    
    # 2. Component control experiment
    print("Component control experiment...")
    torch.manual_seed(7)
    z_base = model.sample_structured_z(1, device)
    
    for comp in ['struct', 'energy', 'rhythm', 'harmony', 'melody']:
        for var_idx in range(4):
            z_varied = {k: v.clone() for k, v in z_base.items()}
            z_varied[comp] = torch.randn_like(z_base[comp]) * 1.5
            seqs = decode_z_to_midi(z_varied, temperature=0.4)
            save_midi(seqs[0], cfg.output_dir / 'control' / f'control_{comp}_var{var_idx+1}.mid')
    
    # 3. Interpolation (требует реальных MIDI, пропускаем если нет)
    print("Check outputs directory for generated files!")
    print(f"Output directory: {cfg.output_dir}")

def step_evaluate():
    """Шаг 4: Оценка модели"""
    print("\n" + "="*60)
    print("STEP 4: Evaluating model")
    print("="*60)
    
    device = setup_environment()
    
    # Загружаем информацию
    import json
    with open(cfg.processed_dir / 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    cfg.roll_flat_dim = dataset_info['roll_flat_dim']
    cfg.t_frames = dataset_info['t_frames']
    
    # Загружаем модель
    model = StructuredController(cfg, cfg.roll_flat_dim, cfg.t_frames).to(device)
    checkpoint = torch.load(cfg.checkpoint_dir / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Загружаем тестовые данные
    test_ds = ZDataset(
        np.load(cfg.processed_dir / 'musicvae_z_vectors.npy')[-dataset_info['test_size']:],
        np.load(cfg.processed_dir / 'piano_rolls.npy')[-dataset_info['test_size']:],
        np.load(cfg.processed_dir / 'energy_profiles.npy')[-dataset_info['test_size']:]
    )
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # Вычисляем метрики
    metrics = compute_all_metrics(model, test_loader, cfg, device)
    
    print("\n" + "="*55)
    print("  Structured Controller — Test Metrics")
    print("="*55)
    print(f"  z_recon_mse     : {metrics['z_recon_mse']:.4f}")
    print(f"  energy_corr     : {metrics['energy_corr']:.4f}")
    print(f"  inter_track_corr: {metrics['inter_track_corr']:.4f}")
    print(f"  tc_score        : {metrics['tc_score']:.4f}")
    
    # Сохраняем метрики
    with open(cfg.checkpoint_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Сохраняем визуализации
    from src.metrics import plot_training_curves, plot_kl_per_component
    plot_training_curves(cfg.checkpoint_dir / 'history.json', cfg.output_dir / 'plots')
    plot_kl_per_component(metrics['kl_per_component'], cfg.output_dir / 'plots')

def main():
    parser = argparse.ArgumentParser(description='Run Structured Controller pipeline')
    parser.add_argument('--step', type=str, default='all',
                       choices=['prepare', 'train', 'generate', 'evaluate', 'all'],
                       help='Which step to run')
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
        print("Running full pipeline...")
        step_prepare()
        step_train()
        step_generate()
        step_evaluate()
        print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()