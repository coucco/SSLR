import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Пути
    project_root: Path = Path(__file__).parent
    data_dir: Path = Path("./data/raw_midi")
    processed_dir: Path = Path("./data/processed")
    checkpoint_dir: Path = Path("./data/checkpoints/structured_controller")
    output_dir: Path = Path("./outputs")
    
    # Music VAE
    musicvae_z_dim: int = 512
    musicvae_model: str = "cat-mel_2bar_big"  # or 'hierdec-mel_16bar'
    
    # Structured latent dimensions (sum = 512)
    z_struct_dim: int = 128
    z_energy_dim: int = 64
    z_rhythm_dim: int = 100
    z_harmony_dim: int = 110
    z_melody_dim: int = 110
    
    # Network
    hidden_dim: int = 1024
    
    # Loss weights
    lambda_recon: float = 1.0
    lambda_kl: float = 0.5
    lambda_energy: float = 2.0
    lambda_role: float = 1.5
    lambda_dis: float = 0.3
    kl_warmup: int = 10
    
    # Training
    batch_size: int = 128
    lr: float = 3e-4
    n_epochs: int = 60
    seed: int = 42
    
    # Data
    max_midi_files: int = 3000
    num_workers: int = 4
    
    def __post_init__(self):
        # Создаем все необходимые директории
        for d in [self.processed_dir, self.checkpoint_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Вычисляем общую размерность
        self.z_total = (self.z_struct_dim + self.z_energy_dim + 
                       self.z_rhythm_dim + self.z_harmony_dim + self.z_melody_dim)
        
        # Для совместимости с кодом
        self.roll_flat_dim = None  # Заполнится при загрузке данных
        self.t_frames = None

# Глобальный экземпляр конфига
cfg = Config()