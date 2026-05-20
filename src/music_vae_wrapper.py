import os
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
import note_seq
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel


class MusicVAEWrapper:
    """Обертка для работы с Music VAE (оригинальный код из Colab)"""
    
    def __init__(self, model_name: str, checkpoint_dir: Path, batch_size: int = 32):
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.batch_size = batch_size
        self.model = None
        
    def download_if_needed(self):
        """Скачивает веса модели если их нет (как в Colab)"""
        ckpt_path = self.checkpoint_dir / f"{self.model_name}.tar"
        extracted_dir = self.checkpoint_dir / self.model_name
        
        if not extracted_dir.exists():
            print(f'Downloading Music VAE weights ({self.model_name})...')
            url = f"https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/{self.model_name}.tar"
            urllib.request.urlretrieve(url, ckpt_path)
            with tarfile.open(ckpt_path, 'r') as tar:
                tar.extractall(self.checkpoint_dir)
            os.remove(ckpt_path)
            print('Downloaded.')
        return extracted_dir
    
    def load(self):
        """Загружает модель Music VAE"""
        extracted_dir = self.download_if_needed()
        print('Loading Music VAE...')
        musicvae_config = configs.CONFIG_MAP[self.model_name]
        self.model = TrainedModel(
            config=musicvae_config,
            batch_size=self.batch_size,
            checkpoint_dir_or_path=str(extracted_dir),
        )
        print('Music VAE loaded successfully.')
        return self.model
    
    def encode(self, sequences):
        """Кодирует NoteSequences в z-векторы"""
        if self.model is None:
            self.load()
        z, _, _ = self.model.encode(sequences)
        if isinstance(z, list):
            z = np.stack(z, axis=0)
        return z.astype(np.float32)
    
    def decode(self, z_vectors, length=32, temperature=0.5):
        """Декодирует z-векторы в NoteSequences"""
        if self.model is None:
            self.load()
        return self.model.decode(z_vectors, length=length, temperature=temperature)
    
    def sample(self, n=4, length=32, temperature=0.5):
        """Генерирует случайные семплы из prior"""
        if self.model is None:
            self.load()
        return self.model.sample(n=n, length=length, temperature=temperature)
