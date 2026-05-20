import random
import glob
from pathlib import Path
from typing import Optional
import numpy as np
import pretty_midi
import note_seq
import torch
from torch.utils.data import Dataset

BASS_PROGRAMS = set(range(32, 40))


def assign_role(inst):
    """Оригинальная функция из Colab"""
    if inst.is_drum:
        return 'rhythm'
    if inst.program in BASS_PROGRAMS:
        return 'harmony'
    if inst.notes and np.mean([n.pitch for n in inst.notes]) < 48:
        return 'harmony'
    return 'melody'


def midi_to_note_seq(midi_path: str):
    """MIDI файл → NoteSequence для Music VAE (оригинал)"""
    try:
        return note_seq.midi_file_to_note_sequence(midi_path)
    except Exception:
        return None


def extract_piano_roll(midi_path: str, fs: int = 8, clip_frames: int = 256) -> Optional[np.ndarray]:
    """Простая версия — без разделения на роли"""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    
    # Просто объединяем все инструменты
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)
    
    if not all_notes:
        return None
    
    # Создаем один трек со всеми нотами
    tmp = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = all_notes
    tmp.instruments.append(inst)
    
    roll = tmp.get_piano_roll(fs=fs)
    
    if roll.shape[1] < clip_frames:
        pad = np.zeros((128, clip_frames - roll.shape[1]))
        roll = np.hstack([roll, pad])
    else:
        roll = roll[:, :clip_frames]
    
    # Дублируем для трех каналов (чтобы сохранить интерфейс)
    roll = np.stack([roll, roll, roll], axis=-1).transpose(1, 0, 2)
    
    return roll.astype(np.float32)


def compute_energy(roll: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Оригинальная функция из Colab"""
    density = roll.sum(axis=(1, 2)) / (128 * 3)
    return (alpha * density + (1 - alpha) * density).astype(np.float32)


class ZDataset(Dataset):
    """Оригинальный класс Dataset из Colab"""
    def __init__(self, z_mv, rolls, energy):
        self.z_mv = torch.from_numpy(z_mv)
        self.x_flat = torch.from_numpy(rolls.reshape(len(rolls), -1))
        self.energy = torch.from_numpy(energy)

    def __len__(self):
        return len(self.z_mv)

    def __getitem__(self, idx):
        return self.z_mv[idx], self.x_flat[idx], self.energy[idx]


def prepare_datasets(midi_dir, musicvae, cfg, force_recompute=False):
    """Оригинальная логика подготовки данных из Colab"""
    Z_CACHE = cfg.processed_dir / 'musicvae_z_vectors.npy'
    ROLL_CACHE = cfg.processed_dir / 'piano_rolls.npy'
    ENERGY_CACHE = cfg.processed_dir / 'energy_profiles.npy'

    if not force_recompute and Z_CACHE.exists() and ROLL_CACHE.exists():
        print('Loading cached data...')
        z_musicvae_all = np.load(Z_CACHE)
        rolls_all = np.load(ROLL_CACHE)
        energy_all = np.load(ENERGY_CACHE)
        print(f'Loaded: {len(z_musicvae_all)} samples')
    else:
        midi_files = (glob.glob(str(midi_dir / '**/*.mid'), recursive=True) +
                      glob.glob(str(midi_dir / '**/*.midi'), recursive=True))
        random.shuffle(midi_files)
        midi_files = midi_files[:cfg.max_midi_files]
        print(f'Processing {len(midi_files)} MIDI files...')

        z_list, roll_list, energy_list = [], [], []
        skipped = 0
        BATCH_ENCODE = 32
        batch_seqs, batch_rolls = [], []

        from tqdm import tqdm
        for midi_path in tqdm(midi_files, desc='Encoding MIDI → z_musicvae'):
            ns = midi_to_note_seq(midi_path)
            roll = extract_piano_roll(midi_path)
            if ns is None or roll is None:
                skipped += 1
                continue

            batch_seqs.append(ns)
            batch_rolls.append(roll)

            if len(batch_seqs) >= BATCH_ENCODE:
                try:
                    z_batch = musicvae.encode(batch_seqs)
                    if isinstance(z_batch, list):
                        z_batch = np.stack(z_batch, axis=0)
                    for i in range(len(batch_seqs)):
                        z_list.append(z_batch[i])
                        roll_list.append(batch_rolls[i])
                        energy_list.append(compute_energy(batch_rolls[i]))
                except Exception:
                    pass
                batch_seqs, batch_rolls = [], []

        if batch_seqs:
            try:
                z_batch = musicvae.encode(batch_seqs)
                if isinstance(z_batch, list):
                    z_batch = np.stack(z_batch, axis=0)
                for i in range(len(batch_seqs)):
                    z_list.append(z_batch[i])
                    roll_list.append(batch_rolls[i])
                    energy_list.append(compute_energy(batch_rolls[i]))
            except Exception:
                pass

        print(f'Done. Accepted: {len(z_list)}, Skipped: {skipped}')

        z_musicvae_all = np.stack(z_list, axis=0).astype(np.float32)
        rolls_all = np.stack(roll_list, axis=0).astype(np.float32)
        energy_all = np.stack(energy_list, axis=0).astype(np.float32)

        np.save(Z_CACHE, z_musicvae_all)
        np.save(ROLL_CACHE, rolls_all)
        np.save(ENERGY_CACHE, energy_all)
        print(f'Saved to {cfg.processed_dir}')

    print(f'\nDataset shapes:')
    print(f'  z_musicvae: {z_musicvae_all.shape}   (N, 512)')
    print(f'  rolls:      {rolls_all.shape}   (N, T, 128, 3)')
    print(f'  energy:     {energy_all.shape}  (N, T)')

    N = len(z_musicvae_all)
    idx = np.random.permutation(N)
    n_train = int(0.85 * N)
    n_val = int(0.10 * N)

    train_ds = ZDataset(z_musicvae_all[idx[:n_train]], rolls_all[idx[:n_train]], energy_all[idx[:n_train]])
    val_ds = ZDataset(z_musicvae_all[idx[n_train:n_train + n_val]], rolls_all[idx[n_train:n_train + n_val]], energy_all[idx[n_train:n_train + n_val]])
    test_ds = ZDataset(z_musicvae_all[idx[n_train + n_val:]], rolls_all[idx[n_train + n_val:]], energy_all[idx[n_train + n_val:]])

    cfg.roll_flat_dim = rolls_all.shape[1] * 128 * 3
    cfg.t_frames = rolls_all.shape[1]

    return train_ds, val_ds, test_ds
