import random
import glob
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pretty_midi
import note_seq
import torch
from torch.utils.data import Dataset, DataLoader

# Константы
BASS_PROGRAMS = set(range(32, 40))

def assign_role(inst):
    """Определяет роль инструмента (ритм/гармония/мелодия)"""
    if inst.is_drum:
        return 'rhythm'
    if inst.program in BASS_PROGRAMS:
        return 'harmony'
    if inst.notes and np.mean([n.pitch for n in inst.notes]) < 48:
        return 'harmony'
    return 'melody'

def midi_to_note_seq(midi_path: str):
    """MIDI файл → NoteSequence"""
    try:
        return note_seq.midi_file_to_note_sequence(midi_path)
    except Exception:
        return None

def extract_piano_roll(midi_path: str, fs: int = 8, clip_frames: int = 256) -> Optional[np.ndarray]:
    """Извлекает piano-roll (clip_frames, 128, 3)"""
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
            roll = tmp.get_piano_roll(fs=fs)
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
    ], axis=-1).transpose(1, 0, 2)
    
    return stack.astype(np.float32)

def compute_energy(roll: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """roll: (T, 128, 3) → energy (T,)"""
    density = roll.sum(axis=(1, 2)) / (128 * 3)
    return (alpha * density + (1 - alpha) * density).astype(np.float32)

class ZDataset(Dataset):
    """Датасет для обучения"""
    def __init__(self, z_mv, rolls, energy):
        self.z_mv = torch.from_numpy(z_mv)
        self.x_flat = torch.from_numpy(rolls.reshape(len(rolls), -1))
        self.energy = torch.from_numpy(energy)
    
    def __len__(self):
        return len(self.z_mv)
    
    def __getitem__(self, idx):
        return self.z_mv[idx], self.x_flat[idx], self.energy[idx]

def prepare_datasets(midi_dir: Path, musicvae, cfg, force_recompute=False):
    """Подготавливает датасеты из MIDI файлов"""
    
    # Кэш файлы
    z_cache = cfg.processed_dir / 'musicvae_z_vectors.npy'
    roll_cache = cfg.processed_dir / 'piano_rolls.npy'
    energy_cache = cfg.processed_dir / 'energy_profiles.npy'
    
    if not force_recompute and z_cache.exists() and roll_cache.exists():
        print("Loading cached data...")
        z_musicvae_all = np.load(z_cache)
        rolls_all = np.load(roll_cache)
        energy_all = np.load(energy_cache)
        print(f"Loaded: {len(z_musicvae_all)} samples")
    else:
        # Собираем MIDI файлы
        midi_files = (glob.glob(str(midi_dir / '**/*.mid'), recursive=True) +
                     glob.glob(str(midi_dir / '**/*.midi'), recursive=True))
        random.shuffle(midi_files)
        midi_files = midi_files[:cfg.max_midi_files]
        print(f'Processing {len(midi_files)} MIDI files...')
        
        z_list, roll_list, energy_list = [], [], []
        batch_seqs, batch_rolls = [], []
        
        from tqdm import tqdm
        for midi_path in tqdm(midi_files, desc='Encoding MIDI'):
            ns = midi_to_note_seq(midi_path)
            roll = extract_piano_roll(midi_path)
            if ns is None or roll is None:
                continue
            
            batch_seqs.append(ns)
            batch_rolls.append(roll)
            
            if len(batch_seqs) >= 32:
                try:
                    z_batch = musicvae.encode(batch_seqs)
                    for i in range(len(batch_seqs)):
                        z_list.append(z_batch[i])
                        roll_list.append(batch_rolls[i])
                        energy_list.append(compute_energy(batch_rolls[i]))
                except Exception:
                    pass
                batch_seqs, batch_rolls = [], []
        
        # Обработка остатка
        if batch_seqs:
            try:
                z_batch = musicvae.encode(batch_seqs)
                for i in range(len(batch_seqs)):
                    z_list.append(z_batch[i])
                    roll_list.append(batch_rolls[i])
                    energy_list.append(compute_energy(batch_rolls[i]))
            except Exception:
                pass
        
        z_musicvae_all = np.stack(z_list, axis=0).astype(np.float32)
        rolls_all = np.stack(roll_list, axis=0).astype(np.float32)
        energy_all = np.stack(energy_list, axis=0).astype(np.float32)
        
        np.save(z_cache, z_musicvae_all)
        np.save(roll_cache, rolls_all)
        np.save(energy_cache, energy_all)
        print(f"Saved to {cfg.processed_dir}")
    
    # Разделение на train/val/test
    N = len(z_musicvae_all)
    idx = np.random.permutation(N)
    n_train = int(0.85 * N)
    n_val = int(0.10 * N)
    
    train_ds = ZDataset(z_musicvae_all[idx[:n_train]], 
                       rolls_all[idx[:n_train]], 
                       energy_all[idx[:n_train]])
    val_ds = ZDataset(z_musicvae_all[idx[n_train:n_train+n_val]], 
                     rolls_all[idx[n_train:n_train+n_val]], 
                     energy_all[idx[n_train:n_train+n_val]])
    test_ds = ZDataset(z_musicvae_all[idx[n_train+n_val:]], 
                      rolls_all[idx[n_train+n_val:]], 
                      energy_all[idx[n_train+n_val:]])
    
    # Сохраняем размерности для модели
    cfg.roll_flat_dim = rolls_all.shape[1] * 128 * 3
    cfg.t_frames = rolls_all.shape[1]
    
    return train_ds, val_ds, test_ds