import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from constant import IEMOCAP_DIR

EMOTIONS = ['ang', 'exc', 'neu', 'sad']
EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}


def compute_audio_stats(data_root):
    """
    Compute per-feature mean/std from training set (non-padded frames only).
    Returns mean (34,) and std (34,) as float32 numpy arrays.
    """
    audio_root = os.path.join(data_root, 'audio', 'train')
    all_frames = []
    for emotion in EMOTIONS:
        audio_dir = os.path.join(audio_root, emotion)
        if not os.path.exists(audio_dir):
            continue
        for path in sorted(glob.glob(f"{audio_dir}/*.npy")):
            audio = np.load(path)                    # (100, 34)
            mask  = np.any(audio != 0, axis=1)       # real frames only
            if mask.any():
                all_frames.append(audio[mask])
    all_frames = np.vstack(all_frames)               # (N_real, 34)
    mean = all_frames.mean(axis=0).astype('float32')
    std  = all_frames.std(axis=0).astype('float32')
    std[std < 1e-8] = 1.0                            # avoid /0
    print(f"[audio stats] computed from {len(all_frames)} frames  "
          f"mean_abs={np.abs(mean).mean():.4f}  std_mean={std.mean():.4f}")
    return mean, std

class IEMOCAPDataset(Dataset):
    def __init__(self, data_root, split, audio_mean=None, audio_std=None):
        self.samples    = []
        self.audio_mean = audio_mean
        self.audio_std  = audio_std
        audio_root = os.path.join(data_root, 'audio', split)

        for emotion in EMOTIONS:
            audio_dir = os.path.join(audio_root, emotion)
            if not os.path.exists(audio_dir):
                continue

            for audio_path in glob.glob(f"{audio_dir}/*.npy"):
                text_path = audio_path.replace('/audio/', '/text/')
                if not os.path.exists(text_path):
                    print(f"Warning: missing text for {audio_path}")
                    continue
                self.samples.append((audio_path, text_path, EMOTION2IDX[emotion]))

        print(f"[{split}] {len(self.samples)} samples loaded"
              f"  audio_normalize={'yes' if audio_mean is not None else 'no'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text_path, label = self.samples[idx]
        audio = np.load(audio_path)                      # (100, 34)
        text  = torch.FloatTensor(np.load(text_path))    # (500, 300)
        label = torch.tensor(label, dtype=torch.long)

        if self.audio_mean is not None:
            mask = np.any(audio != 0, axis=1)
            audio[mask] = (audio[mask] - self.audio_mean) / self.audio_std

        audio = torch.FloatTensor(audio)

        # 真實 token 數（非全零的 row）
        text_len = int((text.abs().sum(dim=1) > 0).sum())
        text_len = max(text_len, 1)   # 至少 1，避免 pack 出錯

        return audio, text, label, text_len


def get_dataloaders(data_root, batch_size=64, normalize_audio=True):
    audio_mean = audio_std = None
    if normalize_audio:
        audio_mean, audio_std = compute_audio_stats(data_root)

    train_dataset = IEMOCAPDataset(data_root, split='train',
                                   audio_mean=audio_mean, audio_std=audio_std)
    test_dataset  = IEMOCAPDataset(data_root, split='test',
                                   audio_mean=audio_mean, audio_std=audio_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=4)
    return train_loader, test_loader

# 在 dataset.py 最後加這個函式

class IEMOCAPAudioDataset(Dataset):
    def __init__(self, data_root, split, mean=None, std=None, impro_only=False):
        self.samples = []
        self.mean = mean   # (34,) float32 or None
        self.std  = std    # (34,) float32 or None
        audio_root = os.path.join(data_root, 'audio', split)

        for emotion in EMOTIONS:
            audio_dir = os.path.join(audio_root, emotion)
            if not os.path.exists(audio_dir):
                continue
            for audio_path in sorted(glob.glob(f"{audio_dir}/*.npy")):
                # impro_only=True 時只保留檔名含 'impro' 的樣本
                if impro_only and 'impro' not in os.path.basename(audio_path):
                    continue
                self.samples.append((audio_path, EMOTION2IDX[emotion]))

        mode = 'impro only' if impro_only else 'impro+script'
        print(f"[audio/{split}] {len(self.samples)} samples loaded"
              f"  normalize={'yes' if mean is not None else 'no'}"
              f"  data={mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        audio = np.load(audio_path)                      # (100, 34)

        if self.mean is not None:
            # Normalize real frames; keep padding rows as zero
            mask = np.any(audio != 0, axis=1)
            audio[mask] = (audio[mask] - self.mean) / self.std

        audio = torch.FloatTensor(audio)
        label = torch.tensor(label, dtype=torch.long)
        return audio, label


def get_audio_dataloaders(data_root, batch_size=64, normalize=True, impro_only=False):
    mean = std = None
    if normalize:
        mean, std = compute_audio_stats(data_root)

    train_dataset = IEMOCAPAudioDataset(data_root, split='train', mean=mean, std=std,
                                        impro_only=impro_only)
    test_dataset  = IEMOCAPAudioDataset(data_root, split='test',  mean=mean, std=std,
                                        impro_only=False)   # test 永遠用完整 session 5

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=4)
    return train_loader, test_loader

if __name__ == '__main__':
    DATA_ROOT = f"{IEMOCAP_DIR}/processed_paper"
    train_loader, test_loader = get_dataloaders(DATA_ROOT)

    for audio, text, label, text_len in train_loader:   # ← 加 text_len
        print("audio:   ", audio.shape)      # (64, 100, 34)
        print("text:    ", text.shape)       # (64, 500, 300)
        print("label:   ", label.shape)      # (64,)
        print("text_len:", text_len.shape)   # (64,)
        print("text_len sample:", text_len[:5])
        break