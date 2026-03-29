"""
eff_ad/dataset.py
仅使用正常样本（train/good），与 EfficientAD 一致；背景拍平与 det_model 一致。
"""

import glob
import os
import sys
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def flatten_background(gray: np.ndarray) -> np.ndarray:
    ksize = max(51, gray.shape[1] // 10) | 1
    bg = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    return np.clip(gray.astype(np.float32) - bg + 128.0, 0, 255).astype(np.uint8)


class NormalOnlyDataset(Dataset):
    """仅正常样本，用于 Student-Teacher 训练。"""

    def __init__(self, data_dir: str, img_size: int = 256):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pattern = os.path.join(data_dir, "train", "good", "*.*")
        self.images = sorted(glob.glob(pattern))
        if not self.images:
            raise FileNotFoundError(f"No images in {os.path.join(data_dir, 'train', 'good')}")
        print(f"  eff_ad Dataset: {len(self.images)} images from {data_dir}")

    def __getitem__(self, idx):
        gray = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read {self.images[idx]}")
        if gray.shape[1] > 20:
            gray = gray[:, 10:-10]
        gray = flatten_background(gray)
        x = self.transform(Image.fromarray(gray))
        return x, os.path.basename(self.images[idx])

    def __len__(self):
        return len(self.images)


def get_dataloader(data_dir: str, img_size: int = 256, batch_size: int = 8) -> DataLoader:
    ds = NormalOnlyDataset(data_dir, img_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
    )
