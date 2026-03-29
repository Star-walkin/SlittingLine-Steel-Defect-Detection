"""
draem_model/dataset.py
DRAEM 训练数据集：
  返回 (normal_tensor, anomalous_tensor, mask_tensor) 三元组
  - normal:   干净图，归一化 [-1,1]
  - anomalous: 加入伪缺陷的图，归一化 [-1,1]
  - mask:     缺陷区域二值 mask，值域 [0,1]（float32）
"""

import glob
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch

from draem_model.anomaly_gen import gen_anomaly_with_mask


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """自适应背景拍平：高斯估计背景后以 128 为中心相减。"""
    ksize = max(51, gray.shape[1] // 10) | 1
    bg = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    return np.clip(gray.astype(np.float32) - bg + 128.0, 0, 255).astype(np.uint8)


class DRAEMDataset(Dataset):
    def __init__(self, data_dir: str, img_size: int = 256):
        self.img_size = img_size
        pattern = os.path.join(data_dir, "train", "good", "*.*")
        self.images = sorted(glob.glob(pattern))
        if not self.images:
            raise FileNotFoundError(
                f"No training images found in: {os.path.join(data_dir, 'train', 'good')}"
            )
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print(f"  DRAEMDataset: {len(self.images)} images from {data_dir}")

    def __getitem__(self, idx):
        gray = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read: {self.images[idx]}")

        if gray.shape[1] > 20:
            gray = gray[:, 10:-10]

        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gray = flatten_background(gray)

        anomalous, mask = gen_anomaly_with_mask(gray)

        normal_t = self._to_tensor(Image.fromarray(gray))
        anomalous_t = self._to_tensor(Image.fromarray(anomalous))
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return normal_t, anomalous_t, mask_t

    def __len__(self):
        return len(self.images)


def get_dataloader(data_dir: str, img_size: int = 256, batch_size: int = 8) -> DataLoader:
    ds = DRAEMDataset(data_dir, img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
