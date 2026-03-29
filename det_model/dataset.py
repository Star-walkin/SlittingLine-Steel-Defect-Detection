"""
det_model/dataset.py
- 与 seg_model_train 数据流一致，独立维护
- 所有相机统一应用背景拍平（flatten_background）
- 合成异常：条状 scar_creat + 点状 add_stain（深色或浅色）
"""

import glob
import os
import sys
import random
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

_SEG_TRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seg_model_train"
)
if _SEG_TRAIN_DIR not in sys.path:
    sys.path.insert(0, _SEG_TRAIN_DIR)

from scar_ano import scar_creat
from scar_anomaly import add_stain


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """自适应背景拍平：高斯模糊估计渐变背景，减去后以 128 为中心。"""
    ksize = max(51, gray.shape[1] // 10) | 1
    bg = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    return np.clip(gray.astype(np.float32) - bg + 128.0, 0, 255).astype(np.uint8)


class NormalDataset(Dataset):
    """
    带钢正常图像数据集，用于训练 SimpleAD。
    - 输入（anomaly_image）：合成缺陷图像（条状 + 点状）
    - 目标（normal_image）：干净的正常图像
    - 当 preprocessed=False（默认）：读图后做 crop 10px + flatten_background（高斯 -bg+128）
    - 当 preprocessed=True：图像已由 prepare_dataset_det.py 做过与推理一致的 FFT+拍平，仅做 normal/anomaly 合成与 transform
    """

    def __init__(self, data_dir: str, img_size: int = 256, preprocessed: bool = False):
        self.img_size = img_size
        self.preprocessed = preprocessed
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pattern = os.path.join(data_dir, "train", "good", "*.*")
        self.images = sorted(glob.glob(pattern))
        if not self.images:
            raise FileNotFoundError(
                f"No training images in: {os.path.join(data_dir, 'train', 'good')}"
            )
        print(f"  Dataset: {len(self.images)} images from {data_dir} (preprocessed={preprocessed})")

    def __getitem__(self, idx):
        gray = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read: {self.images[idx]}")

        if not self.preprocessed:
            if gray.shape[1] > 20:
                gray = gray[:, 10:-10]
            gray = flatten_background(gray)

        normal_np = gray.copy()
        anomaly_np = gray.copy()

        add_strip = random.random() < 0.5
        add_point = random.random() < 0.5
        if add_strip:
            anomaly_np = scar_creat(anomaly_np.copy())
        if add_point:
            # add_stain 需要 BGR，用灰度复制 3 通道再取回灰度
            gray_3 = np.dstack([anomaly_np, anomaly_np, anomaly_np])
            color = random.choice(["150-255", "0-100"])
            anomaly_3, _ = add_stain(
                gray_3,
                size="0.1-4",
                color=color,
                irregularity=0.3,
                blur=0.01,
            )
            anomaly_np = cv2.cvtColor(anomaly_3, cv2.COLOR_BGR2GRAY)
        if not add_strip and not add_point:
            if random.random() < 0.5:
                anomaly_np = scar_creat(anomaly_np.copy())
            else:
                gray_3 = np.dstack([anomaly_np, anomaly_np, anomaly_np])
                color = random.choice(["150-255", "0-100"])
                anomaly_3, _ = add_stain(
                    gray_3, size="0.1-4", color=color, irregularity=0.3, blur=0.01
                )
                anomaly_np = cv2.cvtColor(anomaly_3, cv2.COLOR_BGR2GRAY)

        return (
            self.transform(Image.fromarray(normal_np)),
            self.transform(Image.fromarray(anomaly_np)),
            os.path.basename(self.images[idx]),
        )

    def __len__(self):
        return len(self.images)


def get_dataloader(
    data_dir: str, img_size: int = 256, batch_size: int = 8, preprocessed: bool = False
) -> DataLoader:
    ds = NormalDataset(data_dir, img_size, preprocessed=preprocessed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
