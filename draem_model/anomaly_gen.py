"""
draem_model/anomaly_gen.py
伪缺陷生成器，同时返回缺陷图像与对应的二值 mask（0/255）。
供 DRAEM dataset 使用。
"""

import os
import sys
import random
import numpy as np
import cv2

_SEG_TRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seg_model_train"
)
if _SEG_TRAIN_DIR not in sys.path:
    sys.path.insert(0, _SEG_TRAIN_DIR)

from stain_anomaly import add_stain


# ─────────────────────────── 条状缺陷（带 mask） ─────────────────────────

def gen_scar_with_mask(gray: np.ndarray,
                       length_range=(1, 20),
                       width_range=(600, 1024),
                       brightness_change_range=(5, 60),
                       small_length_threshold=5):
    """
    生成条状缺陷（细长划痕），同时返回 mask。
    gray: 2D uint8 灰度图，大小通常为 256x256
    返回: (augmented_gray, binary_mask_uint8)

    注意：
      - 原始实现的 width_range=(600,1024) 在 256x256 上会直接导致 randint 抛错，
        这里改为基于图像尺寸自适应地生成“长而窄”的划痕，避免异常并更贴近实际轻微划痕。
    """
    H, W = gray.shape[:2]

    # ---- 自适应确定划痕尺寸（长而窄） ----
    # 划痕沿水平方向延伸：长度占宽度的 40%~90%
    len_min = max(int(W * 0.4), 8)
    len_max = max(len_min + 1, int(W * 0.9))
    len_max = min(len_max, W)
    if len_max <= len_min:
        len_min = max(2, W // 3)
        len_max = max(len_min + 1, W)
    strip_length = np.random.randint(len_min, len_max)

    # 划痕沿垂直方向的“宽度”很窄：1~4 像素左右
    wid_min = max(1, int(H * 0.004))   # ~1px
    wid_max = max(wid_min + 1, int(H * 0.02))  # ~3-5px
    wid_max = min(wid_max, H)
    if wid_max <= wid_min:
        wid_min = 1
        wid_max = max(2, min(4, H))
    strip_width = np.random.randint(wid_min, wid_max)

    # 起始位置，保证完全落在图内
    start_x = np.random.randint(0, max(1, W - strip_length))
    start_y = np.random.randint(0, max(1, H - strip_width))

    region = gray[start_y:start_y + strip_width, start_x:start_x + strip_length]
    mean_brightness = float(np.mean(region))

    # 划痕亮度变化幅度适中，避免过亮/过暗
    eff_small_th = max(small_length_threshold, int(W * 0.1))
    if strip_length < eff_small_th:
        scar_val = 255.0
    else:
        # 相对温和的亮度变化，更贴近轻微划痕
        delta_low, delta_high = brightness_change_range
        delta_high = max(delta_low + 1, min(delta_high, 40))
        delta = float(np.random.randint(delta_low, delta_high))
        if random.random() < 0.5:
            delta = -delta
        scar_val = np.clip(mean_brightness + delta, 0, 255)

    strip = np.full_like(region, scar_val, dtype=np.float32)
    strip = cv2.GaussianBlur(strip, (9, 9), 0)

    augmented = gray.copy().astype(np.float32)
    aug_region = augmented[start_y:start_y + strip_width, start_x:start_x + strip_length]
    blended = cv2.addWeighted(aug_region.astype(np.uint8), 0.7,
                               strip.astype(np.uint8), 0.3, 0)
    augmented[start_y:start_y + strip_width, start_x:start_x + strip_length] = blended

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[start_y:start_y + strip_width, start_x:start_x + strip_length] = 255

    return augmented.astype(np.uint8), mask


# ─────────────────────────── 点状缺陷（带 mask） ─────────────────────────

def gen_stain_with_mask(gray: np.ndarray, size='0.2-4', color='150-255'):
    """
    生成点状/块状污点缺陷，同时返回 mask。
    gray: 2D uint8 灰度图
    返回: (augmented_gray, binary_mask_uint8) — mask 已二值化（>127 → 255）
    """
    augmented, mask_float = add_stain(
        gray,
        size=size,
        color=color,
        irregularity=0.3,
        blur=0.01,
    )
    mask_bin = np.where(mask_float > 64, 255, 0).astype(np.uint8)
    return augmented, mask_bin


# ─────────────────────────── 组合生成接口 ────────────────────────────────

def gen_anomaly_with_mask(gray: np.ndarray, p_scar: float = 0.5, p_stain: float = 0.5):
    """
    随机叠加条状 / 点状缺陷，至少生成一种。
    返回: (anomalous_gray, combined_mask_uint8)
    """
    H, W = gray.shape[:2]
    aug = gray.copy()
    combined_mask = np.zeros((H, W), dtype=np.uint8)

    add_scar = random.random() < p_scar
    add_stain_flag = random.random() < p_stain
    if not add_scar and not add_stain_flag:
        if random.random() < 0.5:
            add_scar = True
        else:
            add_stain_flag = True

    if add_scar:
        try:
            aug, m = gen_scar_with_mask(aug)
            combined_mask = np.maximum(combined_mask, m)
        except Exception:
            pass

    if add_stain_flag:
        try:
            color = random.choice(["150-230", "20-100"])
            aug, m = gen_stain_with_mask(aug, size='0.3-5', color=color)
            combined_mask = np.maximum(combined_mask, m)
        except Exception:
            pass

    return aug, combined_mask
