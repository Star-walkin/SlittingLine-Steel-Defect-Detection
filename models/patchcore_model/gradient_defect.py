"""
热力图梯度法缺陷检测模块。

核心思想：真实缺陷在热力图上产生剧烈数值突变，通过 Sobel 梯度幅值检测突变区域，
而非依赖绝对分数阈值。适用于 PatchCore 等异常分数热力图。
"""

import os
from typing import Tuple

import cv2
import numpy as np


def detect_defects_by_gradient(
    heatmap: np.ndarray,
    valid_mask: np.ndarray,
    blur_ksize: int = 5,
    grad_threshold: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过分析热力图的梯度幅值来检测突变型缺陷。

    参数:
        heatmap: PatchCore 输出的原始热力图 (2D 浮点数矩阵)
        valid_mask: 有效区域掩膜 (0 为边缘丢弃区，>0 为有效区)
        blur_ksize: 高斯模糊的核大小 (必须是奇数)，用于平滑热力图
        grad_threshold: 梯度幅值的判定阈值 (需根据实际数据微调)

    返回:
        defect_mask: 最终的二值化缺陷掩膜 (255 为缺陷，0 为正常)
        gradient_magnitude: 计算出的梯度幅值矩阵 (方便可视化和调试)
    """
    # 1) 对 heatmap 应用高斯模糊以消除毛刺噪声
    if blur_ksize % 2 == 0:
        blur_ksize = blur_ksize + 1
    heatmap_smooth = cv2.GaussianBlur(heatmap.astype(np.float32), (blur_ksize, blur_ksize), 0)

    # 2) 使用 cv2.Sobel 分别计算 X 方向和 Y 方向的梯度
    gx = cv2.Sobel(heatmap_smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(heatmap_smooth, cv2.CV_64F, 0, 1, ksize=3)

    # 3) 计算梯度幅值
    gradient_magnitude = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)

    # 4) 根据 grad_threshold 判定，并与 valid_mask 与运算，剔除边缘伪影
    valid_uint = (valid_mask > 0).astype(np.uint8)
    defect_bool = (gradient_magnitude > grad_threshold) & (valid_uint > 0)

    defect_mask = np.zeros_like(heatmap, dtype=np.uint8)
    defect_mask[defect_bool] = 255

    return defect_mask, gradient_magnitude.astype(np.float32)


def visualize_gradient_results(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    defect_mask: np.ndarray,
    gradient_magnitude: np.ndarray,
    save_path: str | None = None,
) -> np.ndarray:
    """
    将原图、原始热力图(伪彩色)、梯度幅值图(伪彩色) 和 最终判定掩膜 拼接在一起，
    用于调试与可视化。

    参数:
        original_img: 2D 灰度图或 3 通道 BGR 图
        heatmap: 2D 浮点热力图
        defect_mask: 0/255 二值缺陷掩膜
        gradient_magnitude: 2D 浮点梯度幅值矩阵
        save_path: 若提供，则保存拼接结果到该路径

    返回:
        vis: 拼接后的 BGR 图像 (左→右: 原图 | 热力图 | 梯度幅值 | 缺陷掩膜)
    """
    h, w = heatmap.shape

    # 统一 resize 到相同尺寸（若原图尺寸不一致）
    orig_rs = cv2.resize(original_img, (w, h), interpolation=cv2.INTER_LINEAR)
    if orig_rs.ndim == 2:
        orig_rs = cv2.cvtColor(orig_rs, cv2.COLOR_GRAY2BGR)

    # 热力图 min-max 归一化到 0-255 再伪彩色
    hm_min = float(np.min(heatmap))
    hm_max = float(np.max(heatmap))
    hm_denom = hm_max - hm_min if (hm_max - hm_min) > 1e-8 else 1.0
    hm_norm = np.uint8(np.clip((heatmap - hm_min) / hm_denom * 255, 0, 255))
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

    # 梯度幅值 min-max 归一化
    gm_min = float(np.min(gradient_magnitude))
    gm_max = float(np.max(gradient_magnitude))
    gm_denom = gm_max - gm_min if (gm_max - gm_min) > 1e-8 else 1.0
    gm_norm = np.uint8(np.clip((gradient_magnitude - gm_min) / gm_denom * 255, 0, 255))
    gm_color = cv2.applyColorMap(gm_norm, cv2.COLORMAP_JET)

    # 缺陷掩膜 (单通道转 BGR)
    def_vis = cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2BGR)

    vis = np.hstack([orig_rs, hm_color, gm_color, def_vis])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, vis)

    return vis
