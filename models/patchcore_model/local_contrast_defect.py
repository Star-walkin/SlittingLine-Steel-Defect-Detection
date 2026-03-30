"""
局部背景相减法缺陷检测模块。

核心思想：PatchCore 热力图经上采样后，缺陷表现为“局部相对高度差极大”而非高梯度突变。
使用大核平滑得到局部背景 H_background，计算 H_diff = H - H_background，再对 H_diff 做阈值得到缺陷掩膜。
"""

import os
from typing import Tuple

import cv2
import numpy as np


def _hdiff_edge_blend(i: int, band: int, edge_alpha: float, profile: str = "ease_out_cubic") -> float:
    """
    将距外缘第 i 行/列（0..band-1）映射到权重 [edge_alpha, ~1]。
    ease_out_cubic：w = edge_alpha + (1-edge_alpha)*(1-(1-t)^3)，t=i/band，
    靠外缘变化快、靠内缘趋近 1 更缓，减轻与 Gaussian 背景相减后的边界伪峰。
    """
    if band <= 0:
        return 1.0
    t = float(i) / float(band)
    t = float(np.clip(t, 0.0, 1.0))
    if profile == "linear":
        ease = t
    else:
        ease = 1.0 - (1.0 - t) ** 3
    return float(edge_alpha + (1.0 - edge_alpha) * ease)


def detect_defects_by_local_contrast(
    heatmap: np.ndarray,
    valid_mask: np.ndarray,
    bg_ksize: int = 101,
    diff_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过计算局部相对高度差（局部对比度）来检测缺陷。

    参数:
        heatmap: PatchCore 输出的原始热力图 (2D float32 矩阵)
        valid_mask: 有效区域掩膜 (0 为边缘丢弃区，255 或 1 为有效区)
        bg_ksize: 估算背景的滤波核大小 (必须是奇数)。应大于带钢表面常见缺陷的像素尺寸。
        diff_threshold: 局部相对高度差的判定阈值。

    返回:
        defect_mask: 最终的二值化缺陷掩膜 (255 为缺陷，0 为正常, uint8)
        height_diff_map: 计算出的相对高度差矩阵 (float32，方便可视化和调试)
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.uint8)

    # 1) 大核平滑，提取局部背景地形 (H_background)
    if bg_ksize % 2 == 0:
        bg_ksize = bg_ksize + 1
    ksize = (bg_ksize, bg_ksize)
    H_background = cv2.GaussianBlur(heatmap, ksize, 0)

    # 2) 相对高度差: H_diff = heatmap - H_background
    H_diff = heatmap - H_background

    # 3) 消除负值
    H_diff = np.maximum(H_diff, 0.0)

    # 4) 边缘权重：压低边缘 H_diff，保留中心（ease_out cubic，避免线性 ramp 在过渡末端放大 diff 伪峰）
    Hm, Wm = H_diff.shape
    soft_border = min(20, Hm // 4, Wm // 4)
    edge_alpha = 0.4  # 边缘最低权重（0~1，可调）
    if soft_border > 0:
        wy = np.ones(Hm, dtype=np.float32)
        wx = np.ones(Wm, dtype=np.float32)
        for i in range(soft_border):
            w = _hdiff_edge_blend(i, soft_border, edge_alpha, "ease_out_cubic")
            wy[i] = min(wy[i], w)
            wy[-i - 1] = min(wy[-i - 1], w)
            wx[i] = min(wx[i], w)
            wx[-i - 1] = min(wx[-i - 1], w)
        weight2d = np.outer(wy, wx)
        H_diff = H_diff * weight2d

    # 5) 结合 diff_threshold 和 valid_mask 二值化
    valid_bool = (valid_mask > 0)
    defect_bool = (H_diff > diff_threshold) & valid_bool

    defect_mask = np.zeros(heatmap.shape, dtype=np.uint8)
    defect_mask[defect_bool] = 255

    height_diff_map = H_diff.astype(np.float32)
    return defect_mask, height_diff_map


def debug_local_contrast_visualization(
    heatmap: np.ndarray,
    valid_mask: np.ndarray,
    bg_ksize: int,
    diff_threshold: float,
    save_path: str = "debug_combined.jpg",
    original_img: np.ndarray | None = None,
    heatmap_raw: np.ndarray | None = None,
) -> None:
    """
    将局部背景相减法的关键中间结果做可视化，生成横向拼接图。

    - heatmap: 与业务一致的热力图（PatchCore 路径下为「正方形边缘抑制后」纵向拼接结果），
      用于 H_background / H_diff / 缺陷判定。
    - heatmap_raw: 可选；若提供，则在「原始热力图」与「背景图」之间多插一列「抑制后热力图」，
      便于对比推理原值与边缘抑制后的条带。

    无 heatmap_raw 时共 6 张：原图 | 热力图 | 背景 | Diff | Result | 原图+框
    有 heatmap_raw 时共 7 张：原图 | 热力原图 | 抑制后热力 | 背景 | Diff | Result | 原图+框
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.uint8)
    if heatmap_raw is not None:
        heatmap_raw = np.asarray(heatmap_raw, dtype=np.float32)
        if heatmap_raw.shape != heatmap.shape:
            raise ValueError(
                f"heatmap_raw shape {heatmap_raw.shape} 与 heatmap {heatmap.shape} 不一致"
            )

    # --- 核心计算，与 detect_defects_by_local_contrast 保持一致 ---
    if bg_ksize % 2 == 0:
        bg_ksize += 1
    ksize = (bg_ksize, bg_ksize)
    H_background = cv2.GaussianBlur(heatmap, ksize, 0)

    H_diff = np.maximum(heatmap - H_background, 0.0)

    # 与 detect_defects_by_local_contrast 中相同的边缘权重
    Hm, Wm = H_diff.shape
    soft_border = min(20, Hm // 4, Wm // 4)
    edge_alpha = 0.4
    if soft_border > 0:
        wy = np.ones(Hm, dtype=np.float32)
        wx = np.ones(Wm, dtype=np.float32)
        for i in range(soft_border):
            w = _hdiff_edge_blend(i, soft_border, edge_alpha, "ease_out_cubic")
            wy[i] = min(wy[i], w)
            wy[-i - 1] = min(wy[-i - 1], w)
            wx[i] = min(wx[i], w)
            wx[-i - 1] = min(wx[-i - 1], w)
        weight2d = np.outer(wy, wx)
        H_diff = H_diff * weight2d

    defect_bool = (H_diff > diff_threshold) & (valid_mask > 0)
    defect_mask = np.zeros_like(heatmap, dtype=np.uint8)
    defect_mask[defect_bool] = 255

    # 形态学闭运算 + 轮廓（用于确定裁剪边界中心）
    kernel = np.ones((3, 3), np.uint8)
    defect_closed = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(
        defect_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- 归一化辅助 ---
    def to_jet_colormap(data: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if vmin is None:
            vmin = float(data.min())
        if vmax is None:
            vmax = float(data.max())
        if vmax - vmin < 1e-8:
            norm = np.zeros_like(data, dtype=np.uint8)
        else:
            norm = (data - vmin) / (vmax - vmin)
            norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # 统一的 min/max：若有 raw+抑制 两路，则用联合范围便于对比
    hm_min = float(heatmap.min())
    hm_max = float(heatmap.max())
    if heatmap_raw is not None:
        hm_min = min(hm_min, float(heatmap_raw.min()))
        hm_max = max(hm_max, float(heatmap_raw.max()))

    # 原图（若提供则 resize 到与热力图相同大小）
    Hh, Wh = heatmap.shape
    if original_img is not None:
        orig = np.asarray(original_img)
        # 统一为 uint8，避免 float [0,1] 或异常类型在 imwrite/hstack 后整列发黑
        if orig.dtype in (np.float32, np.float64):
            omax = float(orig.max()) if orig.size else 0.0
            if omax <= 1.0 + 1e-5:
                orig = np.clip(orig * 255.0, 0, 255).astype(np.uint8)
            else:
                orig = np.clip(orig, 0, 255).astype(np.uint8)
        elif orig.dtype != np.uint8:
            orig = np.clip(orig.astype(np.float64), 0, 255).astype(np.uint8)
        if orig.ndim == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        orig_rs = cv2.resize(orig, (Wh, Hh), interpolation=cv2.INTER_LINEAR)
    else:
        # 若没有原图，则以全灰背景代替
        orig_rs = np.full((Hh, Wh, 3), 128, dtype=np.uint8)

    # 热力图列：可选「推理原图」与「边缘抑制后」两列
    if heatmap_raw is not None:
        img_heat_raw = to_jet_colormap(heatmap_raw, vmin=hm_min, vmax=hm_max)
        img_heat_sup = to_jet_colormap(heatmap, vmin=hm_min, vmax=hm_max)
        max_raw = float(heatmap_raw.max())
        max_sup = float(heatmap.max())
    else:
        img_heat_raw = None
        img_heat_sup = to_jet_colormap(heatmap, vmin=hm_min, vmax=hm_max)
        max_raw = float(heatmap.max())
        max_sup = max_raw

    # 背景：与业务热力图（抑制后）同尺度着色
    img_bg = to_jet_colormap(H_background, vmin=hm_min, vmax=hm_max)

    # 图4：相对高度差
    img_diff = to_jet_colormap(H_diff)

    # 计算与 obtain_anomaly_location 一致的正方形裁剪边界
    crop_rects: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) <= 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        H, W = Hh, Wh
        crop_size = 32
        left = max(cx - crop_size // 2, 0)
        right = min(cx + crop_size // 2, W)
        top = max(cy - crop_size // 2, 0)
        bottom = min(cy + crop_size // 2, H)
        if right - left < crop_size:
            if left == 0:
                right = min(crop_size, W)
            elif right == W:
                left = max(W - crop_size, 0)
        if bottom - top < crop_size:
            if top == 0:
                bottom = min(crop_size, H)
            elif bottom == H:
                top = max(H - crop_size, 0)
        crop_rects.append((left, top, right, bottom))

    # 图5：Result，在热力图上按裁剪边界画正方形框（与业务一致：抑制后热力图）
    img_result = img_heat_sup.copy()
    for left, top, right, bottom in crop_rects:
        cv2.rectangle(img_result, (left, top), (right, bottom), (0, 255, 0), 2)

    # 图6：在原图上画同样裁剪边界
    orig_anno = orig_rs.copy()
    for left, top, right, bottom in crop_rects:
        cv2.rectangle(orig_anno, (left, top), (right, bottom), (0, 255, 0), 2)

    # 文本标注（白字+黑描边）
    def put_label(img: np.ndarray, text: str, y_offset: int = 28) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (5, y_offset)
        cv2.putText(img, text, org, font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    put_label(orig_rs, "0. Raw Image")
    if heatmap_raw is not None:
        put_label(img_heat_raw, f"1. Heatmap raw Max={max_raw:.4f}")
        put_label(img_heat_sup, f"2. Edge-suppressed Max={max_sup:.4f}")
        put_label(img_bg, f"3. Background (k={bg_ksize})")
        put_label(img_diff, "4. Diff Map")
        put_label(img_result, f"5. Result (Thresh={diff_threshold:.3f})")
        put_label(orig_anno, "6. Raw + Crops")
    else:
        put_label(img_heat_sup, f"1. Heatmap  Max={max_sup:.4f}")
        put_label(img_bg, f"2. Background (k={bg_ksize})")
        put_label(img_diff, "3. Diff Map")
        put_label(img_result, f"4. Result (Thresh={diff_threshold:.3f})")
        put_label(orig_anno, "5. Raw + Crops")

    # 尺寸对齐并横向拼接
    H, W = img_heat_sup.shape[:2]
    imgs = []
    if heatmap_raw is not None:
        seq = (orig_rs, img_heat_raw, img_heat_sup, img_bg, img_diff, img_result, orig_anno)
    else:
        seq = (orig_rs, img_heat_sup, img_bg, img_diff, img_result, orig_anno)
    for im in seq:
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        imgs.append(im)

    combined = np.hstack(imgs)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, combined)


# ================= 调试辅助 =================
def test_local_contrast(
    heatmap_path: str | None = None,
    save_dir: str | None = None,
    bg_ksize: int = 101,
    diff_threshold: float = 1.0,
    edge_crop: int = 20,
) -> None:
    """
    读取一个 .npy 热力图（或使用随机热力图），调用 detect_defects_by_local_contrast，
    并使用 matplotlib 可视化：原始热力图、局部背景图、相对高度差图、最终缺陷掩膜。

    参数:
        heatmap_path: .npy 热力图文件路径；若为 None 则使用随机生成的示例热力图
        save_dir: 可视化结果保存目录；若为 None 则仅显示不保存
        bg_ksize: 背景平滑核大小
        diff_threshold: 相对高度差阈值
        edge_crop: 有效区域边缘裁切像素（用于构建 valid_mask）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib，请安装: pip install matplotlib")
        return

    if heatmap_path and os.path.isfile(heatmap_path):
        heatmap = np.load(heatmap_path).astype(np.float32)
        if heatmap.ndim > 2:
            heatmap = heatmap.squeeze()
    else:
        # 示例：随机生成带局部凸起的 256x256 热力图
        np.random.seed(42)
        H, W = 256, 256
        heatmap = np.random.rand(H, W).astype(np.float32) * 0.5
        heatmap[80:120, 80:120] += 2.0
        heatmap[150:190, 150:190] += 1.5

    Hm, Wm = heatmap.shape
    ec = max(0, min(edge_crop, Hm // 2 - 1, Wm // 2 - 1))
    valid_mask = np.zeros((Hm, Wm), dtype=np.uint8)
    valid_mask[ec : Hm - ec, ec : Wm - ec] = 255

    defect_mask, height_diff_map = detect_defects_by_local_contrast(
        heatmap, valid_mask, bg_ksize=bg_ksize, diff_threshold=diff_threshold
    )

    # 伪彩色：热力图、背景、高度差、缺陷掩膜
    def to_jet_uint8(arr: np.ndarray) -> np.ndarray:
        a_min, a_max = float(arr.min()), float(arr.max())
        denom = a_max - a_min if (a_max - a_min) > 1e-8 else 1.0
        norm = np.uint8(np.clip((arr - a_min) / denom * 255, 0, 255))
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    if bg_ksize % 2 == 0:
        _k = bg_ksize + 1
    else:
        _k = bg_ksize
    H_background = cv2.GaussianBlur(heatmap.astype(np.float32), (_k, _k), 0)

    img_heat = to_jet_uint8(heatmap)
    img_bg = to_jet_uint8(H_background)
    img_diff = to_jet_uint8(height_diff_map)
    img_def = cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2BGR)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(cv2.cvtColor(img_heat, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. 原始热力图 H")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. 局部背景 H_background")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(img_diff, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("3. 相对高度差 H_diff")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("4. 缺陷掩膜 defect_mask")
    axes[1, 1].axis("off")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "local_contrast_debug.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"已保存: {out_path}")
    plt.show()


if __name__ == "__main__":
    test_local_contrast(
        heatmap_path=None,
        save_dir=os.path.join(os.path.dirname(__file__), "debug_local_contrast"),
        bg_ksize=51,
        diff_threshold=1.0,
        edge_crop=20,
    )
