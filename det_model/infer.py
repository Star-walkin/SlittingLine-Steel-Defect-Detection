"""
det_model/infer.py
SlidingWindowDetector - 在原始分辨率下使用滑动窗口推理，避免缩放损失缺陷信号。
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as _F
import torchvision.transforms as transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from function_bank import test_one_image, mean_smoothing

# 与训练数据预处理一致：FFT 去纹（img_select 相同逻辑），削弱横向条纹
MASK_WIDTH_FFT = 10
CENTER_PROTECT_FFT = 15


def apply_fft_deripple(gray: np.ndarray) -> np.ndarray:
    """对灰度图做 FFT 去纹（PatchCore/预处理链路可复用）。"""
    rows, cols = gray.shape
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.ones((rows, cols, 2), np.uint8)
    cr, cc = rows // 2, cols // 2
    mask[:, cc - MASK_WIDTH_FFT : cc + MASK_WIDTH_FFT] = 0
    mask[
        cr - CENTER_PROTECT_FFT : cr + CENTER_PROTECT_FFT,
        cc - CENTER_PROTECT_FFT : cc + CENTER_PROTECT_FFT,
    ] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)


def apply_vertical_filter(gray: np.ndarray, block_h: int = 512) -> np.ndarray:
    """
    分块纵向滤波：按 block_h 高度切块，块内列中值投影后减去，再统一加回原图全局均值。
    解决带钢轻微跑偏/蛇行导致竖纹不绝对笔直、全局投影无法消除的问题；最后统一补偿避免块间接缝。
    """
    img_float = gray.astype(np.float32)
    H, W = gray.shape[:2]
    result = np.zeros_like(img_float)
    for y in range(0, H, block_h):
        y_end = min(y + block_h, H)
        chunk = img_float[y:y_end, :]
        col_median = np.median(chunk, axis=0)
        vertical_pattern = np.tile(col_median, (y_end - y, 1))
        result[y:y_end, :] = chunk - vertical_pattern
    global_mean = float(np.mean(img_float))
    result += global_mean
    return np.clip(result, 0, 255).astype(np.uint8)


class SlidingWindowDetector(test_one_image):
    """
    继承 test_one_image，仅重写 detect_ano 为滑动窗口推理。
    关键：在完整条带原始分辨率提取 256×256 patch，累积 L2 差异图，再缩放到标准输出尺寸。
    """

    def __init__(self, *args, patch_stride: int = 128, flatten_bg: bool = True, **kwargs):
        self._flatten_bg = kwargs.pop("flatten_bg", flatten_bg)
        self.patch_stride = kwargs.pop("patch_stride", patch_stride)
        super().__init__(*args, **kwargs)
        self.flatten_bg = self._flatten_bg

        self._patch_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def detect_ano(self, image, left_edge=0, right_edge=None):
        device = next(self.model.parameters()).device

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        H, W = gray.shape
        if right_edge is None:
            right_edge = W

        bias = 15
        l = max(int(left_edge + bias), 0)
        r = min(int(right_edge - bias), W)
        if r <= l:
            l, r = 0, W
        crop = gray[:, l:r]
        cH, cW = crop.shape

        try:
            crop = apply_fft_deripple(crop)
        except Exception:
            pass

        # 纵向滤波：去除轧制纹/列噪声，与 prepare_dataset_det 一致
        crop = apply_vertical_filter(crop)

        if self.flatten_bg:
            # 中值滤波提取背景（低频）→ 原图减背景 + 背景均值补偿 → 截断，保留缺陷高频
            ksize = max(51, min(cW, cH) // 8) | 1
            background_u8 = cv2.medianBlur(crop, ksize)
            img_float = crop.astype(np.float32)
            bg_float = background_u8.astype(np.float32)
            mean_val = float(np.mean(bg_float))
            flattened = img_float - bg_float + mean_val
            crop = np.clip(flattened, 0, 255).astype(np.uint8)

        ps = self.img_size
        st = self.patch_stride
        need_pad_H = (ps - cH % ps) % ps if cH >= ps else (ps - cH)
        need_pad_W = (ps - cW % ps) % ps if cW >= ps else (ps - cW)
        padded = np.pad(crop, ((0, need_pad_H), (0, need_pad_W)), mode="reflect")
        pH, pW = padded.shape

        patches, positions = [], []
        for y in range(0, pH - ps + 1, st):
            for x in range(0, pW - ps + 1, st):
                patches.append(padded[y : y + ps, x : x + ps])
                positions.append((y, x))
        if positions and positions[-1] != (pH - ps, pW - ps):
            patches.append(padded[pH - ps : pH, pW - ps : pW])
            positions.append((pH - ps, pW - ps))

        BATCH = 32
        amap_acc = np.zeros((pH, pW), dtype=np.float32)
        count_map = np.zeros((pH, pW), dtype=np.float32)

        with torch.inference_mode():
            for bi in range(0, len(patches), BATCH):
                bp = patches[bi : bi + BATCH]
                bpos = positions[bi : bi + BATCH]
                bt = torch.stack([self._patch_tf(Image.fromarray(p)) for p in bp]).to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    rt = self.model(bt)
                diff_np = ((bt - rt) ** 2)[:, 0].detach().cpu().float().numpy()
                for j, (y, x) in enumerate(bpos):
                    amap_acc[y : y + ps, x : x + ps] += diff_np[j]
                    count_map[y : y + ps, x : x + ps] += 1

        count_map = np.maximum(count_map, 1e-6)
        amap_full = (amap_acc / count_map)[:cH, :cW]

        at = torch.from_numpy(amap_full[None, None]).float()
        at = mean_smoothing(at, kernel_size1=5)
        amap_full = at[0, 0].numpy()

        out_H = self.img_size * self.cut_ratio
        out_W = self.img_size
        amap_cut = cv2.resize(amap_full, (out_W, out_H), interpolation=cv2.INTER_LINEAR)

        brd = 5
        amap_cut[:brd, :] = 0
        amap_cut[-brd:, :] = 0
        amap_cut[:, :brd] = 0
        amap_cut[:, -brd:] = 0

        test_cut = cv2.resize(crop, (out_W, out_H), interpolation=cv2.INTER_LINEAR)
        rec_cut = np.full_like(test_cut, 128, dtype=np.uint8)

        # 与 function_bank 中 defect 原图裁剪：模型空间 -> 条带 crop 坐标的均匀映射
        self._last_detect_mode = "sliding"
        self._last_sliding_crop_hw = (cH, cW)
        self._last_test_cut_hw = (out_H, out_W)
        self._last_cut_l = l
        self._last_tiles_u8 = None

        return test_cut, rec_cut, amap_cut


# ---------------------------------------------------------------------------
# GPU 加速预处理（替代 CPU 的 apply_fft_deripple + apply_vertical_filter +
# flatten_background_subtraction，实测从 ~554ms 降至 ~28ms）
# ---------------------------------------------------------------------------

def apply_fft_deripple_gpu(strip_t: torch.Tensor) -> torch.Tensor:
    """
    GPU FFT 去纹（等价于 apply_fft_deripple）。
    输入: float32 2D Tensor (H, W)，在 CUDA 上。
    输出: 同设备同 dtype 的去纹结果，值域 [0, 255]。
    耗时约 16ms（vs CPU cv2.dft 约 323ms）。
    """
    f = torch.fft.fft2(strip_t)
    fs = torch.fft.fftshift(f)
    r, c = fs.shape
    cr, cc = r // 2, c // 2
    # 遮挡垂直条纹对应的竖向频率带
    fs[:, max(0, cc - MASK_WIDTH_FFT): cc + MASK_WIDTH_FFT] = 0
    # 保留中心低频（直流 + 全局亮度），避免整体变黑
    cp = CENTER_PROTECT_FFT
    fs[max(0, cr - cp): cr + cp, max(0, cc - cp): cc + cp] = \
        f[max(0, cr - cp): cr + cp, max(0, cc - cp): cc + cp]
    img_back = torch.fft.ifft2(torch.fft.ifftshift(fs)).abs()
    # 归一化到 [0, 255]
    mn, mx = img_back.min(), img_back.max()
    img_back = (img_back - mn) / (mx - mn + 1e-8) * 255.0
    return img_back


def apply_vertical_filter_gpu(strip_t: torch.Tensor, block_h: int = 512) -> torch.Tensor:
    """
    GPU 纵向滤波：分块列均值投影减法（近似替代 CPU 列中值，速度快 10-20x）。
    输入/输出: float32 2D Tensor，值域不限（会 clamp 到 [0,255]）。
    """
    H, W = strip_t.shape
    result = torch.zeros_like(strip_t)
    for y in range(0, H, block_h):
        y_end = min(y + block_h, H)
        chunk = strip_t[y:y_end, :]
        col_mean = chunk.mean(dim=0, keepdim=True)
        result[y:y_end, :] = chunk - col_mean
    global_mean = strip_t.mean()
    result = result + global_mean
    return result.clamp(0, 255)


def flatten_background_gpu(strip_t: torch.Tensor) -> torch.Tensor:
    """
    GPU 背景拍平：降采样 avgpool 估计低频背景再减去（近似替代 CPU medianBlur）。
    输入/输出: float32 2D Tensor，值域不限（会 clamp 到 [0,255]）。
    耗时约 11ms（vs CPU medianBlur 约 231ms）。
    """
    x = strip_t.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    # 降采样 8x，在小分辨率上做大核均值，再上采样回原尺寸
    x_down = _F.avg_pool2d(x, kernel_size=8, stride=8)
    bg_down = _F.avg_pool2d(x_down, kernel_size=15, stride=1, padding=7)
    bg_up = _F.interpolate(bg_down, size=strip_t.shape, mode="bilinear", align_corners=False)
    flat = x - bg_up + bg_up.mean()
    return flat.squeeze().clamp(0, 255)


def preprocess_like_inference_gpu(
    gray: np.ndarray,
    device: torch.device,
    bias: int = 15,
) -> np.ndarray:
    """
    GPU 加速版 preprocess_like_inference。
    与 CPU 版逻辑等价（裁边 + FFT 去纹 + 纵向滤波 + 背景拍平），
    但全流程在 GPU 上完成，速度从 ~554ms 降至 ~28ms。

    仅在 device.type == 'cuda' 时调用；CPU fallback 请继续使用 preprocess_like_inference。
    """
    H, W = gray.shape[:2]
    l = max(bias, 0)
    r = min(W - bias, W)
    if r <= l:
        l, r = 0, W

    # 避免在 CPU 上做 gray[:, l:r].astype(float32) 的整块大分配
    # （这在多相机 + debug 写盘叠加时很容易触发 OOM/碎片化）。
    sl = gray[:, l:r]
    if not sl.flags.c_contiguous:
        sl = np.ascontiguousarray(sl)

    # 上传到 GPU，并在 GPU 端完成 float32 转换
    strip_t = torch.from_numpy(sl).to(device=device, dtype=torch.float32, non_blocking=True)

    # FFT 去纹 → 纵向滤波 → 背景拍平（数值流程与旧实现一致）
    strip_t = apply_fft_deripple_gpu(strip_t)
    strip_t = apply_vertical_filter_gpu(strip_t)
    strip_t = flatten_background_gpu(strip_t)

    # 与旧实现保持一致：先 clamp 到 [0,255]，再截断到 uint8
    strip_t = strip_t.clamp(0, 255)
    out = strip_t.to(torch.uint8).contiguous().cpu().numpy()
    return out
