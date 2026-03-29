"""
eff_ad/infer.py
推理封装：与 detectoutline02 接口一致，detect_ano(image) -> (test_cut, rec_cut, amap_cut)。
支持整条带切 tile 后逐块推理再拼成 amap。
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from function_bank import test_one_image, mean_smoothing
from eff_ad.model import StudentTeacher


class _DummyModel(torch.nn.Module):
    """占位，满足 test_one_image 的 model 参数。"""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

# FFT 去纹（与训练/推理一致，可选）
MASK_W, CENTER_P = 10, 15


def apply_fft_deripple(gray: np.ndarray) -> np.ndarray:
    rows, cols = gray.shape
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.ones((rows, cols, 2), np.uint8)
    cr, cc = rows // 2, cols // 2
    mask[:, cc - MASK_W : cc + MASK_W] = 0
    mask[cr - CENTER_P : cr + CENTER_P, cc - CENTER_P : cc + CENTER_P] = 1
    f_ishift = np.fft.ifftshift(dft_shift * mask)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)


class EffADDetector(test_one_image):
    """
    继承 test_one_image，仅重写 detect_ano：用 Student-Teacher 生成 amap。
    输入整幅条带图，切 tile 256x256 后逐块推理，再拼成与 cut_ratio 一致的输出尺寸。
    """

    def __init__(self, model=None, *args, flatten_bg: bool = True, use_fft: bool = True, **kwargs):
        self._flatten_bg = kwargs.pop("flatten_bg", flatten_bg)
        self._use_fft = kwargs.pop("use_fft", use_fft)
        if model is None:
            model = _DummyModel()
        super().__init__(model, *args, **kwargs)

    def load_eff_ad(self, ckpt_path: str, device=None):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.eff_ad = StudentTeacher(feat_channels=128)
        self.eff_ad.teacher.load_state_dict(ckpt["teacher"])
        self.eff_ad.student.load_state_dict(ckpt["student"])
        if device is None:
            device = next(self.model.parameters()).device
        self.eff_ad = self.eff_ad.to(device)
        self.eff_ad.eval()
        self._eff_device = device
        return self

    def detect_ano(self, image, left_edge=0, right_edge=None):
        if not hasattr(self, "eff_ad"):
            raise RuntimeError("Call load_eff_ad(ckpt_path) before detect_ano.")
        device = self._eff_device

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

        if self._use_fft:
            try:
                crop = apply_fft_deripple(crop)
            except Exception:
                pass

        if self._flatten_bg:
            ksize = max(51, cW // 10) | 1
            bg = cv2.GaussianBlur(crop.astype(np.float32), (ksize, ksize), 0)
            crop = np.clip(crop.astype(np.float32) - bg + 128.0, 0, 255).astype(np.uint8)

        ps = self.img_size
        need_pad_H = (ps - cH % ps) % ps if cH >= ps else (ps - cH)
        need_pad_W = (ps - cW % ps) % ps if cW >= ps else (ps - cW)
        padded = np.pad(crop, ((0, need_pad_H), (0, need_pad_W)), mode="reflect")
        pH, pW = padded.shape

        amap_acc = np.zeros((pH, pW), dtype=np.float32)
        count_map = np.zeros((pH, pW), dtype=np.float32)

        with torch.inference_mode():
            for y in range(0, pH - ps + 1, ps):
                for x in range(0, pW - ps + 1, ps):
                    patch = padded[y : y + ps, x : x + ps]
                    t = torch.from_numpy(patch).float().div(255.0).sub(0.5).div(0.5)
                    t = t.unsqueeze(0).unsqueeze(0).to(device)

                    # Student-Teacher 的 forward_anomap 通常输出下采样特征图
                    # 例如 16x16，这里统一插值回 256x256 再进行拼接
                    amap = self.eff_ad.forward_anomap(t)
                    if isinstance(amap, (list, tuple)):
                        amap = amap[0]
                    amap = amap[0, 0].detach().cpu().float().numpy()
                    if amap.shape != (ps, ps):
                        amap = cv2.resize(amap, (ps, ps), interpolation=cv2.INTER_LINEAR)

                    amap_acc[y : y + ps, x : x + ps] += amap
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

        return test_cut, rec_cut, amap_cut
