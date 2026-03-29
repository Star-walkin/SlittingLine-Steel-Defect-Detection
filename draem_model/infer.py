"""
draem_model/infer.py
DRAEMDetector — 滑动窗口推理，输出与 test_one_image 接口兼容的 (test_cut, rec_cut, amap_cut)。
预处理流程（与训练完全一致）：FFT 去纹 → 背景拍平 → 归一化送入模型
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from draem_model.dataset import flatten_background

# ──────────────────────── FFT 去纹（与训练集 cam*_filter 一致）────────────

MASK_WIDTH_FFT = 10
CENTER_PROTECT_FFT = 15


def apply_fft_deripple(gray: np.ndarray) -> np.ndarray:
    rows, cols = gray.shape
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.ones((rows, cols, 2), np.uint8)
    cr, cc = rows // 2, cols // 2
    mask[:, cc - MASK_WIDTH_FFT: cc + MASK_WIDTH_FFT] = 0
    mask[cr - CENTER_PROTECT_FFT: cr + CENTER_PROTECT_FFT,
         cc - CENTER_PROTECT_FFT: cc + CENTER_PROTECT_FFT] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)


# ────────────────────── 形态学辅助 + 位置计算 ────────────────────────────

def _morphological_closing(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# ─────────────────────────── DRAEM 推理器 ────────────────────────────────

class DRAEMDetector:
    """
    与 test_one_image / SlidingWindowDetector 接口兼容的独立推理器。
    方法：
      detect_ano(image) → (test_cut, rec_cut, amap_cut)
      obtain_anomaly_location(amap, test_img, save_path, image_id, fukuan) → (state, centers, areas)
    属性：seg_anomaly_thres
    """

    def __init__(self, rec_net, disc_net,
                 conduct_id, fukuan0, cut_ratio, img_size,
                 seg_anomaly_thres, standard_ratio_x, standard_ratio_y, steel_real_y0,
                 patch_stride: int = 128, flatten_bg: bool = True, use_fft: bool = False):
        self.rec_net = rec_net
        self.disc_net = disc_net
        self.conduct_id = conduct_id
        self.fukuan0 = fukuan0
        self.cut_ratio = cut_ratio
        self.img_size = img_size
        self.seg_anomaly_thres = seg_anomaly_thres
        self.standard_ratio_x = standard_ratio_x
        self.standard_ratio_y = standard_ratio_y
        self.steel_real_y0 = steel_real_y0
        self.patch_stride = patch_stride
        self.flatten_bg = flatten_bg
        self.use_fft = use_fft

        self._tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def detect_ano(self, image, left_edge=0, right_edge=None):
        device = next(self.rec_net.parameters()).device

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

        # FFT 去纹（训练阶段未使用 FFT，这里默认关闭；如需一致性可在实例化时显式开启）
        if self.use_fft:
            try:
                crop = apply_fft_deripple(crop)
            except Exception:
                pass

        # 背景拍平
        if self.flatten_bg:
            crop = flatten_background(crop)

        # Reflect padding，保证完整覆盖
        ps = self.img_size
        st = self.patch_stride
        need_pad_H = (ps - cH % ps) % ps if cH >= ps else (ps - cH)
        need_pad_W = (ps - cW % ps) % ps if cW >= ps else (ps - cW)
        padded = np.pad(crop, ((0, need_pad_H), (0, need_pad_W)), mode="reflect")
        pH, pW = padded.shape

        # 收集 patches
        patches, positions = [], []
        for y in range(0, pH - ps + 1, st):
            for x in range(0, pW - ps + 1, st):
                patches.append(padded[y: y + ps, x: x + ps])
                positions.append((y, x))
        if positions and positions[-1] != (pH - ps, pW - ps):
            patches.append(padded[pH - ps: pH, pW - ps: pW])
            positions.append((pH - ps, pW - ps))

        # 批量推理
        BATCH = 32
        amap_acc = np.zeros((pH, pW), dtype=np.float32)
        count_map = np.zeros((pH, pW), dtype=np.float32)
        rec_acc = np.zeros((pH, pW), dtype=np.float32)

        with torch.inference_mode():
            for bi in range(0, len(patches), BATCH):
                bp = patches[bi: bi + BATCH]
                bpos = positions[bi: bi + BATCH]
                bt = torch.stack([self._tf(Image.fromarray(p)) for p in bp]).to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    rec = self.rec_net(bt)
                    disc_in = torch.cat([bt, rec], dim=1)
                    score = self.disc_net(disc_in)   # [B,1,256,256] Sigmoid

                # 防止权重异常导致 rec/score 中出现 NaN/Inf，影响后续 numpy 计算
                rec = torch.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)
                score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

                score_np = score[:, 0].detach().cpu().float().numpy()
                rec_np = (rec[:, 0].detach().cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()

                for j, (y, x) in enumerate(bpos):
                    amap_acc[y: y + ps, x: x + ps] += score_np[j]
                    rec_acc[y: y + ps, x: x + ps] += rec_np[j]
                    count_map[y: y + ps, x: x + ps] += 1

        count_map = np.maximum(count_map, 1e-6)
        amap_full = (amap_acc / count_map)[:cH, :cW]
        rec_full = (rec_acc / count_map)[:cH, :cW]

        # 再次清理可能残留的 NaN/Inf，避免后续 resize / astype 报 warning
        amap_full = np.nan_to_num(amap_full, nan=0.0, posinf=1.0, neginf=0.0)
        rec_full = np.nan_to_num(rec_full, nan=0.0, posinf=1.0, neginf=0.0)

        # 轻微高斯平滑
        amap_full = cv2.GaussianBlur(amap_full, (5, 5), 0)

        # 缩放到标准输出尺寸
        out_H = self.img_size * self.cut_ratio
        out_W = self.img_size
        amap_cut = cv2.resize(amap_full, (out_W, out_H), interpolation=cv2.INTER_LINEAR)
        rec_cut = (cv2.resize(rec_full, (out_W, out_H), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
        test_cut = cv2.resize(crop, (out_W, out_H), interpolation=cv2.INTER_LINEAR)

        # 边缘遮罩
        brd = 5
        amap_cut[:brd, :] = 0
        amap_cut[-brd:, :] = 0
        amap_cut[:, :brd] = 0
        amap_cut[:, -brd:] = 0

        return test_cut, rec_cut, amap_cut

    def obtain_anomaly_location(self, amap, test_img, anomaly_save_path, image_id, fukuan=None):
        """与 test_one_image.obtain_anomaly_location 完全兼容的位置计算。"""
        from PIL import Image as PILImage

        center_coords, area_list = [], []

        # DiscNet 输出是 [0,1]，判断严重异常
        if amap.mean() > self.seg_anomaly_thres + 0.05:
            cx = test_img.shape[1] // 2
            cy = test_img.shape[0] // 2
            real_x = int(fukuan / (self.img_size * self.cut_ratio) * cx) if fukuan else int(cx)
            real_y = int(
                4096 * self.standard_ratio_y *
                (cy / (self.img_size * self.cut_ratio) + image_id)
                + self.steel_real_y0
            )
            center_coords.append((real_x, real_y))
            area_list.append(10000)
        else:
            thres = float(self.seg_anomaly_thres)
            binary = np.where(amap >= thres, 255, 0).astype(np.uint8)
            binary = _morphological_closing(binary)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 若固定阈值下无轮廓但热力图有明显高值，则用自适应阈值再试一次（缓解 DRAEM 输出偏小导致漏检）
            if len(contours) == 0 and amap.size > 0:
                amax = float(np.max(amap))
                if amax > 0.002:
                    thres_adapt = max(0.002, 0.4 * amax)
                    binary = np.where(amap >= thres_adapt, 255, 0).astype(np.uint8)
                    binary = _morphological_closing(binary)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

                if fukuan:
                    real_area = int(
                        area
                        * (4096 * self.standard_ratio_y) / (self.img_size * self.cut_ratio)
                        * (fukuan / (self.img_size * self.cut_ratio))
                    )
                    real_x = int(fukuan / (self.img_size * self.cut_ratio) * cx)
                else:
                    real_area = int(area)
                    real_x = int(cx)

                real_y = int(
                    4096 * self.standard_ratio_y *
                    (cy / (self.img_size * self.cut_ratio) + image_id)
                    + self.steel_real_y0
                )

                H_img, W_img = test_img.shape[:2]
                crop_size = 128
                left = max(cx - crop_size // 2, 0)
                right = min(cx + crop_size // 2, W_img)
                top = max(cy - crop_size // 2, 0)
                bottom = min(cy + crop_size // 2, H_img)
                if right - left < crop_size:
                    left, right = (0, min(crop_size, W_img)) if left == 0 else (max(W_img - crop_size, 0), W_img)
                if bottom - top < crop_size:
                    top, bottom = (0, min(crop_size, H_img)) if top == 0 else (max(H_img - crop_size, 0), H_img)

                cropped = test_img[top:bottom, left:right]
                if cropped.size == 0:
                    continue

                area_list.append(real_area)
                center_coords.append((real_x, real_y))
                save_name = f"{real_x}_{real_y}_{real_area}_img{image_id}.png"
                PILImage.fromarray(cropped.astype(np.uint8)).save(
                    os.path.join(anomaly_save_path, save_name)
                )

        return 1 if len(center_coords) > 0 else 0, center_coords, area_list
