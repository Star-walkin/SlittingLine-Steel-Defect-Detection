import matplotlib
import random
import matplotlib.pyplot as plt
import cv2
from util import mean_smoothing, MSGMSLoss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd
import json
import serial
import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime, timedelta
import torch.nn.functional as F
import itertools

try:
    from scipy.ndimage import median_filter as scipy_median_filter
except Exception:
    scipy_median_filter = None


def flatten_background_subtraction(gray: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """
    背景拍平：大核中值滤波提取背景（低频），原图减背景 + 背景均值补偿，截断后输出。
    保留缺陷高频，剥离光照渐变，与 det_model/infer.py 中 SlidingWindowDetector 拍平一致。
    """
    h, w = gray.shape[:2]
    if kernel_size is None:
        kernel_size = max(51, min(w, h) // 8) | 1
    background_u8 = cv2.medianBlur(gray, kernel_size)
    img_float = gray.astype(np.float32)
    bg_float = background_u8.astype(np.float32)
    mean_val = float(np.mean(bg_float))
    flattened = img_float - bg_float + mean_val
    return np.clip(flattened, 0, 255).astype(np.uint8)


def flatten_background_projection(gray: np.ndarray) -> np.ndarray:
    """
    背景拍平：基于行/列均值的投影法。沿纵向求均值得到 1D 横向光照曲线，平铺成背景后减法拍平。
    适合左右光照渐变的带钢图像，极速且无局部光晕。与 det_model/infer.py 中 SlidingWindowDetector 拍平一致。
    """
    img_f = gray.astype(np.float32)
    H, W = img_f.shape
    bg_1d = np.mean(img_f, axis=0)  # (W,) 每列均值
    background = np.tile(bg_1d, (H, 1))  # (H, W)
    flattened = img_f - background
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(flattened, 0, 255).astype(np.uint8)


def flatten_background_median_div(gray: np.ndarray, ksize: int = None) -> np.ndarray:
    """
    背景拍平：中值滤波估计光照背景 + 除法，消除缓慢灰度渐变。
    与 det_model/infer.py 中 SlidingWindowDetector 的拍平逻辑一致，供 UNet 推理路径使用。
    """
    h, w = gray.shape[:2]
    if ksize is None:
        ksize = max(51, min(w, h) // 8) | 1
    background = cv2.medianBlur(gray, ksize).astype(np.float32)
    epsilon = 1e-5
    img_float = gray.astype(np.float32)
    flattened = img_float / (background + epsilon)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(flattened, 0, 255).astype(np.uint8)


def fix_json(json_data):
    def fix_last_three_brackets(json_data):
        # 确保数据从最后开始检查
        if json_data.count('[') > json_data.count(']'):
            # 找到最后一个 '[' 和 ']'
            open_count = json_data.count('[')
            close_count = json_data.count(']')

            # 如果右括号少于左括号，检查是否完整
            if open_count != close_count:
                # 找到最后一个不完整的最内层列表
                last_open = json_data.rfind('[')
                last_close = json_data.rfind(']')
                if last_open != -1 and last_close != -1:
                    # 删除最后一个最内层的列表部分
                    json_data = json_data[:last_close + 1]
                    # 更新括号数量
                    open_count = json_data.count('[')
                    close_count = json_data.count(']')
            # 补全右括号
            json_data += ']' * (open_count - close_count)
        else:
            json_data = json_data
        return json_data

    # 修复 JSON 数据
    fixed_json_data = fix_last_three_brackets(json_data)

    try:
        # 尝试解析修复后的 JSON 数据
        data = json.loads(fixed_json_data)
        return data
    except json.JSONDecodeError as e:
        print(f"修复失败: {e}")
        return None


def show_img(image):
    # image=cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def see_img_heatmap(data, segresult, ):  # 用来查看被转化为-1到1之间Tensor的图像B,C,H,W,
    '''应该对所有图像设置同一个阈值'''
    x2max = segresult.max()
    # segresult[segresult < 0.007] = 0
    x2min = segresult.min()
    segresult = (255 - 255 * (segresult - x2min) / (x2max - x2min))
    # show_img(segresult.astype(np.uint8))
    segresult = morphological_closing(segresult)
    # segresult = (255 - 255 * (segresult - min_list) / (max_list - min_list))
    # segresult[segresult>190]=255
    # show_img(data)
    overlay = data.copy()
    segresult = segresult.astype(np.uint8)
    # show_img(segresult)
    heatmap = cv2.applyColorMap(segresult, colormap=cv2.COLORMAP_JET)
    alpha = 0
    alpha2 = 0.3
    data = cv2.addWeighted(heatmap, alpha2, overlay, 1 - alpha, 0, overlay)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    show_img(data)
    return data


def morphological_closing(image):
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.dilate(image, kernel, iterations=1)
    return closed_image


# defect_images：从条带原图裁切时的边长（与模型输入 img_size 无关）
DEFECT_PATCH_SIDE = 256


def map_model_space_xy_to_strip_xy(
    cx: float,
    cy: float,
    tiles_u8: list,
    S: int,
    cut_l: int,
):
    """
    将拼接后的模型空间 (cx, cy)（与 test_cut / 热力图同尺寸）映射到
    送入 detect_ano 的条带图像像素坐标（列 x, 行 y，与裁边后条带对齐）。
    """
    n = len(tiles_u8)
    if n == 0:
        return float(cut_l + cx), float(cy)
    tile_idx = int(min(max(cy // S, 0), n - 1))
    cy_loc = float(cy) - tile_idx * S
    cx_loc = float(cx)
    tu = tiles_u8[tile_idx]
    th, tw = tu.shape[:2]
    ox = (cx_loc / float(S)) * float(tw)
    oy_in = (cy_loc / float(S)) * float(th)
    y_off = float(sum(tiles_u8[i].shape[0] for i in range(tile_idx)))
    oy = y_off + oy_in
    cx_strip = float(cut_l) + ox
    return cx_strip, float(oy)


def crop_square_from_image_u8(img: np.ndarray, cx: float, cy: float, side: int = DEFECT_PATCH_SIDE) -> np.ndarray:
    """以 (cx, cy) 为中心裁切 side×side；越界时用 BORDER_REFLECT101 填充。"""
    if img.ndim == 2:
        H, W = img.shape
    else:
        H, W = img.shape[:2]
    half = side // 2
    xi = int(round(cx))
    yi = int(round(cy))
    x0 = xi - half
    y0 = yi - half
    x1 = x0 + side
    y1 = y0 + side
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - W)
    pad_b = max(0, y1 - H)
    xs0 = max(0, x0)
    ys0 = max(0, y0)
    xs1 = min(W, x1)
    ys1 = min(H, y1)
    patch = img[ys0:ys1, xs0:xs1]
    if patch.size == 0:
        if img.ndim == 2:
            return np.zeros((side, side), dtype=np.uint8)
        return np.zeros((side, side, img.shape[2]), dtype=np.uint8)
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT101)
    if patch.shape[0] != side or patch.shape[1] != side:
        patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)
    return patch


class test_one_image(torch.nn.Module):
    def __init__(
        self,
        model,
        conduct_id,
        fukuan0,
        cut_ratio,
        img_size,
        seg_anomaly_thres,
        standard_ratio_x,
        standard_ratio_y,
        steel_real_y0,
        patchcore_k: float = 3.0,
        use_gradient_detection: bool = False,
        grad_threshold: float = 1.5,
        blur_ksize: int = 5,
        edge_crop: int = 20,
        bg_ksize: int = 101,
        diff_threshold: float = 1.0,
    ):
        """PatchCore anomaly detection class."""
        super(test_one_image, self).__init__()
        self.model = model
        self.cut_ratio = cut_ratio
        self.img_size = img_size
        self.conduct_id = conduct_id
        self.seg_anomaly_thres = seg_anomaly_thres
        self.standard_ratio_x = standard_ratio_x
        self.standard_ratio_y = standard_ratio_y
        self.steel_real_y0 = steel_real_y0
        self.fukuan0 = fukuan0
        # PatchCore 动态阈值中的 K，默认 3.0；可按相机在 config 中指定
        self.patchcore_k = float(patchcore_k)
        # 热力图梯度法缺陷检测参数（保留兼容，当前统一用局部背景相减）
        self.use_gradient_detection = bool(use_gradient_detection)
        self.grad_threshold = float(grad_threshold)
        self.blur_ksize = int(blur_ksize)
        self.edge_crop = int(edge_crop)
        # 局部背景相减法参数（四相机缺陷位置统一用此方式）
        self.bg_ksize = int(bg_ksize)
        self.diff_threshold = float(diff_threshold)

        # [修改 1] 适配单通道模型的 Transform
        self.transform_test = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # 单通道归一化，匹配 Tanh 输出范围 [-1, 1]
            transforms.Normalize([0.5], [0.5]),
        ])

        self.get_method = get_one_image_list(self.cut_ratio, self.standard_ratio_x, self.fukuan0)

        # ===== GPU 推理加速设置 =====
        self.model = self.model.cuda()
        self.model.eval()

        self.msgm = MSGMSLoss(in_channels=1)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def concatenate_images_vertical(self, tiles):
        if isinstance(tiles, list):
            return np.concatenate(tiles, axis=0)
        else:
            return np.concatenate([tiles[i] for i in range(tiles.shape[0])], axis=0)

    def get_image_cut0(self, image, left_edge, right_edge, bias=15):
        """
        保持不变，负责将图片转灰度并切片
        """
        # 统一成 2D 灰度
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        H, W = gray.shape[:2]

        l = max(int(left_edge + bias), 0)
        r = min(int(right_edge - bias), W)
        if r <= l:
            l, r = 0, W

        img_cut = gray[:, l:r]
        cut_H, cut_W = img_cut.shape

        MIN_CUT_W = 64
        if cut_W < MIN_CUT_W:
            img_cut = gray
            cut_H, cut_W = img_cut.shape

        tile_height = cut_H // self.cut_ratio
        MIN_TILE_H = 64
        if tile_height < MIN_TILE_H:
            return [img_cut for _ in range(self.cut_ratio)]

        tile_list = []
        for i in range(self.cut_ratio):
            y0 = i * tile_height
            y1 = (i + 1) * tile_height if i < self.cut_ratio - 1 else cut_H
            tile = img_cut[y0:y1, :]
            if tile.size == 0:
                tile = img_cut
            tile_list.append(tile)

        return tile_list

    def data_prepare(self, image, left_edge=0, right_edge=None, bias=15):

        if right_edge is None:
            right_edge = image.shape[1]

        tile_list = self.get_image_cut0(image, left_edge, right_edge, bias=bias)

        test_images = []
        tiles_u8 = []

        for tile in tile_list:
            # tile 已经是 2D 灰度，无需转 BGR
            # 如果需要兼容后续可视化，可以保留 tile 副本，但不要转 3 通道送入模型
            tiles_u8.append(tile.copy())

            # 直接从灰度数组生成 PIL
            tile_pil = Image.fromarray(tile)

            # transform_test 现在只做单通道处理
            t = self.transform_test(tile_pil)  # -> [1, S, S]
            test_images.append(t)

        # 堆叠后形状: [Batch, 1, S, S]
        return torch.stack(test_images, dim=0), tiles_u8

    def detect_ano(self, image, left_edge=0, right_edge=None):
        """
        [修改 3] 推理逻辑适配单通道输出
        """
        device = next(self.model.parameters()).device
        if right_edge is None:
            right_edge = image.shape[1]

        # 转灰度并做背景拍平（与 SlidingWindowDetector 一致，消除缓慢灰度渐变）
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.asarray(image, dtype=np.uint8)
        gray = flatten_background_subtraction(gray)

        # 与 data_prepare / get_image_cut0 一致的左右裁边（用于缺陷图从原条带映射坐标）
        Hg, Wg = gray.shape[:2]
        bias = 15
        l_edge = max(int(left_edge + bias), 0)
        r_edge = min(int(right_edge - bias), Wg)
        if r_edge <= l_edge:
            l_edge, r_edge = 0, Wg

        # 1) 准备数据 (Tensor shape: [B, 1, S, S])
        test_img, tiles_u8 = self.data_prepare(gray, left_edge, right_edge, bias=15)
        self._last_tiles_u8 = tiles_u8
        self._last_cut_l = l_edge
        self._last_cut_r = r_edge
        self._last_detect_mode = "tiled"
        test_img = test_img.to(device, non_blocking=True)

        # 2) 推理
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            rec_img = self.model(test_img)  # 输出 shape: [B, 1, S, S]

            # 计算差异图
            amap = self.msgm(test_img, rec_img, as_loss=False)
            amap = mean_smoothing(amap)

        # 3) 拼接 test_cut (原图)
        # tiles_u8 已经是灰度列表，直接 Resize
        S = self.img_size
        tiles_u8_rs = [cv2.resize(tu, (S, S), interpolation=cv2.INTER_LINEAR) for tu in tiles_u8]
        test_cut = self.concatenate_images_vertical(tiles_u8_rs)  # (H_total, W) 2D 灰度

        # 4) [修改] 拼接 rec_cut (重构图)
        # 此时 rec_img 是 [B, 1, S, S]
        rec_np = rec_img.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, S, S, 1]

        # 反归一化: (-1, 1) -> (0, 255)
        rec_cut_float = (rec_np * 0.5 + 0.5) * 255

        # 去掉最后的通道维度 1 -> [B, S, S]
        rec_cut_float = rec_cut_float.squeeze(-1)

        rec_cut = self.concatenate_images_vertical(rec_cut_float).astype(np.uint8)
        # 此时 rec_cut 是 2D 灰度图 (H, W)

        # 5) 拼接 amap_cut (热力图)
        tiles_amap = amap.permute(0, 2, 3, 1).detach().cpu().numpy()
        amap_cut = self.concatenate_images_vertical(tiles_amap).squeeze(-1).astype(np.float32)

        return test_cut, rec_cut, amap_cut

    def _save_defect_patch_from_detection(
        self,
        anomaly_save_path,
        save_name,
        test_img,
        cx,
        cy,
        fallback_crop_size,
        original_strip_for_crop,
    ):
        """
        优先：根据 detect_ano 缓存的 tiles_u8，将模型空间 (cx,cy) 映射到条带原图并裁 DEFECT_PATCH_SIDE 正方形。
        否则：在拼接后的 test_img 上按 fallback_crop_size 裁切（旧行为）。
        返回是否成功写出文件。
        """
        out_path = os.path.join(anomaly_save_path, save_name)
        mode = getattr(self, "_last_detect_mode", "tiled")
        # SlidingWindowDetector：整幅条带 crop 一次性 resize 成 test_cut，与 tile 拼接不同
        if (
            original_strip_for_crop is not None
            and mode == "sliding"
        ):
            cH, cW = getattr(self, "_last_sliding_crop_hw", (0, 0))
            out_H, out_W = getattr(self, "_last_test_cut_hw", (1, 1))
            cut_l = getattr(self, "_last_cut_l", 0)
            if cH > 0 and cW > 0 and out_H > 0 and out_W > 0:
                cx_s = cut_l + (float(cx) / float(out_W)) * float(cW)
                cy_s = (float(cy) / float(out_H)) * float(cH)
                patch = crop_square_from_image_u8(original_strip_for_crop, cx_s, cy_s, DEFECT_PATCH_SIDE)
                cv2.imwrite(out_path, patch)
                return True
        tiles_u8 = getattr(self, "_last_tiles_u8", None)
        cut_l = getattr(self, "_last_cut_l", 0)
        S = self.img_size
        if (
            original_strip_for_crop is not None
            and tiles_u8 is not None
            and len(tiles_u8) > 0
        ):
            cx_s, cy_s = map_model_space_xy_to_strip_xy(float(cx), float(cy), tiles_u8, S, cut_l)
            patch = crop_square_from_image_u8(original_strip_for_crop, cx_s, cy_s, DEFECT_PATCH_SIDE)
            cv2.imwrite(out_path, patch)
            return True

        H, W = test_img.shape[:2]
        crop_size = fallback_crop_size
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
        cropped = test_img[top:bottom, left:right]
        if cropped.size == 0:
            return False
        if cropped.ndim == 3 and cropped.shape[2] == 1:
            cropped = np.squeeze(cropped, axis=2)
        elif cropped.ndim != 2 and cropped.shape[2] != 3:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        Image.fromarray(cropped.astype(np.uint8)).save(out_path)
        return True

    def obtain_anomaly_location(self, amap, test_img, anomaly_save_path, image_id, fukuan=None, original_strip_for_crop=None):
        """
        缺陷位置检测统一为局部背景相减法：热力图 -> 大核平滑得背景 -> H_diff = H - H_bg -> 阈值 + valid_mask -> 形态学闭运算 -> 轮廓 -> 外接矩形中心 + 面积。
        当局部对比度模块不可用时回退到原逻辑（全图异常中心 + 阈值法）。

        original_strip_for_crop: 可选，与 detect_ano 输入同一条带原图（H×W 或 H×W×3）。
            若提供且 detect_ano 已缓存 _last_tiles_u8，则 defect_images 从原图裁 DEFECT_PATCH_SIDE 正方形；
            JSON 中的 real_x/real_y 仍按原公式写入（业务坐标系），与像素裁剪独立。
        """
        center_coords, area_list = [], []

        try:
            from patchcore_model.local_contrast_defect import detect_defects_by_local_contrast
            local_contrast_available = True
        except ImportError:
            detect_defects_by_local_contrast = None
            local_contrast_available = False

        if local_contrast_available:
            # ---------- 统一仅用局部背景相减法 ----------
            Hm, Wm = amap.shape
            ec = max(0, min(getattr(self, "edge_crop", 20), Hm // 2 - 1, Wm // 2 - 1))
            valid_mask = np.zeros_like(amap, dtype=np.uint8)
            valid_mask[ec : Hm - ec, ec : Wm - ec] = 255
            bg_k = getattr(self, "bg_ksize", 101)
            diff_th = getattr(self, "diff_threshold", 1.0)
            amap_bin, _ = detect_defects_by_local_contrast(
                amap.astype(np.float32), valid_mask,
                bg_ksize=bg_k, diff_threshold=diff_th
            )
            amap_bin = morphological_closing(amap_bin)
            contours, _ = cv2.findContours(amap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fukuan_val = fukuan if fukuan is not None else (self.img_size * self.cut_ratio)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

                real_area = int(
                    area
                    * (4096 * self.standard_ratio_y) / (self.img_size * self.cut_ratio)
                    * (fukuan_val / (self.img_size * self.cut_ratio))
                )
                real_x = int(fukuan_val / (self.img_size * self.cut_ratio) * cx)
                real_y = int(4096 * self.standard_ratio_y *
                             (cy / (self.img_size * self.cut_ratio) + image_id)
                             + self.steel_real_y0)

                save_name = f"{real_x}_{real_y}_{real_area}_img{image_id}.png"
                if not self._save_defect_patch_from_detection(
                    anomaly_save_path,
                    save_name,
                    test_img,
                    cx,
                    cy,
                    32,
                    original_strip_for_crop,
                ):
                    continue
                area_list.append(real_area)
                center_coords.append((real_x, real_y))
            return 1 if len(center_coords) > 0 else 0, center_coords, area_list

        # ---------- 回退：局部对比度模块不可用时保留原逻辑 ----------
        try:
            from patchcore_model.gradient_defect import detect_defects_by_gradient
            gradient_available = True
        except ImportError:
            detect_defects_by_gradient = None
            gradient_available = False

        if gradient_available:
            Hm, Wm = amap.shape
            ec = max(0, min(getattr(self, "edge_crop", 20), Hm // 2 - 1, Wm // 2 - 1))
            valid_mask = np.zeros_like(amap, dtype=np.uint8)
            valid_mask[ec : Hm - ec, ec : Wm - ec] = 255
            blur_k = getattr(self, "blur_ksize", 5)
            grad_th = getattr(self, "grad_threshold", 1.5)
            amap_bin, _ = detect_defects_by_gradient(
                amap.astype(np.float32), valid_mask,
                blur_ksize=blur_k, grad_threshold=grad_th
            )
            amap_bin = morphological_closing(amap_bin)
            contours, _ = cv2.findContours(amap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fukuan_val = fukuan if fukuan is not None else (self.img_size * self.cut_ratio)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                real_area = int(
                    area * (4096 * self.standard_ratio_y) / (self.img_size * self.cut_ratio)
                    * (fukuan_val / (self.img_size * self.cut_ratio))
                )
                real_x = int(fukuan_val / (self.img_size * self.cut_ratio) * cx)
                real_y = int(4096 * self.standard_ratio_y *
                             (cy / (self.img_size * self.cut_ratio) + image_id) + self.steel_real_y0)
                save_name = f"{real_x}_{real_y}_{real_area}_img{image_id}.png"
                if not self._save_defect_patch_from_detection(
                    anomaly_save_path,
                    save_name,
                    test_img,
                    cx,
                    cy,
                    128,
                    original_strip_for_crop,
                ):
                    continue
                area_list.append(real_area)
                center_coords.append((real_x, real_y))
            return 1 if len(center_coords) > 0 else 0, center_coords, area_list
        if amap.mean() > self.seg_anomaly_thres + 0.05:
            cx = test_img.shape[1] // 2
            cy = test_img.shape[0] // 2
            real_x = int(cx) if fukuan is None else int(
                fukuan / (self.img_size * self.cut_ratio) * cx
            )
            real_y = int(
                4096 * self.standard_ratio_y *
                (cy / (self.img_size * self.cut_ratio) + image_id)
                + self.steel_real_y0
            )
            center_coords.append((real_x, real_y))
            area_list.append(10000)
            return 1, center_coords, area_list

        valid_mask = amap > 0
        valid_scores = amap[valid_mask]
        if valid_scores.size > 10:
            mu = float(np.median(valid_scores))
            sigma = float(np.std(valid_scores))
            k = float(getattr(self, "patchcore_k", 3.0))
            T_min = float(self.seg_anomaly_thres)
            T_dynamic = mu + k * sigma
            cond_absolute = amap > T_min
            cond_relative = amap > T_dynamic
            final_bool = cond_absolute & cond_relative & valid_mask
        else:
            final_bool = amap >= self.seg_anomaly_thres
        amap_bin = np.zeros_like(amap, dtype=np.uint8)
        amap_bin[final_bool] = 255
        amap_bin = morphological_closing(amap_bin)
        contours, _ = cv2.findContours(amap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fukuan_val = fukuan if fukuan is not None else (self.img_size * self.cut_ratio)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            real_area = int(
                area
                * (4096 * self.standard_ratio_y) / (self.img_size * self.cut_ratio)
                * (fukuan_val / (self.img_size * self.cut_ratio))
            )
            real_x = int(fukuan_val / (self.img_size * self.cut_ratio) * cx)
            real_y = int(4096 * self.standard_ratio_y *
                         (cy / (self.img_size * self.cut_ratio) + image_id)
                         + self.steel_real_y0)
            save_name = f"{real_x}_{real_y}_{real_area}_img{image_id}.png"
            if not self._save_defect_patch_from_detection(
                anomaly_save_path,
                save_name,
                test_img,
                cx,
                cy,
                128,
                original_strip_for_crop,
            ):
                continue
            area_list.append(real_area)
            center_coords.append((real_x, real_y))

        return 1 if len(center_coords) > 0 else 0, center_coords, area_list

class get_one_image_list():
    def __init__(self, cut_ratio, standard_ratio_x, fukuan0):
        """PatchCore anomaly detection class."""
        super(get_one_image_list, self).__init__()

        self.cut_ratio = cut_ratio
        self.standard_ratio_x = standard_ratio_x
        self.fukuan0 = fukuan0

    def find_plate_edges_by_intensity(self, image, fukuan_mm=None):
        """
        返回 left_edge, right_edge（像素索引，基于 512x512 缩放结果）。
        如果传入 fukuan_mm 则使用该值去估计搜索窗口，否则使用 self.fukuan0（若为 list 则取第一个）。
        """
        # 以 512x512 进行简化计算（和原函数一致）
        img512 = cv2.resize(image, (512, 512))
        gray_img = cv2.cvtColor(img512, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        # 计算左右区域的粗位置（基于 fukuan_mm）
        if fukuan_mm is None:
            # 回退策略：如果 self.fukuan0 是列表，取第一个；否则直接用它
            if isinstance(self.fukuan0, (list, tuple)):
                fukuan_val = float(self.fukuan0[0])
            else:
                fukuan_val = float(self.fukuan0)
        else:
            fukuan_val = float(fukuan_mm)

        # 将 fukuan(mm) -> px（基于 self.standard_ratio_x）
        # 注意：这里保持你原来算法的缩放因子（/8/2 等）的语义不变
        # 可能需要针对你相机实际参数微调这些常数，但先保留原式逻辑
        try:
            offset = int(fukuan_val / self.standard_ratio_x / 8 / 2)
        except Exception:
            offset = int(fukuan_val / (self.standard_ratio_x + 1e-6) / 8 / 2)

        left = int(256 - offset - 15)
        right = int(256 + offset - 15)

        # bounds check
        left = max(5, left)
        right = min(512 - 30 - 5, right)

        # 计算梯度列并返回（原逻辑用 left,right 截取局部梯度）
        left_grad = np.abs(grad_x[:, left:left + 30])
        right_grad = np.abs(grad_x[:, right:right + 30])
        left_gradient_sum = np.sum(left_grad, axis=0)
        right_gradient_sum = np.sum(right_grad, axis=0)
        left_edge = 8 * (np.argmax(left_gradient_sum) + left)
        right_edge = 8 * (np.argmax(right_gradient_sum) + right)

        return left_edge, right_edge

    def measure_all_strips(self, image, num_strips=None):
        """
        自动检测多条带钢的幅宽信息。
        返回列表 [(fukuan, left, right), ...]
        """
        gray = cv2.cvtColor(cv2.resize(image, (512, 512)), cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        column_sum = np.sum(np.abs(grad_x), axis=0)
        smooth = cv2.GaussianBlur(column_sum.reshape(1, -1), (1, 31), 0).flatten()
        peaks = np.where(smooth > np.max(smooth) * 0.4)[0]  # 动态阈值检测边缘
        # 将峰值分组
        boundaries = np.split(peaks, np.where(np.diff(peaks) > 30)[0] + 1)
        edges = [int(np.mean(b)) for b in boundaries if len(b) > 3]

        strips = []
        for i in range(0, len(edges) - 1, 2):
            left_edge, right_edge = edges[i], edges[i + 1]
            fukuan = math.ceil((right_edge - left_edge) * self.standard_ratio_x)
            strips.append((fukuan, left_edge, right_edge))

        if num_strips:
            strips = strips[:num_strips]
        return strips

    def get_image_cut(self, img_cut):
        # 假设输入图像是一个 PyTorch Tensor，形状为 (C, H, W)

        # 裁剪图像
        # img_cut = image[:, :, left_edge + bias:right_edge - bias]

        # 获取图像的高度和宽度
        _, height, width = img_cut.shape

        # 计算每个小图片的高度和宽度
        tile_height = height // self.cut_ratio
        tile_width = width // self.cut_ratio
        tile_list = []

        # 分割图像并保存
        for i in range(self.cut_ratio):
            for j in range(self.cut_ratio):
                # 计算小图片的坐标
                x = j * tile_width
                y = i * tile_height
                # 截取小图片
                tile = img_cut[:, y:y + tile_height, x:x + tile_width]
                tile_list.append(tile)

        return tile_list


class Consecutive_anomaly_Checker:
    def __init__(self, thres_num, calibrate_cam):
        self.n = thres_num
        self.calibrate_cam = calibrate_cam
        self.sequence = []

    def add_number(self, num):
        self.sequence.append(num)
        if len(self.sequence) > self.n:
            self.sequence.pop(0)

        if self.check_consecutive_ones():
            try:
                ser = serial.Serial('COM4', 9600, timeout=1)
                payload = bytes([0xA0, 0x01, 0x01, 0xA2])
                ser.write(payload)
                ser.close()
                print("alarm sent")
            except Exception as e:
                # 不要让检测线程挂掉
                print(f"[WARN] serial alarm failed: {e}")
            finally:
                self.sequence.clear()

    def check_consecutive_ones(self):
        # if now_camid ==self.calibrate_cam:
        if len(self.sequence) < self.n:
            return False
        return all(x == 1 for x in self.sequence[-self.n:])


class Statistic_anomaly:
    def __init__(self, conduct_id, fukuan, range, result_path, start_time, remove_threshold, steel_length_range,
                 update_info, standard_area_tables, colors, class_labels, cls_all, area_range):
        self.fukuan = fukuan
        # 设置字体为新罗马字体
        self.range = range
        self.conduct_id = conduct_id
        self.result_path = result_path
        self.start_time = start_time
        self.remove_threshold = remove_threshold
        self.steel_length_range = steel_length_range
        self.cls_all = cls_all
        self.area_range = area_range
        self.update_info = update_info
        self.standard_area_tables = standard_area_tables
        self.final_area_tables = {"上": [], "下": []}
        self.class_labels = class_labels
        self.colors = colors
        # 共享长度坐标：同一 strip 上下表面取最大值，保证两张分布图 xlim 一致
        # 结构：{strip_id: {"上": {"coords":..., "xmax":...}, "下": {...}}}
        self._dist_pending = {}

    @staticmethod
    def _compute_xmax_km_from_coords(coordinates, fallback=1.0):
        try:
            all_y = []
            for pts in coordinates.values():
                for (_x, y) in pts:
                    all_y.append(float(y))
            if all_y:
                return max(all_y)
        except Exception:
            pass
        return float(fallback)

    def select_anomalies(self, anomalies_info, strip_id, print_cls, surface, updates):
        select_info_coords = {}
        select_info_area = {}

        # 初始化
        for cls in self.cls_all:
            select_info_area[int(cls)] = []
            select_info_coords[int(cls)] = []

        clean_info = {int(k): v for k, v in anomalies_info.items()}

        print(f"\n--- {surface}表面 诊断开始 ---")
        print(f"当前判定标准: 长度范围={self.steel_length_range}, 面积范围={self.area_range}")

        for cls_raw in print_cls:
            cls = int(cls_raw)
            if cls not in clean_info:
                continue

            print_anomalies = clean_info[cls]
            print(f"类别 {cls} 待处理数据量: {len(print_anomalies)}")

            for x_coords, y_coords, area in print_anomalies:
                x_val = float(x_coords)
                y_val = round(float(y_coords) / 1e6, 3)
                area_val = float(area)

                # --- 深度诊断打印 ---
                len_ok = True
                if len(self.steel_length_range) == 2:
                    len_ok = self.steel_length_range[0] <= y_val <= self.steel_length_range[1]

                area_ok = True
                if len(self.area_range) == 2:
                    area_ok = self.area_range[0] <= area_val <= self.area_range[1]
                elif len(self.area_range) == 1:
                    area_ok = area_val >= self.area_range[0]

                # 如果数据被过滤，打印原因
                if not (len_ok and area_ok):
                    print(f"  [过滤] 坐标({x_val}, {y_val}) 面积({area_val}): "
                          f"长度通过={len_ok}, 面积通过={area_ok}")
                else:
                    select_info_area[cls].append(area_val)
                    select_info_coords[cls].append((x_val, y_val))

        total_found = sum(len(v) for v in select_info_coords.values())
        print(f"--- {surface}表面 最终筛选出 {total_found} 个缺陷 ---\n")

        # 映射中文标签和绘图逻辑（保持之前的修改）
        final_coords_for_draw = {}
        label_map = {int(k): v for k, v in self.class_labels.items()}
        for cls_id, pts in select_info_coords.items():
            label_text = label_map.get(cls_id, str(cls_id))
            final_coords_for_draw[label_text] = pts

        standard_area_table, indices_dict = self.create_area_table(select_info_area)

        # 若开启“报告修改”模式，允许通过 updates_* 覆盖表格计数，并同步筛选参与绘图的缺陷点
        if self.update_info and isinstance(updates, list) and len(updates) > 0:
            try:
                standard_area_table, indices_dict = self.batch_update_table_and_indices(
                    standard_area_table, indices_dict, updates
                )
                merged_indices = self.merge_indices_by_class(indices_dict)  # class_label -> [idx,...]

                # 根据 merged_indices 对坐标做同样的抽样（索引与 select_info_coords/area 一致）
                final_coords_for_draw = {}
                inv_label_map = {v: int(k) for k, v in label_map.items()}
                for class_label_text, idx_list in merged_indices.items():
                    cls_id = inv_label_map.get(class_label_text)
                    if cls_id is None:
                        continue
                    src_pts = select_info_coords.get(int(cls_id), [])
                    picked = [src_pts[i] for i in idx_list if 0 <= i < len(src_pts)]
                    final_coords_for_draw[class_label_text] = picked
            except Exception as e:
                # 修改失败不影响原始报告生成，回退到未修改版本
                print(f"[WARN] 应用报告修改 updates 失败，已回退原始统计：{e}")

        # 先缓存分布图数据，待上下表面都准备好后统一用同一 xmax 绘制
        xmax_km = self.steel_length_range[-1] if self.steel_length_range else 0
        if xmax_km <= 0:
            xmax_km = self._compute_xmax_km_from_coords(final_coords_for_draw, fallback=1.0)

        self._dist_pending.setdefault(strip_id, {})
        self._dist_pending[strip_id][surface] = {
            "coords": final_coords_for_draw,
            "xmax": float(xmax_km),
        }

        # 如果上下表面都齐了，则取最大长度统一绘制两张分布图
        if "上" in self._dist_pending[strip_id] and "下" in self._dist_pending[strip_id]:
            xmax_shared = max(
                self._dist_pending[strip_id]["上"]["xmax"],
                self._dist_pending[strip_id]["下"]["xmax"],
            )
            self.draw_anomaly_distribution(
                self._dist_pending[strip_id]["上"]["coords"], strip_id, surface="上", xmax_override=xmax_shared
            )
            self.draw_anomaly_distribution(
                self._dist_pending[strip_id]["下"]["coords"], strip_id, surface="下", xmax_override=xmax_shared
            )
            # 清理该 strip 的缓存，避免跨 strip 串扰
            try:
                del self._dist_pending[strip_id]
            except Exception:
                pass

        self.draw_anomaly_area_cls(standard_area_table, strip_id, surface=surface)
        self.final_area_tables[surface] = standard_area_table

    # 根据索引字典A，从字典B提取对应的数据
    # 根据索引字典A，从字典B提取对应的数据
    def extract_data_by_indices(self, indices_dict_A, data_dict_B, class_mapping):
        result_dict = {}
        for class_label, indices in indices_dict_A.items():
            # 根据类别名称找到对应的数字类别
            # 使用 class_mapping 查找数字类别
            numeric_class = [key for key, value in class_mapping.items() if value == class_label]
            if not numeric_class:
                continue  # 如果没有找到映射，跳过该类别
            numeric_class = numeric_class[0]  # 获取数字类别
            # 获取字典B中对应类别的数据
            class_data = data_dict_B.get(numeric_class, [])
            # 根据索引从字典B中提取数据
            result_dict[class_label] = [class_data[i] for i in indices if i < len(class_data)]  # 防止索引越界

        return result_dict

    def judje_product(self, image,strip_id):

        # ==============================
        #  1）先计算是否有超标
        # ==============================
        is_qualifieds = []
        df_standard = self.standard_area_tables

        for surface in self.final_area_tables.keys():
            difference = self.final_area_tables[surface] - df_standard
            is_qualifieds.append(difference <= 0)

        # 是否存在任意不合格项
        is_bad = (
                (is_qualifieds[0].values == False).any() or
                (is_qualifieds[1].values == False).any()
        )

        # ==============================
        #  2）准备字体加载（包含fallback）
        # ==============================
        from PIL import ImageDraw, ImageFont

        def load_font(size):
            font_path = os.path.join(_REPO_ROOT, "simhei.ttf")
            try:
                return ImageFont.truetype(font_path, size)
            except:
                return ImageFont.load_default()

        font = load_font(40)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # 文本内容
        text = "不合格" if is_bad else "合格"
        circle_color = (255, 0, 0)
        text_color = (255, 0, 0)

        # ==============================
        #  3）计算文字尺寸（替代 textsize）
        # ==============================
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x = (width - text_w) // 2
        text_y = (height - text_h) // 2

        # ==============================
        #  4）绘制文字
        # ==============================
        draw.text((text_x, text_y), text, font=font, fill=text_color)

        # ==============================
        #  5）绘制红圈
        # ==============================
        circle_radius = 100
        cx, cy = width // 2, height // 2
        draw.ellipse(
            (cx - circle_radius, cy - circle_radius,
             cx + circle_radius, cy + circle_radius),
            outline=circle_color,
            width=5
        )

        # ==============================
        #  6）保存图片
        # ==============================
        if self.update_info:
            save_path = os.path.join(self.result_path, f"表面检测报告-strip_{strip_id+1}-修正.png")
        else:
            save_path = os.path.join(self.result_path, f"表面检测报告-strip_{strip_id+1}.png")

        image.save(save_path)
        image.show()

    def create_area_table(self, data):
        # 获取最大区间的上限值
        max_end = self.range[-1][1]
        # 创建列标题
        columns = [f"{start}-{end}" for start, end in self.range]
        columns.append(f"> {max_end}")  # 添加 >最大值 的列标题

        # 创建一个空的 DataFrame，行是类别标签，列是区间
        area_table = pd.DataFrame(0, index=[self.class_labels.get(class_id, class_id) for class_id in data.keys()],
                                  columns=columns)
        # 创建一个字典来存储每个类别每个区间的原始索引
        indices_dict = {class_label: {f"{start}-{end}": [] for start, end in self.range}
                        for class_label in self.class_labels.values()}
        for class_id in data.keys():
            class_label = self.class_labels.get(class_id, class_id)
            indices_dict.setdefault(class_label, {f"{start}-{end}": [] for start, end in self.range})
            indices_dict[class_label].setdefault(f"> {max_end}", [])
            # 对于大于最大区间的情况angjian2
        # 遍历字典中的每个类别
        for class_id, areas in data.items():
            # 获取对应类别标签
            class_label = self.class_labels.get(class_id, class_id)
            # 遍历每个类别的面积
            for idx, area in enumerate(areas):
                # 根据面积确定它属于哪个区间
                for i, (start, end) in enumerate(self.range):
                    if start <= area < end:
                        # 计数这个类别在该区间的面积
                        area_table.loc[class_label, f"{start}-{end}"] += 1
                        # 记录索引
                        indices_dict[class_label][f"{start}-{end}"].append(idx)
                        break  # 找到对应区间后跳出循环
                else:
                    # 如果面积大于最大区间的结束值
                    if area >= max_end:
                        area_table.loc[class_label, f"> {max_end}"] += 1
                        indices_dict[class_label][f"> {max_end}"].append(idx)

        return area_table, indices_dict

    # 批量更新函数
    def batch_update_table_and_indices(self, area_table, indices_dict, updates):
        """
        批量更新表格和索引字典。

        参数:
        area_table: DataFrame, 存储类别和区间的表格
        indices_dict: dict, 存储类别和区间对应的索引
        updates: list, 包含更新信息的字典列表，每个字典包含类标签、区间、和新计数值
        """
        for update in updates:
            class_label = update['class_label']
            area_interval = update['area_interval']
            area_interval = area_interval.strip("[]").replace(",", "-")
            area_interval = area_interval.replace(" ", "")
            new_count = update['new_count']

            # 获取原始数目
            original_count = area_table.loc[class_label, area_interval]

            # 检查新数目是否大于原始数目
            if new_count > original_count:
                print(
                    f"警告: {class_label} 类别的 {area_interval} 区间的新数目 ({new_count}) 超出了原始数目 ({original_count})，将其限制为原始数目")
                new_count = original_count  # 或者抛出异常 raise ValueError("New count exceeds original count")

            # 更新表格中的数据
            area_table.loc[class_label, area_interval] = new_count

            # 获取该类别和区间的索引列表
            current_indices = indices_dict[class_label][area_interval]

            # 如果该区间有索引，随机选择一个索引并修改
            if current_indices:
                # 更新表格
                area_table.loc[class_label, area_interval] = new_count
                # 随机选择与新数目相同数量的索引
                selected_indices = random.sample(current_indices, new_count)
                # 更新索引字典
                indices_dict[class_label][area_interval] = selected_indices
        # 返回更新后的表格和索引字典
        return area_table, indices_dict

    # 更新索引字典，将每个类别下的所有区间的索引合并在一起
    def merge_indices_by_class(self, indices_dict):
        merged_dict = {}

        for class_label, intervals in indices_dict.items():
            # 将所有区间的索引合并在一起
            merged_indices = []
            for interval, indices in intervals.items():
                merged_indices.extend(indices)  # 使用extend将多个区间的索引合并

            merged_dict[class_label] = merged_indices

        return merged_dict

    def draw_anomaly_distribution(self, coordinates, strip_id, surface, xmax_override=None):
        # 创建一个新的图像
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        fig, ax = plt.subplots(figsize=(10, 3))
        # 绘制每个类别的点
        for class_label, points in coordinates.items():
            print("示例缺陷坐标前 5 个：", list(points)[:5])

            if len(points) > 0:
                # 拆解坐标
                x_vals, y_vals = zip(*points)
                ax.scatter(y_vals, x_vals,  # <——交换横纵轴
                           color=self.colors[class_label],
                           label=class_label,
                           s=8)
        # 添加图例
        ax.legend()

        # 启用网格，并且设置为细而密集
        ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5)  # 设置网格线颜色为灰色
        # 设置 x 轴和 y 轴范围
        # 设置 x 轴从 0 开始
        # ax.set_xlim(left=0)  # 设置 x 轴的最小值为 0
        ax.set_ylim(0, self.fukuan)  # 假设 y 轴范围也设置为0-1000，如果需要其他范围可调整此行
        # 假设坐标是 km；若传入 xmax_override，则上下表面统一取该值
        if xmax_override is not None:
            xmax = float(xmax_override)
        else:
            xmax = self.steel_length_range[-1] if self.steel_length_range else 0
            if xmax <= 0:
                xmax = self._compute_xmax_km_from_coords(coordinates, fallback=1.0)
        ax.set_xlim(0, xmax)

        # rcParams['font.family'] = 'serif'
        rcParams['font.sans-serif'] = ['SimHei']
        # 设置坐标轴刻度的字体和大小
        plt.xticks(fontname='Times New Roman', fontsize=10)
        plt.yticks(fontname='Times New Roman', fontsize=10, )
        # 设置图像的标题和标签
        ax.set_title(f"缺陷分布图_{surface}表面", fontname='SimHei', fontweight='bold', fontsize=11)
        ax.set_xlabel('钢带长度:Km', fontname='SimHei', fontweight='bold', fontsize=10)
        ax.set_ylabel(f"钢带宽度:mm", fontname='SimHei', fontweight='bold', fontsize=10)
        plt.savefig(
            os.path.join(self.result_path, f"strip_{strip_id+1}_anomaly_distribution_{surface}表面.png"),
            bbox_inches="tight",
            pad_inches=0.1
        )

        # 显示图像
        # plt.show(block=True)
        # return self.mask

    def draw_anomaly_area_cls(self, area_cls, strip_id,surface):
        df = area_cls
        # 你可以根据你的系统安装情况选择适合的字体
        font_path = os.path.join(_REPO_ROOT, "simhei.ttf")  # 替换为实际路径
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()

        # 绘制表格
        fig, ax = plt.subplots(figsize=(10, 5))  # 调整表格大小
        ax.axis('off')  # 关闭坐标轴
        # 创建表格
        table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center')
        # 设置字体和行距
        for key, cell in table.get_celld().items():
            cell.set_height(0.08)  # 设置行高
            cell.visible_edges = 'horizontal'

        # 设置表格字体大小
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        # 去掉白边显示图像
        plt.margins(0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        # 添加标题
        # ax.set_title(f"缺陷统计表-{surface}表面", fontname='SimHei', fontweight='bold', fontsize=14,pad=-8)
        # 保存表格为图片
        plt.savefig(
            os.path.join(self.result_path, f"strip_{strip_id+1}_anomaly_area_cls_{surface}表面.png"),
            bbox_inches="tight",
            pad_inches=0.1
        )

        # 显示表格
        # plt.show()



    def gen_report(self, up_dis, up_cls, down_dis, down_cls,
                   fukuan_max, fukuan_min, fukuan_fig):

        # ---------------------------------------
        # 1）准备画布与字体
        # ---------------------------------------
        up_dis_y = 190
        up_cls_crop = up_cls.crop((0, 125, up_cls.width, 290))
        up_cls_y = up_dis_y + up_dis.height + 20

        bottom_start = up_cls_y + up_cls_crop.height + 50
        down_dis_y = bottom_start + 30

        down_cls_crop = down_cls.crop((0, 125, down_cls.width, 320))
        down_cls_y = bottom_start + down_dis.height + 50

        fukuan_y = down_cls_y + down_cls_crop.height + 40

        total_height_needed = fukuan_y + fukuan_fig.height + 60

        # 画布宽度不变，高度自动撑开
        canvas_width = 1000
        canvas_height = max(1500, total_height_needed)

        # ---------- 创建画布 ----------
        image = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # 字体路径（只改这里）
        font_path = os.path.join(_REPO_ROOT, "simhei.ttf")

        # 字体加载函数（带fallback）
        def load_font(size):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                return ImageFont.load_default()

        font_title = load_font(22)
        font_main = load_font(16)

        # 计算文本宽高的函数
        def text_size(text, font):
            box = draw.textbbox((0, 0), text, font=font)
            return box[2] - box[0], box[3] - box[1]

        # ---------------------------------------
        # 2）标题栏
        # ---------------------------------------
        title = "钢带表面缺陷分析报告"
        w, h = text_size(title, font_title)
        draw.text(((canvas_width - w) // 2, 30),
                  title, fill="black", font=font_title)

        # ---------------------------------------
        # 3）基本信息文字区
        # ---------------------------------------
        current_time = Path(self.result_path).parts[2]  # 文件夹名中的日期

        draw.text((80, 90), f"生产卡号：{self.conduct_id}",
                  fill="black", font=font_main)

        draw.text((300, 90), f"输入幅宽：{self.fukuan} mm",
                  fill="black", font=font_main)

        draw.text((520, 90), f"测量幅宽：{fukuan_min}-{fukuan_max} mm",
                  fill="black", font=font_main)

        # 检测区间显示
        if len(self.steel_length_range) == 2:
            steel_text = f"检测区间：{self.steel_length_range[0]}-{self.steel_length_range[1]} km"
        elif len(self.steel_length_range) == 1:
            steel_text = f"检测区间：{self.steel_length_range[0]}- km"
        else:
            steel_text = "检测区间：全部钢带"

        draw.text((80, 120), steel_text, fill="black", font=font_main)
        draw.text((400, 120), f"检测时间：{current_time}",
                  fill="black", font=font_main)

        # ---------------------------------------
        # 4）上表面图像与标题
        # ---------------------------------------
        up_title = "上表面缺陷分类统计"
        w, h = text_size(up_title, font_main)
        draw.text(((canvas_width - w) // 2, 160),
                  up_title, fill="black", font=font_main)

        # 粘贴上表面分布图
        up_dis_y = 190
        image.paste(up_dis, ((canvas_width - up_dis.width) // 2, up_dis_y))

        # 粘贴上表面分类条数（裁剪）
        up_cls_crop = up_cls.crop((0, 125, up_cls.width, 290))
        image.paste(up_cls_crop, (
            (canvas_width - up_cls_crop.width) // 2,
            up_dis_y + up_dis.height + 20
        ))

        # ---------------------------------------
        # 5）下表面图像与标题
        # ---------------------------------------
        bottom_start = up_dis_y + up_dis.height + up_cls_crop.height + 70
        down_title = "下表面缺陷分类统计"
        w, h = text_size(down_title, font_main)
        draw.text(((canvas_width - w) // 2, bottom_start),
                  down_title, fill="black", font=font_main)

        # 粘贴下表面分布图
        image.paste(down_dis, (
            (canvas_width - down_dis.width) // 2,
            bottom_start + 30
        ))

        # 下表面统计条图
        down_cls_crop = down_cls.crop((0, 125, down_cls.width, 320))
        image.paste(down_cls_crop, (
            (canvas_width - down_cls_crop.width) // 2,
            bottom_start + down_dis.height + 50
        ))

        # ---------------------------------------
        # 6）幅宽曲线图
        # ---------------------------------------
        fukuan_y = bottom_start + down_dis.height + down_cls_crop.height + 90
        image.paste(
            fukuan_fig,
            ((canvas_width - fukuan_fig.width) // 2, fukuan_y)
        )
        print("up_dis:", up_dis.width, up_dis.height)
        print("up_cls:", up_cls.width, up_cls.height)
        print("down_dis:", down_dis.width, down_dis.height)
        print("down_cls:", down_cls.width, down_cls.height)
        print("fukuan_fig:", fukuan_fig.width, fukuan_fig.height)

        return image


import os
import json
from datetime import datetime, timedelta

# === 修改后版本 1：增强型 Anomaly_info_List ===
class Anomaly_info_List:
    def __init__(self, filepath, multi_strip=True):
        """
        :param filepath: 检测结果的主目录路径，例如 D:\\detect result\\20251022\\product001
        :param multi_strip: 是否启用多条带钢模式，默认 False 保持单带钢兼容
        """
        self.filepath = filepath
        self.multi_strip = multi_strip

    def save_anomaly_info(self, name, data):
        import os, json
        if not name.lower().endswith(".json"):
            name = name + ".json"
        path = os.path.join(self.filepath, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_anomaly_info(self, id, listname):
        """
        兼容多种路径布局的加载函数：
        - 如果 self.filepath 本身就是 strip_x 目录，则在该目录下查找 listname。
        - 如果 self.filepath 是 CAM 目录（含多个 strip_ 子目录），则为每个 strip_ 查找并返回字典。
        - 当 multi_strip==False 时，使用原始单带钢逻辑（保持兼容）。
        返回：
          - 单条带钢：list（或 []）
          - 多条带钢：dict，键为 strip 名称，值为对应的数据列表（或空列表）
        """

        def _try_load_file(path_candidates):
            """按优先级尝试打开候选路径，返回解析后的数据或 None"""
            for p in path_candidates:
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            content = f.read()
                            content = fix_json(content)
                            if isinstance(content, str):
                                try:
                                    content = json.loads(content)
                                except:
                                    # 如果还是字符串，尝试返回原始字符串
                                    pass
                            return content
                    except Exception as e:
                        print(f"[ERROR] 读取 {p} 出错: {e}")
            return None

        # ---------- 单带钢原始逻辑（保留） ----------
        if not self.multi_strip:
            candidates = [
                os.path.join(self.filepath, str(id), listname),
                os.path.join(self.filepath, str(id), f"{listname}.json"),
                os.path.join(self.filepath, listname),
                os.path.join(self.filepath, f"{listname}.json"),
            ]
            loaded = _try_load_file(candidates)
            if loaded is None:
                return []
            return loaded

        # ---------- multi_strip=True 时的智能逻辑 ----------
        # 规范化路径
        fp = os.path.normpath(self.filepath)
        basename = os.path.basename(fp)

        # 如果 filepath 本身就是 strip_x（或以 strip_ 开头），把它当作单条带钢目录处理
        if basename.startswith("strip_"):
            candidates = [
                os.path.join(fp, str(id), listname),
                os.path.join(fp, str(id), f"{listname}.json"),
                os.path.join(fp, listname),
                os.path.join(fp, f"{listname}.json"),
            ]
            loaded = _try_load_file(candidates)
            if loaded is None:
                # 无文件时返回空列表（保持与单带钢返回类型一致）
                return []
            return loaded

        # 否则把 filepath 当作 CAM 目录，遍历其下的 strip_ 子文件夹
        try:
            subfolders = [
                d for d in os.listdir(fp)
                if d.startswith("strip_") and os.path.isdir(os.path.join(fp, d))
            ]
        except Exception as e:
            print(f"[ERROR] 列出目录 {fp} 失败: {e}")
            return {}

        if not subfolders:
            # 没有 strip_ 子目录：可能是用户传入了 CAM 目录但命名不同，尝试直接在 CAM 下读取
            candidates = [
                os.path.join(fp, str(id), listname),
                os.path.join(fp, str(id), f"{listname}.json"),
                os.path.join(fp, listname),
                os.path.join(fp, f"{listname}.json"),
            ]
            loaded = _try_load_file(candidates)
            if loaded is None:
                return {}
            # 返回一个单元素字典，键使用 basename（CAM 名）以便上层处理一致性
            return {basename: loaded}

        all_data = {}
        for sub in sorted(subfolders):
            strip_dir = os.path.join(fp, sub)
            candidates = [
                os.path.join(strip_dir, str(id), listname),
                os.path.join(strip_dir, str(id), f"{listname}.json"),
                os.path.join(strip_dir, listname),
                os.path.join(strip_dir, f"{listname}.json"),
            ]
            loaded = _try_load_file(candidates)
            if loaded is None:
                all_data[sub] = []
            else:
                all_data[sub] = loaded

        return all_data


def find_folders_with_id0(root_folder, target_id):
    matching_folders = []

    # 遍历主文件夹中的所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 确保只处理文件夹
        if os.path.isdir(folder_path):
            # 检查文件夹名是否包含给定的id
            if folder_name.endswith(f"{target_id}"):
                matching_folders.append(folder_path + "/")

    return matching_folders


# === 修改后版本 2：增强型 find_folders_with_id ===
def find_folders_with_id(base_path, product_id, multi_strip=False):
    """
    查找或创建对应的文件夹：
    1. 优先查找当天的日期文件夹。
    2. 如果当天没有，则查找前一天最后6小时的文件夹。
    3. 如果都没有，则返回空列表。
    4. 若 multi_strip=True，则在找到的文件夹下搜索 strip_1、strip_2 等子目录。

    :param base_path: 主文件夹路径
    :param product_id: 产品ID
    :param multi_strip: 是否启用多带钢模式，默认 False
    :return: list[str] — 找到或创建的文件夹路径列表
    """
    now = datetime.now()
    today_date = now.strftime('%Y%m%d')
    yesterday_date = (now - timedelta(days=1)).strftime('%Y%m%d')

    folder_path = []
    today_folder_path = os.path.join(base_path, today_date, product_id)
    yesterday_folder_path = os.path.join(base_path, yesterday_date, product_id)
    test_folder_path = os.path.join(base_path, "20251125", product_id)

    # --- 情况 1：找到当天文件夹 ---
    if os.path.exists(today_folder_path):
        print(f"找到当天的文件夹: {today_folder_path}")

        if multi_strip:
            subfolders = [
                os.path.join(today_folder_path, f)
                for f in os.listdir(today_folder_path)
                if f.startswith("strip_") and os.path.isdir(os.path.join(today_folder_path, f))
            ]
            if subfolders:
                print(f"检测到多条钢带: {subfolders}")
                return subfolders
            else:
                print(f"[提示] {today_folder_path} 下未发现 strip_ 子目录，返回主路径。")
        folder_path.append(today_folder_path)
        print(folder_path)
        return folder_path

    # --- 情况 2：时间在凌晨前6小时，查找前一天 ---
    if now.hour < 6 and os.path.exists(yesterday_folder_path):
        print(f"找到前一天的文件夹: {yesterday_folder_path}")

        if multi_strip:
            subfolders = [
                os.path.join(yesterday_folder_path, f)
                for f in os.listdir(yesterday_folder_path)
                if f.startswith("strip_") and os.path.isdir(os.path.join(yesterday_folder_path, f))
            ]
            if subfolders:
                print(f"检测到前一天的多条钢带: {subfolders}")
                return subfolders
            else:
                print(f"[提示] {yesterday_folder_path} 下未发现 strip_ 子目录，返回主路径。")
        folder_path.append(yesterday_folder_path)
        print(folder_path)
        return folder_path

    if os.path.exists(test_folder_path):
        print(f"找测试文件夹: {test_folder_path}")

        if multi_strip:
            subfolders = [
                os.path.join(test_folder_path, f)
                for f in os.listdir(test_folder_path)
                if f.startswith("strip_") and os.path.isdir(os.path.join(test_folder_path, f))
            ]
            if subfolders:
                print(f"检测到多条钢带: {subfolders}")
                return subfolders
            else:
                print(f"[提示] {test_folder_path} 下未发现 strip_ 子目录，返回主路径。")
        folder_path.append(test_folder_path)
        return folder_path

    # --- 情况 3：均未找到 ---
    print(f"[警告] 未找到 {product_id} 的检测结果文件夹。")
    return folder_path




# ---------------- GPU Sobel计算 ----------------
def _smooth_1d(x, k=51):
    k = int(k) | 1
    return cv2.GaussianBlur(x.astype(np.float32).reshape(1, -1), (k, 1), 0).ravel()


# ---------------- 定位用预处理：降维 + 去趋势（抗光照不均） ----------------
def _projection_for_split(gray_u8, use_median=True, sample_step=None):
    """
    纵向降维得到横向 1D 信号。用于 split 定位，抗噪声。
    use_median: True 用中值（抗缺陷/划痕），False 用均值。
    sample_step: 行抽样步长，None 则按高度自动（约 512 行）。
    """
    H, W = gray_u8.shape[:2]
    step = int(sample_step) if sample_step is not None else max(1, H // 512)
    block = gray_u8[0:H:step, :]
    if use_median:
        proj_v = np.median(block, axis=0).astype(np.float32)
    else:
        proj_v = block.mean(axis=0).astype(np.float32)
    return proj_v


def _detrend_1d(proj_v, sigma_px=100):
    """
    去趋势：大核高斯平滑得到基线，原向量减基线，使黑缝在信号中呈现统一量级的负峰。
    sigma_px: 基线平滑尺度（像素），越大去趋势越强。
    """
    W = proj_v.size
    k = min(max(int(6 * sigma_px) | 1, 31), max(W // 2, 51) | 1)
    base = _smooth_1d(proj_v, k=k)
    signal = proj_v - base
    return signal.astype(np.float32)


def _sobel_projection_for_split(gray_u8):
    """
    横向 Sobel 投影：黑缝和条带边缘在横向上梯度大，光照是缓慢变化。
    返回 1D: projection[x] = sum_y |grad_x(x,y)|，在投影上找峰值更稳。
    """
    grad_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    projection = np.sum(np.abs(grad_x), axis=0).astype(np.float32)
    return projection


def _build_positioning_signal(gray_u8, method="detrend", sample_step=None, detrend_sigma=100):
    """
    构建仅用于定位的 1D 信号。不要求图像质量，只要求特征极简。
    method:
      - "detrend": 纵向中值降维 + 去趋势，在 signal 上找谷值（黑缝）和导数边缘。
      - "sobel":   横向 Sobel 投影取负并平滑，兼容现有找谷值逻辑。
    """
    W = gray_u8.shape[1]
    if method == "sobel":
        proj = _sobel_projection_for_split(gray_u8)
        proj_smooth = _smooth_1d(proj, k=max(31, (W // 80) | 1))
        return (-proj_smooth).astype(np.float32)
    else:
        proj_v = _projection_for_split(gray_u8, use_median=True, sample_step=sample_step)
        signal = _detrend_1d(proj_v, sigma_px=detrend_sigma)
        return signal


# ---------------- 暗场专属：高通 Prominence 寻 N-1 条缝 + 几何外扩（粗定位） ----------------
def _split_dark_field_1d(gray_img, num_strips=3, min_strip_width_px=50, prom_thresh=2.0):
    """
    不找带钢、只找 N-1 条机械黑缝；Prominence = 大核平滑 − 小核平滑，缝为峰。
    返回半开区间列表 [(L,R), ...]，R 为 numpy 切片用右端（不包含）。
    """
    h, w = gray_img.shape[:2]
    if num_strips < 1 or w < 50 or h < 8:
        return None
    if num_strips == 1:
        return [(0, w)]

    row0, row1 = h // 4, 3 * h // 4
    if row1 <= row0:
        return None
    Y = np.median(gray_img[row0:row1, :], axis=0).astype(np.float32)
    Y_bg = cv2.GaussianBlur(Y.reshape(1, w), (51, 1), 0).flatten()
    Y_detail = cv2.GaussianBlur(Y.reshape(1, w), (11, 1), 0).flatten()
    Prom = (Y_bg - Y_detail).astype(np.float32)

    half_w = max(w // (2 * num_strips), min(30, w // 20))
    gaps = []
    P = Prom.copy()
    for _ in range(num_strips - 1):
        gi = int(np.argmax(P))
        if float(P[gi]) < prom_thresh:
            return None
        gaps.append(gi)
        lo = max(0, gi - half_w)
        hi = min(w, gi + half_w)
        P[lo:hi] = -9999.0
    gaps.sort()
    if len(gaps) != num_strips - 1:
        return None

    if len(gaps) >= 2:
        W_mid = float(np.mean(np.diff(np.asarray(gaps, dtype=np.float64))))
    else:
        W_mid = float(w) / float(num_strips)
    W_mid = max(W_mid, float(min_strip_width_px))

    margin = 10
    L_outer = max(0, int(gaps[0] - 1.15 * W_mid))
    R_right_px = min(w - 1, int(gaps[-1] + 1.15 * W_mid))
    R_outer_excl = min(w, R_right_px + 1)

    min_w = max(int(min_strip_width_px), w // 80)
    splits = []

    r0 = gaps[0] - margin
    if r0 <= L_outer or (r0 - L_outer) < min_w:
        return None
    splits.append((L_outer, r0))

    for k in range(1, num_strips - 1):
        L = gaps[k - 1] + margin
        R = gaps[k] - margin
        if R <= L or (R - L) < min_w:
            return None
        splits.append((L, R))

    L_last = gaps[-1] + margin
    R_last = max(L_last + 1, R_outer_excl)
    R_last = min(w, R_last)
    if R_last <= L_last or (R_last - L_last) < min_w:
        return None
    splits.append((L_last, R_last))

    return splits


def _bright_field_w_mid_ruler(splits_half_open, num_strips):
    """半开区间下，用中间带钢(们)像素宽度作尺子。"""
    if num_strips < 2:
        return float(splits_half_open[0][1] - splits_half_open[0][0])
    if num_strips == 2:
        w0 = splits_half_open[0][1] - splits_half_open[0][0]
        w1 = splits_half_open[1][1] - splits_half_open[1][0]
        return 0.5 * (w0 + w1)
    mid = num_strips // 2
    if num_strips % 2 == 1:
        return float(splits_half_open[mid][1] - splits_half_open[mid][0])
    w_a = splits_half_open[mid - 1][1] - splits_half_open[mid - 1][0]
    w_b = splits_half_open[mid][1] - splits_half_open[mid][0]
    return 0.5 * (w_a + w_b)


# ---------------- 明场专属：Otsu 1D + 内侧信任外推（粗定位，半开区间） ----------------
def _split_bright_field_1d(gray_img, num_strips=3, min_strip_width_px=50):
    """
    大津法 + 形态学清洗 + 跳变配对；条数过多时保留最宽 N 条再排序。
    信任内侧缝，最外两条各按中间尺子 ×1.2 外推；再整体外扩 margin=15。
    返回半开区间 [(L,R), ...]。
    """
    h, w = gray_img.shape[:2]
    if num_strips < 1 or w < 50 or h < 8:
        return None
    if num_strips == 1:
        return [(0, w)]

    row0, row1 = h // 4, 3 * h // 4
    if row1 <= row0:
        return None
    y = np.median(gray_img[row0:row1, :], axis=0).astype(np.float32)
    y_smooth = cv2.GaussianBlur(y.reshape(1, w), (11, 1), 0).flatten()
    y_norm = cv2.normalize(y_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(y_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = thresh.flatten()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    thresh = cv2.morphologyEx(thresh.reshape(1, w), cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel).flatten()

    mask = (thresh / 255).astype(np.int8)
    diff = np.diff(mask)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0] == 1:
        starts = np.insert(starts, 0, 0)
    if mask[-1] == 1:
        ends = np.append(ends, w - 1)

    min_w = max(min_strip_width_px, w // 80)
    splits_inc = []
    for s, e in zip(starts, ends):
        if e - s + 1 >= min_w:
            splits_inc.append((int(s), int(e)))

    if len(splits_inc) < num_strips:
        return None

    if len(splits_inc) > num_strips:
        splits_inc = sorted(splits_inc, key=lambda x: x[1] - x[0], reverse=True)[:num_strips]
    splits_inc = sorted(splits_inc, key=lambda x: x[0])

    # 半开区间 [L, R)
    splits = [(s, e + 1) for s, e in splits_inc]

    if num_strips >= 2:
        W_mid = _bright_field_w_mid_ruler(splits, num_strips)
        W_mid = max(W_mid, float(min_w))

        L0, R0 = splits[0]
        new_L0 = max(0, int(np.floor(R0 - W_mid * 1.20)))
        splits[0] = (new_L0, R0)

        Ln, Rn = splits[-1]
        ext = max(min_w, int(np.ceil(W_mid * 1.20)))
        new_Rn = min(w, Ln + ext)
        if new_Rn <= Ln:
            new_Rn = min(w, Ln + min_w)
        splits[-1] = (Ln, new_Rn)

    margin = 15
    out = []
    for L, R in splits:
        L2 = max(0, L - margin)
        R2 = min(w, R + margin)
        if R2 <= L2 + min_w:
            return None
        out.append((L2, R2))
    return out


def _pick_distinct_peaks(y_smooth, n_peaks, min_dist):
    """
    从 1D 曲线中选取 n 个互斥峰值锚点（峰间至少 min_dist）。
    返回升序坐标列表；失败返回 None。
    """
    w = y_smooth.shape[0]
    if w <= 0 or n_peaks <= 0:
        return None

    peaks = []
    temp_y = y_smooth.copy()
    for _ in range(n_peaks):
        p = int(np.argmax(temp_y))
        if temp_y[p] < 0:
            break
        peaks.append(p)
        start = max(0, p - min_dist)
        end = min(w, p + min_dist)
        temp_y[start:end] = -1

    if len(peaks) < n_peaks:
        return None
    peaks.sort()
    return peaks


def _robust_split_1d_two_strips(gray_img, min_strip_width_px=50):
    """
    2条带钢专用：双锚点 + 中缝谷值 + 外边界几何推断。
    """
    h, w = gray_img.shape[:2]
    if w < 50 or h < 50:
        return None

    y = np.median(gray_img[h // 4 : 3 * h // 4, :], axis=0).astype(np.float32)
    y_smooth = cv2.GaussianBlur(y.reshape(1, w), (51, 1), 0).flatten()

    peaks = _pick_distinct_peaks(y_smooth, n_peaks=2, min_dist=max(20, w // 5))
    if peaks is None:
        return None
    c1, c2 = peaks
    if c2 - c1 < max(20, w // 10):
        return None

    gap = int(np.argmin(y_smooth[c1:c2])) + c1

    # 用左右两侧腹地到中缝的距离估计两条带钢的宽度
    left_w = max(1, gap - c1)
    right_w = max(1, c2 - gap)
    ref_w = int(max(left_w, right_w) * 2.35)  # 与3条策略一致，留足边界余量
    l0 = max(0, int(gap - ref_w))
    r1 = min(w - 1, int(gap + ref_w))

    margin = 10
    splits = [
        (l0, max(l0 + 1, gap - margin)),
        (min(w - 1, gap + margin), r1),
    ]

    min_w = max(min_strip_width_px, w // 80)
    valid = []
    for l, r in splits:
        if r - l > min_w:
            valid.append((int(l), int(r)))
    if len(valid) == 2:
        return valid
    return None


def _robust_split_1d_four_strips(gray_img, min_strip_width_px=50):
    """
    4条带钢专用：四锚点 + 三中缝谷值 + 外边界几何推断。
    """
    h, w = gray_img.shape[:2]
    if w < 80 or h < 50:
        return None

    y = np.median(gray_img[h // 4 : 3 * h // 4, :], axis=0).astype(np.float32)
    y_smooth = cv2.GaussianBlur(y.reshape(1, w), (51, 1), 0).flatten()

    peaks = _pick_distinct_peaks(y_smooth, n_peaks=4, min_dist=max(20, w // 10))
    if peaks is None:
        return None
    c1, c2, c3, c4 = peaks
    if min(c2 - c1, c3 - c2, c4 - c3) < max(20, w // 20):
        return None

    g1 = int(np.argmin(y_smooth[c1:c2])) + c1
    g2 = int(np.argmin(y_smooth[c2:c3])) + c2
    g3 = int(np.argmin(y_smooth[c3:c4])) + c3

    # 用中间两条估计单条典型宽度，再向外推左右外边界
    w2 = max(1, g2 - g1)
    w3 = max(1, g3 - g2)
    ref_w = int(np.median([w2, w3]))
    l0 = max(0, int(g1 - ref_w * 1.25))
    r4 = min(w - 1, int(g3 + ref_w * 1.25))

    margin = 10
    splits = [
        (l0, max(l0 + 1, g1 - margin)),
        (min(w - 1, g1 + margin), max(g1 + margin + 1, g2 - margin)),
        (min(w - 1, g2 + margin), max(g2 + margin + 1, g3 - margin)),
        (min(w - 1, g3 + margin), r4),
    ]

    min_w = max(min_strip_width_px, w // 90)
    valid = []
    for l, r in splits:
        if r - l > min_w:
            valid.append((int(l), int(r)))
    if len(valid) == 4:
        return valid
    return None


# ---------------- 锚点+山谷算法：先找中心峰，再找区间内最低谷，梯度找外边缘 ---------------
def _robust_split_1d(gray_img, num_strips=3, median_bg_size=201, min_strip_width_px=50,
                     fukuan_list_mm=None, standard_ratio_x=None):
    """
    终极锚点算法（Anchor & Valley）：找中心峰值 -> 区间内找最低谷 -> 梯度找外边缘。
    不依赖全图阈值，先锁定三条带钢的“山顶”，再在锚点之间找黑缝“山谷”，最后用梯度找左右悬崖。
    median_bg_size 保留以兼容调用方，本实现中未使用。
    """
    if num_strips != 3:
        return None  # 专为3条带设计，其他回退

    h, w = gray_img.shape[:2]
    if w < 50 or h < 50:
        return None

    # 1. 取中间高度压扁成 1D，并做一次重度平滑（消除局部噪点）
    y = np.median(gray_img[h // 4 : 3 * h // 4, :], axis=0).astype(np.float32)
    y_smooth = cv2.GaussianBlur(y.reshape(1, w), (51, 1), 0).flatten()

    # 2. 找锚点 (Anchors)：寻找3个距离足够远的局部最高峰
    peaks = []
    temp_y = y_smooth.copy()
    min_dist = w // 6  # 假设带钢宽度至少占全图的 1/6，防止在同一条带上找两个峰

    for _ in range(3):
        p = int(np.argmax(temp_y))
        peaks.append(p)
        # 挖掉这个峰附近区域，逼迫下次循环去别的带钢上找
        start = max(0, p - min_dist)
        end = min(w, p + min_dist)
        temp_y[start:end] = -1

    peaks.sort()
    if len(peaks) < 3:
        return None
    C1, C2, C3 = peaks  # 这三个坐标必定分别在三条带钢的腹地

    # 3. 找内缝 (Gaps)：两个锚点之间的绝对最低点
    Gap1 = int(np.argmin(y_smooth[C1:C2])) + C1
    Gap2 = int(np.argmin(y_smooth[C2:C3])) + C2

    # 计算中间带钢的基准宽度（黑缝间距），作为像素域的“尺子”
    W_mid = int(max(1, Gap2 - Gap1))

    # 4) 外边界推断：
    # - 优先：用 UI 输入幅宽(毫米)与中间像素宽度做比例推断左右像素宽度
    # - 失败：回退到原方案（左右都按 W_mid * 1.25 推断）
    margin = 10
    min_w = max(min_strip_width_px, w // 80)

    def _assemble_and_validate(L1, R3):
        L1 = int(np.clip(L1, 0, w - 2))
        R3 = int(np.clip(R3, L1 + 2, w - 1))
        _splits = [
            (L1, max(L1 + 1, Gap1 - margin)),
            (min(w - 1, Gap1 + margin), max(Gap1 + margin + 1, Gap2 - margin)),
            (min(w - 1, Gap2 + margin), R3),
        ]
        valid = []
        lastR = 0
        for (L, R) in _splits:
            L = max(int(L), lastR)
            R = int(np.clip(R, L + 2, w))
            if R - L >= min_w:
                valid.append((L, R))
            lastR = R
        return valid if len(valid) == 3 else None

    # --- 方案A：比例推断（仅在配置有效时启用） ---
    use_ratio = (
        isinstance(fukuan_list_mm, (list, tuple))
        and len(fukuan_list_mm) == 3
        and standard_ratio_x is not None
        and float(standard_ratio_x) > 1e-6
        and all(float(v) > 0 for v in fukuan_list_mm)
    )

    if use_ratio:
        try:
            fw1, fw2, fw3 = [float(v) for v in fukuan_list_mm]
            r1 = fw1 / (fw2 + 1e-6)
            r3 = fw3 / (fw2 + 1e-6)

            # 用“中间像素宽度”缩放出左右像素宽度，并保留与原方案一致的安全系数
            w1_pred = max(1.0, W_mid * r1)
            w3_pred = max(1.0, W_mid * r3)

            L1_ratio = Gap1 - w1_pred * 1.25
            R3_ratio = Gap2 + w3_pred * 1.25

            candidate = _assemble_and_validate(L1_ratio, R3_ratio)
            if candidate is not None:
                return candidate
        except Exception:
            # 比例推断失败则走回退方案
            pass

    # --- 方案B：原始回退（只依赖中间宽度推左右） ---
    L1_fallback = Gap1 - W_mid * 1.25
    R3_fallback = Gap2 + W_mid * 1.25
    candidate = _assemble_and_validate(L1_fallback, R3_fallback)
    return candidate


# ---------------- bool runs ----------------
def _runs_from_binary(b):
    idx = np.flatnonzero(b)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, cuts)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]  # [L,R)

# ---------------- prominence 找黑缝 ----------------
def _find_gap_valleys_by_prominence(profile_s, topk=2, min_dist=600, margin_ratio=0.08, win=300):
    x = profile_s.astype(np.float32)
    W = x.size
    Lm = int(W * margin_ratio)
    Rm = int(W * (1 - margin_ratio))

    mins = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
    mins = mins[(mins >= Lm) & (mins <= Rm)]
    if mins.size == 0:
        return []

    candidates = []
    for idx in mins:
        l0 = max(Lm, idx - win); l1 = idx
        r0 = idx + 1;           r1 = min(Rm, idx + win)
        if l1 <= l0 or r1 <= r0:
            continue
        left_max = float(np.max(x[l0:l1]))
        right_max = float(np.max(x[r0:r1]))
        prom = min(left_max, right_max) - float(x[idx])
        candidates.append((prom, int(idx)))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)

    kept = []
    for prom, idx in candidates:
        if prom <= 0:
            continue
        if all(abs(idx - j) >= min_dist for j in kept):
            kept.append(idx)
            if len(kept) >= topk:
                break
    kept.sort()
    return kept

# ---------------- 鲁棒导数阈值 ----------------
def _robust_derivative_threshold(d_roi_abs, q=90, min_th=0.2):
    th = float(np.percentile(d_roi_abs, q))
    return max(th, float(min_th))

# ---------------- 选 run 的“结束点” ----------------
def _pick_run_end(runs, prefer="left"):
    """
    prefer='left': 取最靠左 run 的结束点
    prefer='right':取最靠右 run 的结束点
    返回的是 ROI-local 的 R（右开边界），适合直接做边界
    """
    if not runs:
        return None
    if prefer == "left":
        runs = sorted(runs, key=lambda lr: lr[0])  # L 升序
    else:
        runs = sorted(runs, key=lambda lr: lr[1], reverse=True)  # R 降序
    L, R = runs[0]
    return int(R)

# ---------------- 黑缝边界估计（导数端点） ----------------
def _gap_bounds_by_derivative(d, ad, v, W, q=92, min_th=0.2):
    win = max(120, W // 30)
    gap_pad = max(20, W // 200)

    # 左侧：下降段结束点（进入黑缝）
    l0 = max(0, v - win); l1 = max(0, v - gap_pad)
    gL = v
    if l1 - l0 > 30:
        roi_d = d[l0:l1]; roi_ad = ad[l0:l1]
        th = _robust_derivative_threshold(roi_ad, q=q, min_th=min_th)
        b = (roi_d < 0) & (roi_ad >= th)
        runs = _runs_from_binary(b)
        min_run = max(8, (l1 - l0) // 60)
        runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
        edge = _pick_run_end(runs, prefer="right")  # 靠近 v 的那段下降结束
        if edge is not None:
            gL = l0 + edge

    # 右侧：上升段结束点（离开黑缝）
    r0 = min(W, v + gap_pad); r1 = min(W, v + win)
    gR = v
    if r1 - r0 > 30:
        roi_d = d[r0:r1]; roi_ad = ad[r0:r1]
        th = _robust_derivative_threshold(roi_ad, q=q, min_th=min_th)
        b = (roi_d > 0) & (roi_ad >= th)
        runs = _runs_from_binary(b)
        min_run = max(8, (r1 - r0) // 60)
        runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
        edge = _pick_run_end(runs, prefer="left")  # 最靠左那段上升结束
        if edge is not None:
            gR = r0 + edge

    gL = int(np.clip(gL, 0, W - 2))
    gR = int(np.clip(gR, gL + 1, W))
    return gL, gR

# ---------------- 右侧黑边精修：把 R 往左收缩 ----------------
# ---------------- 增强版：双向边缘精修 ----------------
# ---------------- 增强版：基于局部梯度的暴力切分 ----------------
def _fine_trim_strip_v2(img_strip, edge_margin=150):
    """
    针对单条 strip 进行局部梯度检测，精准定位带钢亮部起点
    """
    if img_strip.ndim == 3:
        gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_strip

    H, W = gray.shape
    if W < edge_margin * 2: return 0, W

    # 计算列平均值并平滑
    profile = gray.mean(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (11, 1), 0).ravel()

    # 计算一阶导数（梯度）
    grad = np.gradient(profile)

    # --- 左边缘：在左侧 search_win 像素内寻找最大的上升梯度 ---
    # 我们找的是从“黑变白”的最剧烈位置
    search_win = edge_margin
    left_roi_grad = grad[:search_win]
    # 找到最大正梯度的位置
    new_L = np.argmax(left_roi_grad)

    # 为了切得更“干净”，我们把 L 往右再偏移 3-5 个像素，确保避开过渡带
    new_L = min(new_L + 5, search_win)

    # --- 右边缘：在右侧 search_win 像素内寻找最大的下降梯度 ---
    # 我们找的是从“白变黑”的最剧烈位置
    right_roi_grad = grad[W - search_win:]
    # 找到绝对值最大的负梯度位置
    new_R = (W - search_win) + np.argmin(right_roi_grad)

    # 同理，把 R 往左偏移几个像素
    new_R = max(new_R - 5, W - search_win)

    # 安全检查：如果切完宽度太离谱，就退回
    if new_R <= new_L + 100:
        return 0, W

    return int(new_L), int(new_R)


def _try_legacy_valley_3_splits(gray_for_position, W, H, positioning_method, detrend_sigma):
    """
    三条带时的谷值 + 一阶导数粗定位；返回半开区间（与缩略图坐标系 W 一致），失败返回 None。
    """
    profile_s = _build_positioning_signal(
        gray_for_position,
        method=positioning_method,
        sample_step=max(1, H // 512),
        detrend_sigma=min(detrend_sigma, max(W // 4, 50)),
    )
    profile_s = _smooth_1d(profile_s, k=max(31, (W // 80) | 1)).astype(np.float32)

    valleys = _find_gap_valleys_by_prominence(
        profile_s, topk=2, min_dist=max(300, W // 6),
        margin_ratio=0.06, win=max(200, W // 20)
    )
    if len(valleys) != 2:
        return None

    v1, v2 = valleys
    if v1 > v2:
        v1, v2 = v2, v1

    d = np.gradient(profile_s)
    d = _smooth_1d(d, k=max(15, (W // 200) | 1)).astype(np.float32)
    ad = np.abs(d)

    gap_pad = max(30, W // 120)

    left_roi = (0, max(0, v1 - gap_pad))
    L0 = 0
    if left_roi[1] - left_roi[0] > 80:
        a0, a1 = left_roi
        roi_d = d[a0:a1]
        roi_ad = ad[a0:a1]
        thL = _robust_derivative_threshold(roi_ad, q=90, min_th=0.15)
        b = (roi_d > 0) & (roi_ad >= thL)
        runs = _runs_from_binary(b)
        min_run = max(12, (a1 - a0) // 80)
        runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
        edge = _pick_run_end(runs, prefer="left")
        if edge is not None:
            L0 = a0 + edge

    right_roi = (min(W, v2 + gap_pad), W)
    R0 = W
    if right_roi[1] - right_roi[0] > 80:
        a0, a1 = right_roi
        roi_d = d[a0:a1]
        roi_ad = ad[a0:a1]
        thR = _robust_derivative_threshold(roi_ad, q=90, min_th=0.15)
        b = (roi_d < 0) & (roi_ad >= thR)
        runs = _runs_from_binary(b)
        min_run = max(12, (a1 - a0) // 80)
        runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]

        dyn = float(np.percentile(profile_s, 90) - np.percentile(profile_s, 10))
        min_drop = max(3.0, 0.10 * dyn)

        pickR = None
        for (L, R) in sorted(runs, key=lambda lr: lr[1], reverse=True):
            drop = float(np.sum(-roi_d[L:R]))
            if drop >= min_drop:
                pickR = R
                break

        if pickR is None:
            edge = _pick_run_end(runs, prefer="right")
            if edge is not None:
                pickR = edge

        if pickR is not None:
            R0 = a0 + int(pickR)

    g1L, g1R = _gap_bounds_by_derivative(d, ad, v1, W, q=92, min_th=0.2)
    g2L, g2R = _gap_bounds_by_derivative(d, ad, v2, W, q=92, min_th=0.2)

    splits = [
        (int(np.clip(L0, 0, W - 2)), int(np.clip(g1L, 1, W))),
        (int(np.clip(g1R, 0, W - 2)), int(np.clip(g2L, 1, W))),
        (int(np.clip(g2R, 0, W - 2)), int(np.clip(R0, 1, W))),
    ]

    min_strip_w = max(80, W // 40)
    fixed = []
    lastR = 0
    for L, R in splits:
        L = max(int(L), lastR)
        R = int(np.clip(R, L + 2, W))
        if R - L < min_strip_w:
            R = min(W, L + min_strip_w)
        fixed.append((L, R))
        lastR = R
    return fixed


def _uniform_n_splits(W, num_strips):
    """均分粗定位，半开区间。"""
    return [(int(i * W / num_strips), int((i + 1) * W / num_strips)) for i in range(num_strips)]


def _is_bright_field_cam(cam_id):
    """明场路由：cam_id==1（二号相机）或标识串中含 bright。"""
    if cam_id is None:
        return False
    if isinstance(cam_id, str):
        return "bright" in cam_id.lower()
    try:
        return int(cam_id) == 1
    except (TypeError, ValueError):
        return False


def _otsu_vertical_valid_strips(gray_img, num_strips):
    """
    中间 50% 行均值投影 + Otsu + diff，得到若干条「亮带」半开区间 (l,r)。
    过滤宽度过小的噪点段；min 宽 = w // (num_strips * 5)。
    返回 (valid_strips, roi_gray, w)；图像过小时返回 (None, None, w)。
    """
    h, w = gray_img.shape[:2]
    if w < 100 or h < 8:
        return None, None, w
    roi_top = h // 4
    roi_bottom = 3 * h // 4
    if roi_bottom <= roi_top:
        return None, None, w
    roi_gray = gray_img[roi_top:roi_bottom, :]
    v_u8 = np.mean(roi_gray, axis=0).astype(np.uint8)
    _, binary_profile = cv2.threshold(
        v_u8.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    is_strip = binary_profile.flatten() > 0
    padded_is_strip = np.concatenate([[False], is_strip, [False]])
    diff = np.diff(padded_is_strip.astype(np.int8))
    left_edges = np.where(diff == 1)[0]
    right_edges = np.where(diff == -1)[0]
    min_width_px = max(1, w // (num_strips * 5))
    valid_strips = []
    for l, r in zip(left_edges, right_edges):
        if r - l > min_width_px:
            valid_strips.append((int(l), int(r)))
    return valid_strips, roi_gray, w


def _refine_outer_edge_1d(grad, lo, hi, side, inset_px=4):
    """
    在列区间 [lo, hi) 内，用一维梯度精修最外缘（半开坐标系）。
    side='left': 找最大正梯度（暗→亮），返回带钢左缘列索引。
    side='right': 找最小负梯度（亮→暗），返回带钢右缘半开 R（不包含）。
    """
    lo = int(max(0, lo))
    hi = int(min(len(grad), hi))
    if hi - lo < 3:
        return None
    sl = grad[lo:hi]
    if side == "left":
        k = lo + int(np.argmax(sl))
        return int(min(k + inset_px, hi - 1))
    k = lo + int(np.argmin(sl))
    R = k - inset_px + 1
    return int(max(lo + 1, R))


def _split_3strip_by_gradient(gray_img, fukuan_list_mm, standard_ratio_x=1.0,
                               min_strip_width_px=80, outer_safety_px=15, joint_alpha=0.5):
    """
    三条带钢：Otsu 垂直投影定三条亮带 + 两条黑缝；中间幅宽 + 工人输入幅宽比推算左右外缘，
    再在预测窗口内用 1D 梯度精修最左/最右边界。

    1. 中间 50% 行均值投影 → Otsu → diff 得三条亮带 [l0,r0), [l1,r1), [l2,r2)。
    2. 信任中间带钢 [l1,r1) 与两缝区间 [r0,l1)、[r1,l2)（Otsu 对缝/中带对比度好）。
    3. W_mid = r1 - l1；按 fukuan 比例 W1 ≈ W_mid*(fw0/fw1)，W3 ≈ W_mid*(fw2/fw1)。
    4. 左缘预测中心 r0 - W1、右缘预测中心 l2 + W3，± 搜索窗内在平滑投影梯度上精修。
    5. 失败（条数≠3 等）仍走均分三等分兜底。outer_safety_px：精修后再向外扩若干像素给下游 fine_trim。

    standard_ratio_x / joint_alpha 仅为接口兼容，未使用。
    """
    num_strips = 3
    if len(fukuan_list_mm) != num_strips:
        return None

    fw = [float(x) for x in fukuan_list_mm]
    if fw[1] < 1e-6:
        return None

    valid_strips, roi_gray, w = _otsu_vertical_valid_strips(gray_img, num_strips)
    if roi_gray is None:
        return None

    used_ratio_refine = False
    if len(valid_strips) == num_strips:
        (l0, r0), (l1, r1), (l2, r2) = valid_strips
        W_mid = r1 - l1
        if W_mid >= min_strip_width_px and r0 <= l1 and r1 <= l2:
            ratio1 = fw[0] / fw[1]
            ratio3 = fw[2] / fw[1]
            W1 = max(float(min_strip_width_px), W_mid * ratio1)
            W3 = max(float(min_strip_width_px), W_mid * ratio3)

            Y = np.mean(roi_gray, axis=0).astype(np.float32)
            Y_s = cv2.GaussianBlur(Y.reshape(1, w), (7, 1), 0).flatten()
            grad = np.gradient(Y_s)

            win1 = max(20, int(0.30 * W1), w // 25)
            win3 = max(20, int(0.30 * W3), w // 25)

            cL = int(r0 - W1)
            L_lo = max(0, cL - win1)
            L_hi = min(r0, cL + win1)
            L_new = _refine_outer_edge_1d(grad, L_lo, L_hi, "left", inset_px=4)
            if L_new is None:
                L_new = l0
            L_new = max(0, min(L_new, r0 - 2))

            cR = int(l2 + W3)
            R_lo = max(l2, cR - win3)
            R_hi = min(w, cR + win3)
            R_new = _refine_outer_edge_1d(grad, R_lo, R_hi, "right", inset_px=4)
            if R_new is None:
                R_new = r2
            R_new = max(l2 + 2, min(R_new, w))

            L_new = max(0, L_new - outer_safety_px)
            R_new = min(w, R_new + outer_safety_px)

            out = [(L_new, r0), (l1, r1), (l2, R_new)]
            used_ratio_refine = True

    if not used_ratio_refine:
        if len(valid_strips) != num_strips:
            avg_w = w // num_strips
            valid_strips = [(i * avg_w, (i + 1) * avg_w) for i in range(num_strips)]
        out = []
        for l, r in valid_strips:
            l = max(0, int(l))
            r = min(w, int(r))
            if r <= l + 1:
                return None
            out.append((l, r))

    for L, R in out:
        if R - L < min_strip_width_px:
            return None

    return out


def _split_2strip_by_gradient(gray_img, fukuan_list_mm, standard_ratio_x=1.0,
                               min_strip_width_px=80, outer_safety_px=15, joint_alpha=0.5):
    """
    两条带钢：Otsu 垂直投影定两条亮带 + 一条黑缝；用对侧 Otsu 幅宽 × 工人幅宽比推算左右外缘，
    再在 1D 梯度窗内精修（与三带钢外缘思路一致）。

    - 左外缘：W1 = W2_otsu * (fw0/fw1)，预测 r0 - W1 附近梯度精修。
    - 右外缘：W2 = W1_otsu * (fw1/fw0)，预测 l1 + W2 附近梯度精修。
    """
    num_strips = 2
    if len(fukuan_list_mm) != num_strips:
        return None
    fw = [float(x) for x in fukuan_list_mm]
    if fw[0] < 1e-6 or fw[1] < 1e-6:
        return None

    valid_strips, roi_gray, w = _otsu_vertical_valid_strips(gray_img, num_strips)
    if roi_gray is None:
        return None

    used_ratio_refine = False
    if len(valid_strips) == num_strips:
        (l0, r0), (l1, r1) = valid_strips
        W1_otsu = r0 - l0
        W2_otsu = r1 - l1
        if (
            W1_otsu >= min_strip_width_px
            and W2_otsu >= min_strip_width_px
            and r0 <= l1
        ):
            W1 = max(float(min_strip_width_px), W2_otsu * (fw[0] / fw[1]))
            W2 = max(float(min_strip_width_px), W1_otsu * (fw[1] / fw[0]))

            Y = np.mean(roi_gray, axis=0).astype(np.float32)
            Y_s = cv2.GaussianBlur(Y.reshape(1, w), (7, 1), 0).flatten()
            grad = np.gradient(Y_s)

            win1 = max(20, int(0.30 * W1), w // 25)
            win2 = max(20, int(0.30 * W2), w // 25)

            cL = int(r0 - W1)
            L_lo = max(0, cL - win1)
            L_hi = min(r0, cL + win1)
            L_new = _refine_outer_edge_1d(grad, L_lo, L_hi, "left", inset_px=4)
            if L_new is None:
                L_new = l0
            L_new = max(0, min(L_new, r0 - 2))

            cR = int(l1 + W2)
            R_lo = max(l1, cR - win2)
            R_hi = min(w, cR + win2)
            R_new = _refine_outer_edge_1d(grad, R_lo, R_hi, "right", inset_px=4)
            if R_new is None:
                R_new = r1
            R_new = max(l1 + 2, min(R_new, w))

            L_new = max(0, L_new - outer_safety_px)
            R_new = min(w, R_new + outer_safety_px)
            out = [(L_new, r0), (l1, R_new)]
            used_ratio_refine = True

    if not used_ratio_refine:
        if len(valid_strips) != num_strips:
            avg_w = w // num_strips
            valid_strips = [(i * avg_w, (i + 1) * avg_w) for i in range(num_strips)]
        out = []
        for l, r in valid_strips:
            l = max(0, int(l))
            r = min(w, int(r))
            if r <= l + 1:
                return None
            out.append((l, r))

    for L, R in out:
        if R - L < min_strip_width_px:
            return None
    return out


def _split_4strip_by_gradient(gray_img, fukuan_list_mm, standard_ratio_x=1.0,
                               min_strip_width_px=80, outer_safety_px=15, joint_alpha=0.5):
    """
    四条带钢：Otsu 垂直投影定四条亮带 + 三条黑缝；中间两条幅宽取平均作「尺」，
    再按 (fw1+fw2)/2 与 fw0、fw3 的比例推算最左/最右外缘并梯度精修（三带中间尺思路的推广）。

    内侧 [l1,r1)、[l2,r2) 及缝界 r0,l1,r1,l2,l3 均信任 Otsu；只精修第一条左缘与第四条右缘。
    """
    num_strips = 4
    if len(fukuan_list_mm) != num_strips:
        return None
    fw = [float(x) for x in fukuan_list_mm]
    fw_mid = 0.5 * (fw[1] + fw[2])
    if fw_mid < 1e-6:
        return None

    valid_strips, roi_gray, w = _otsu_vertical_valid_strips(gray_img, num_strips)
    if roi_gray is None:
        return None

    used_ratio_refine = False
    if len(valid_strips) == num_strips:
        (l0, r0), (l1, r1), (l2, r2), (l3, r3) = valid_strips
        W2 = r1 - l1
        W3 = r2 - l2
        W_mid = 0.5 * (W2 + W3)
        if (
            W_mid >= min_strip_width_px
            and r0 <= l1
            and r1 <= l2
            and r2 <= l3
        ):
            W1 = max(float(min_strip_width_px), W_mid * (fw[0] / fw_mid))
            W4 = max(float(min_strip_width_px), W_mid * (fw[3] / fw_mid))

            Y = np.mean(roi_gray, axis=0).astype(np.float32)
            Y_s = cv2.GaussianBlur(Y.reshape(1, w), (7, 1), 0).flatten()
            grad = np.gradient(Y_s)

            win1 = max(20, int(0.30 * W1), w // 25)
            win4 = max(20, int(0.30 * W4), w // 25)

            cL = int(r0 - W1)
            L_lo = max(0, cL - win1)
            L_hi = min(r0, cL + win1)
            L_new = _refine_outer_edge_1d(grad, L_lo, L_hi, "left", inset_px=4)
            if L_new is None:
                L_new = l0
            L_new = max(0, min(L_new, r0 - 2))

            cR = int(l3 + W4)
            R_lo = max(l3, cR - win4)
            R_hi = min(w, cR + win4)
            R_new = _refine_outer_edge_1d(grad, R_lo, R_hi, "right", inset_px=4)
            if R_new is None:
                R_new = r3
            R_new = max(l3 + 2, min(R_new, w))

            L_new = max(0, L_new - outer_safety_px)
            R_new = min(w, R_new + outer_safety_px)
            out = [(L_new, r0), (l1, r1), (l2, r2), (l3, R_new)]
            used_ratio_refine = True

    if not used_ratio_refine:
        if len(valid_strips) != num_strips:
            avg_w = w // num_strips
            valid_strips = [(i * avg_w, (i + 1) * avg_w) for i in range(num_strips)]
        out = []
        for l, r in valid_strips:
            l = max(0, int(l))
            r = min(w, int(r))
            if r <= l + 1:
                return None
            out.append((l, r))

    for L, R in out:
        if R - L < min_strip_width_px:
            return None
    return out


# ---------------- 主函数 ----------------
def split_multi_strips(img, fukuan_list_mm=None, standard_ratio_x=1,
                       min_peak_dist_px=30, search_margin_ratio=0.1,
                       use_thumbnail=False, positioning_method="detrend",
                       detrend_sigma=100, morph_closing=True, cam_id=None):
    """
    条带切分。在列投影前增加“定位用”预处理，抗光照不均与背景纹理。
    cam_id: 相机 ID（0=cam1, 1=cam2 明场, 2=cam3 暗场, 3=cam4）。
    cam_id==1：明场 _split_bright_field_1d；否则：暗场 _split_dark_field_1d。
    2/3/4 条带优先：Otsu 垂直投影 + 幅宽比推算最外缘 + 1D 梯度精修（与同条数 fukuan 配置一致）。
    失败时再走双轨/三条谷值导数/均分；最后统一 _fine_trim_strip_v2。
    """
    if fukuan_list_mm is None:
        fukuan_list_mm = [400, 400, 400]
    num_strips = len(fukuan_list_mm)

    if img.ndim == 3:
        gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_u8 = np.asarray(img, dtype=np.uint8)
    H, W = gray_u8.shape[:2]

    # ---------- 缩略图路径 ----------
    # gray_raw_loc: 原始灰度（未拍平），供谷值/梯度/Otsu 等需要横向亮度对比的算法使用
    # gray_for_position: 拍平后灰度，仅供遗留的 detrend+导数算法使用
    scale_from_thumb = 1.0
    gray_raw_loc   = gray_u8   # 原始（不拍平）
    gray_for_position = gray_u8
    if use_thumbnail and W > 512:
        thumb_size = 512
        gray_thumb = cv2.resize(gray_u8, (thumb_size, thumb_size),
                                interpolation=cv2.INTER_AREA)
        gray_raw_loc      = gray_thumb                             # 原始缩略图
        gray_for_position = flatten_background_projection(gray_thumb)  # 拍平版，仅给遗留方法
        scale_from_thumb  = W / thumb_size
        H, W = gray_thumb.shape[0], gray_thumb.shape[1]

    # 形态学闭运算去细小噪点（两张图都做）
    if morph_closing:
        kernel = np.ones((3, 3), np.uint8)
        gray_raw_loc      = cv2.morphologyEx(gray_raw_loc,      cv2.MORPH_CLOSE, kernel)
        gray_for_position = cv2.morphologyEx(gray_for_position, cv2.MORPH_CLOSE, kernel)

    min_w_px = max(50, W // 80)

    splits_1d = None

    # ---------- 2/3/4 条：Otsu 垂直投影 + 外缘幅宽比 + 梯度精修（须用 gray_raw_loc） ----------
    if num_strips == 2:
        splits_1d = _split_2strip_by_gradient(
            gray_raw_loc,
            fukuan_list_mm=fukuan_list_mm,
            standard_ratio_x=standard_ratio_x,
            min_strip_width_px=min_w_px,
        )
    elif num_strips == 3:
        splits_1d = _split_3strip_by_gradient(
            gray_raw_loc,
            fukuan_list_mm=fukuan_list_mm,
            standard_ratio_x=standard_ratio_x,
            min_strip_width_px=min_w_px,
        )
    elif num_strips == 4:
        splits_1d = _split_4strip_by_gradient(
            gray_raw_loc,
            fukuan_list_mm=fukuan_list_mm,
            standard_ratio_x=standard_ratio_x,
            min_strip_width_px=min_w_px,
        )

    # ---------- 上述失败或其它条数：走双轨算法（同样用原始图） ----------
    if splits_1d is None or len(splits_1d) != num_strips:
        if _is_bright_field_cam(cam_id):
            splits_1d = _split_bright_field_1d(
                gray_raw_loc, num_strips=num_strips, min_strip_width_px=min_w_px
            )
        else:
            splits_1d = _split_dark_field_1d(
                gray_raw_loc, num_strips=num_strips, min_strip_width_px=min_w_px
            )

    # ---------- 仍然失败：三条走谷值+导数旧算法（可接受拍平图），否则均分 ----------
    if splits_1d is None or len(splits_1d) != num_strips:
        if num_strips == 3:
            splits_1d = _try_legacy_valley_3_splits(
                gray_for_position, W, H, positioning_method, detrend_sigma
            )
        if splits_1d is None or len(splits_1d) != num_strips:
            splits_1d = _uniform_n_splits(W, num_strips)

    full_W = img.shape[1]
    if scale_from_thumb != 1.0:
        splits = [(int(L * scale_from_thumb), int(R * scale_from_thumb)) for (L, R) in splits_1d]
        splits = [(max(0, L), min(full_W, R)) for (L, R) in splits]
    else:
        splits = [(int(L), int(R)) for (L, R) in splits_1d]

    min_strip_w = max(80, full_W // 40)
    fixed = []
    lastR = 0
    for L, R in splits:
        L = max(int(L), lastR)
        R = int(np.clip(R, L + 2, full_W))
        if R - L < min_strip_w:
            R = min(full_W, L + min_strip_w)
        fixed.append((L, R))
        lastR = R
    splits = fixed

    refined_strips = []
    refined_splits = []
    for (L, R) in splits:
        temp_strip = img[:, L:R]
        dL, dR = _fine_trim_strip_v2(temp_strip, edge_margin=400)
        final_L = L + dL
        final_R = L + dR
        refined_splits.append((final_L, final_R))
        refined_strips.append(img[:, final_L:final_R])

    measured_mm = [float((r - l) * standard_ratio_x) for (l, r) in refined_splits]
    return refined_strips, measured_mm, refined_splits
#
# def _smooth_1d(x, k=51):
#     k = int(k) | 1
#     return cv2.GaussianBlur(x.astype(np.float32).reshape(1, -1), (k, 1), 0).ravel()
#
# # ---------------- bool runs ----------------
# def _runs_from_binary(b):
#     idx = np.flatnonzero(b)
#     if idx.size == 0:
#         return []
#     cuts = np.where(np.diff(idx) > 1)[0] + 1
#     groups = np.split(idx, cuts)
#     return [(int(g[0]), int(g[-1]) + 1) for g in groups]  # [L,R)
#
# # ---------------- prominence 找黑缝 ----------------
# def _find_gap_valleys_by_prominence(profile_s, topk=2, min_dist=600, margin_ratio=0.08, win=300):
#     x = profile_s.astype(np.float32)
#     W = x.size
#     Lm = int(W * margin_ratio)
#     Rm = int(W * (1 - margin_ratio))
#
#     mins = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
#     mins = mins[(mins >= Lm) & (mins <= Rm)]
#     if mins.size == 0:
#         return []
#
#     candidates = []
#     for idx in mins:
#         l0 = max(Lm, idx - win); l1 = idx
#         r0 = idx + 1;           r1 = min(Rm, idx + win)
#         if l1 <= l0 or r1 <= r0:
#             continue
#         left_max = float(np.max(x[l0:l1]))
#         right_max = float(np.max(x[r0:r1]))
#         prom = min(left_max, right_max) - float(x[idx])
#         candidates.append((prom, int(idx)))
#
#     if not candidates:
#         return []
#
#     candidates.sort(key=lambda t: t[0], reverse=True)
#
#     kept = []
#     for prom, idx in candidates:
#         if prom <= 0:
#             continue
#         if all(abs(idx - j) >= min_dist for j in kept):
#             kept.append(idx)
#             if len(kept) >= topk:
#                 break
#     kept.sort()
#     return kept
#
# # ---------------- 鲁棒导数阈值 ----------------
# def _robust_derivative_threshold(d_roi_abs, q=90, min_th=0.2):
#     th = float(np.percentile(d_roi_abs, q))
#     return max(th, float(min_th))
#
# # ---------------- 选 run 的“结束点” ----------------
# def _pick_run_end(runs, prefer="left"):
#     """
#     prefer='left': 取最靠左 run 的结束点
#     prefer='right':取最靠右 run 的结束点
#     返回的是 ROI-local 的 R（右开边界），适合直接做边界
#     """
#     if not runs:
#         return None
#     if prefer == "left":
#         runs = sorted(runs, key=lambda lr: lr[0])  # L 升序
#     else:
#         runs = sorted(runs, key=lambda lr: lr[1], reverse=True)  # R 降序
#     L, R = runs[0]
#     return int(R)
#
# # ---------------- 黑缝边界估计（导数端点） ----------------
# def _gap_bounds_by_derivative(d, ad, v, W, q=92, min_th=0.2):
#     win = max(120, W // 30)
#     gap_pad = max(20, W // 200)
#
#     # 左侧：下降段结束点（进入黑缝）
#     l0 = max(0, v - win); l1 = max(0, v - gap_pad)
#     gL = v
#     if l1 - l0 > 30:
#         roi_d = d[l0:l1]; roi_ad = ad[l0:l1]
#         th = _robust_derivative_threshold(roi_ad, q=q, min_th=min_th)
#         b = (roi_d < 0) & (roi_ad >= th)
#         runs = _runs_from_binary(b)
#         min_run = max(8, (l1 - l0) // 60)
#         runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
#         edge = _pick_run_end(runs, prefer="right")  # 靠近 v 的那段下降结束
#         if edge is not None:
#             gL = l0 + edge
#
#     # 右侧：上升段结束点（离开黑缝）
#     r0 = min(W, v + gap_pad); r1 = min(W, v + win)
#     gR = v
#     if r1 - r0 > 30:
#         roi_d = d[r0:r1]; roi_ad = ad[r0:r1]
#         th = _robust_derivative_threshold(roi_ad, q=q, min_th=min_th)
#         b = (roi_d > 0) & (roi_ad >= th)
#         runs = _runs_from_binary(b)
#         min_run = max(8, (r1 - r0) // 60)
#         runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
#         edge = _pick_run_end(runs, prefer="left")  # 最靠左那段上升结束
#         if edge is not None:
#             gR = r0 + edge
#
#     gL = int(np.clip(gL, 0, W - 2))
#     gR = int(np.clip(gR, gL + 1, W))
#     return gL, gR
#
# # ---------------- 右侧黑边精修：把 R 往左收缩 ----------------
# def _trim_right_black_border(gray_strip, max_trim=160, alpha=0.25):
#     """
#     gray_strip: 2D 灰度 strip
#     返回：需要从右侧裁掉多少列 trim_r（>=0）
#     思路：在右端窗口里找“最后一个>=阈值”的位置，阈值用背景->平台相对比例。
#     """
#     H, W = gray_strip.shape
#     if W < 10:
#         return 0
#
#     m = int(min(max_trim, W // 2))
#     if m <= 5:
#         return 0
#
#     # 右端窗口的列均值
#     prof = gray_strip[:, W - m:W].mean(axis=0).astype(np.float32)
#
#     # 背景（右端窗口中较暗部分）+ 平台（整条 strip 的高灰部分）
#     bg = float(np.percentile(prof, 10))
#     col_mean_all = gray_strip.mean(axis=0).astype(np.float32)
#     fg = float(np.percentile(col_mean_all, 90))
#
#     # 阈值：越大裁得越“狠”
#     T = bg + alpha * (fg - bg)
#
#     idx = np.where(prof >= T)[0]
#     if idx.size == 0:
#         return 0
#
#     last_good = int(idx[-1])  # 右窗口内最后一个属于带钢的平台列
#     trim_r = (m - 1) - last_good
#     return max(0, int(trim_r))
#
# # ---------------- 主函数 ----------------
# def split_multi_strips(img, fukuan_list_mm=None, standard_ratio_x=1,
#                        min_peak_dist_px=30, search_margin_ratio=0.1):
#     """
#     关键改动：
#     - 外边界：左=上升段结束点；右=下降段结束点（导数 run 的 end）
#     - 新增：对每条 strip 做右侧 black-border 精修（把 R 往左收缩）
#     """
#     if fukuan_list_mm is None:
#         fukuan_list_mm = [400, 400, 400]
#     num_strips = len(fukuan_list_mm)
#
#     if img.ndim == 3:
#         gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_u8 = img
#     H, W = gray_u8.shape[:2]
#
#     # 1) 列投影（抽样行提速）
#     step = max(1, H // 512)
#     profile = gray_u8[0:H:step, :].mean(axis=0)
#
#     # 2) 平滑
#     profile_s = _smooth_1d(profile, k=max(31, (W // 80) | 1)).astype(np.float32)
#
#     # 3) 找两条黑缝
#     valleys = _find_gap_valleys_by_prominence(
#         profile_s, topk=2, min_dist=max(300, W // 6),
#         margin_ratio=0.06, win=max(200, W // 20)
#     )
#
#     if len(valleys) != 2 or num_strips != 3:
#         splits = [(int(i * W / num_strips), int((i + 1) * W / num_strips)) for i in range(num_strips)]
#         measured_mm = [float((r - l) * standard_ratio_x) for (l, r) in splits]
#         strips = [img[:, L:R] for (L, R) in splits]
#         return strips, measured_mm, splits
#
#     v1, v2 = valleys
#     if v1 > v2:
#         v1, v2 = v2, v1
#
#     # 4) 导数 + 导数平滑（抗缺陷尖刺）
#     d = np.gradient(profile_s)
#     d = _smooth_1d(d, k=max(15, (W // 200) | 1)).astype(np.float32)
#     ad = np.abs(d)
#
#     gap_pad = max(30, W // 120)
#
#     # ---------- 左外边界：上升段结束点 ----------
#     left_roi = (0, max(0, v1 - gap_pad))
#     L0 = 0
#     if left_roi[1] - left_roi[0] > 80:
#         a0, a1 = left_roi
#         roi_d = d[a0:a1]
#         roi_ad = ad[a0:a1]
#         thL = _robust_derivative_threshold(roi_ad, q=90, min_th=0.15)
#         b = (roi_d > 0) & (roi_ad >= thL)
#         runs = _runs_from_binary(b)
#         min_run = max(12, (a1 - a0) // 80)
#         runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
#         edge = _pick_run_end(runs, prefer="left")  # 上升段结束点
#         if edge is not None:
#             L0 = a0 + edge
#
#     # ---------- 右外边界：下降段结束点 ----------
#     right_roi = (min(W, v2 + gap_pad), W)
#     R0 = W
#     if right_roi[1] - right_roi[0] > 80:
#         a0, a1 = right_roi
#         roi_d = d[a0:a1]
#         roi_ad = ad[a0:a1]
#         thR = _robust_derivative_threshold(roi_ad, q=90, min_th=0.15)
#         b = (roi_d < 0) & (roi_ad >= thR)
#         runs = _runs_from_binary(b)
#         min_run = max(12, (a1 - a0) // 80)
#         runs = [(L, R) for (L, R) in runs if (R - L) >= min_run]
#
#         # 从右往左挑“真正下坡”（累计下降量过滤）
#         dyn = float(np.percentile(profile_s, 90) - np.percentile(profile_s, 10))
#         min_drop = max(3.0, 0.10 * dyn)
#
#         pickR = None
#         for (L, R) in sorted(runs, key=lambda lr: lr[1], reverse=True):
#             drop = float(np.sum(-roi_d[L:R]))
#             if drop >= min_drop:
#                 pickR = R  # 下降段结束点
#                 break
#
#         if pickR is None:
#             edge = _pick_run_end(runs, prefer="right")
#             if edge is not None:
#                 pickR = edge
#
#         if pickR is not None:
#             R0 = a0 + int(pickR)
#
#     # ---------- 黑缝边界 ----------
#     g1L, g1R = _gap_bounds_by_derivative(d, ad, v1, W, q=92, min_th=0.2)
#     g2L, g2R = _gap_bounds_by_derivative(d, ad, v2, W, q=92, min_th=0.2)
#
#     # ---------- 初始 splits ----------
#     splits = [
#         (int(np.clip(L0, 0, W - 2)), int(np.clip(g1L, 1, W))),
#         (int(np.clip(g1R, 0, W - 2)), int(np.clip(g2L, 1, W))),
#         (int(np.clip(g2R, 0, W - 2)), int(np.clip(R0, 1, W))),
#     ]
#
#     # ---------- 单调修正 + 最小宽度 ----------
#     min_strip_w = max(80, W // 40)
#     fixed = []
#     lastR = 0
#     for L, R in splits:
#         L = max(int(L), lastR)
#         R = int(np.clip(R, L + 2, W))
#         if R - L < min_strip_w:
#             R = min(W, L + min_strip_w)
#         fixed.append((L, R))
#         lastR = R
#     splits = fixed
#
#     # ---------- ✅ 右侧黑边精修（关键新增） ----------
#     refined = []
#     for (L, R) in splits:
#         # 用灰度图做精修判断，避免颜色影响
#         strip_gray = gray_u8[:, L:R]
#         trim_r = _trim_right_black_border(strip_gray, max_trim=160, alpha=0.25)
#         R2 = max(L + 2, R - trim_r)  # 防止裁成空
#         refined.append((L, R2))
#     splits = refined
#
#     # ---------- 输出 ----------
#     measured_mm = [float((r - l) * standard_ratio_x) for (l, r) in splits]
#     strips = [img[:, L:R] for (L, R) in splits]
#     return strips, measured_mm, splits








# 简易空上下文管理器，方便写法
class dummy_context:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

def dummy_lock():
    class L:
        def __enter__(self): return None
        def __exit__(self, a,b,c): return False
    return L()
