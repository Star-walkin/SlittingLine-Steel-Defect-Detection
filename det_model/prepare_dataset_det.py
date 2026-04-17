"""
det_model/prepare_dataset_det.py
将原始图像处理成与「离线检测」完全一致的预处理，生成可直接用于 SimpleAD 训练的数据集。

与离线检测对齐的预处理流程（与 det_model/infer.py SlidingWindowDetector.detect_ano 一致）：
  1. 灰度
  2. 左右各裁 bias 像素（默认 15，与推理一致）
  3. FFT 去纹（apply_fft_deripple）
  4. 纵向滤波（apply_vertical_filter）：去除轧制纹/列 PRNU
  5. 背景拍平：中值滤波 + 减背景 + 背景均值补偿

数据集步骤沿用 seg_model_train/data_prepare.py：
  - 使用 function_bank.split_multi_strips 切带
  - 按条带竖直分段，输出到 CAMx/train/good/

重要：必须先对「整条带钢」做全局预处理（裁边、FFT、拍平），再竖直切块保存。
若先切块再对每块做 FFT/中值滤波，会在切缝处产生吉布斯振铃和边界伪影。

使用方式：
  python det_model/prepare_dataset_det.py --raw_root <原始图根目录> --exp_name <实验名> [--out_root <输出根目录>]
训练时指定相同 data_root + exp_name，并加 --preprocessed，即与推理一致。
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import cv2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from function_bank import split_multi_strips as fb_split_multi_strips
from det_model.infer import apply_fft_deripple, apply_vertical_filter


def flatten_background_subtraction(gray: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """与 infer.py 中 flatten_bg 块一致：中值滤波 + 减背景 + 均值补偿"""
    h, w = gray.shape[:2]
    if kernel_size is None:
        kernel_size = max(51, min(w, h) // 8) | 1
    background_u8 = cv2.medianBlur(gray, kernel_size)
    img_float = gray.astype(np.float32)
    bg_float = background_u8.astype(np.float32)
    mean_val = float(np.mean(bg_float))
    flattened = img_float - bg_float + mean_val
    return np.clip(flattened, 0, 255).astype(np.uint8)


def preprocess_like_inference(gray: np.ndarray, bias: int = 15) -> np.ndarray:
    """
    与 SlidingWindowDetector.detect_ano 中对 crop 的处理一致：
    灰度 -> 左右裁 bias -> FFT 去纹 -> 纵向滤波去竖纹 -> 2D 背景拍平。
    """
    H, W = gray.shape[:2]
    l = max(bias, 0)
    r = min(W - bias, W)
    if r <= l:
        l, r = 0, W
    crop = gray[:, l:r].copy()
    try:
        crop = apply_fft_deripple(crop)
    except Exception:
        pass
    crop = apply_vertical_filter(crop)
    crop = flatten_background_subtraction(crop)
    return crop


def list_images(folder: str, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp")):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    return files


def main():
    p = argparse.ArgumentParser(description="生成与离线检测预处理一致的 SimpleAD 训练数据集")
    p.add_argument("--raw_root", type=str, default=r"D:\pycharm_project\steeldefect\img_raw_0228")
    p.add_argument("--out_root", type=str, default=r"D:\pycharm_project\steeldefect\image_all")
    p.add_argument("--exp_name", type=str, default="image_data_02_28")
    p.add_argument("--cam_names", type=str, nargs="+",
                   default=["cam1_filter", "cam2_filter", "cam3_filter", "cam4_filter"])
    p.add_argument("--cam_output", type=str, nargs="+", default=["CAM1", "CAM2", "CAM3", "CAM4"])
    p.add_argument("--target_per_cam", type=int, default=10, help="每相机最多抽取原图张数，0=全部")
    p.add_argument("--max_strips", type=int, default=3)
    p.add_argument("--split_vertical_parts", type=int, default=3)
    p.add_argument("--valid_h", type=int, default=4096)
    p.add_argument("--valid_w", type=int, default=4096)
    p.add_argument("--bias", type=int, default=15, help="左右裁边像素，与推理一致")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if len(args.cam_names) != len(args.cam_output):
        raise ValueError("cam_names 与 cam_output 数量须一致")

    random.seed(args.seed)
    np.random.seed(args.seed)

    for cam_folder, cam_out in zip(args.cam_names, args.cam_output):
        in_dir = os.path.join(args.raw_root, cam_folder)
        if not os.path.isdir(in_dir):
            print(f"跳过 {cam_out}: 目录不存在 {in_dir}")
            continue

        img_files = list_images(in_dir)
        if not img_files:
            print(f"跳过 {cam_out}: 无图片 {in_dir}")
            continue

        k = min(args.target_per_cam, len(img_files)) if args.target_per_cam > 0 else len(img_files)
        selected = random.sample(img_files, k)
        out_dir = os.path.join(args.out_root, args.exp_name, cam_out, "train", "good")
        os.makedirs(out_dir, exist_ok=True)

        count_written = 0
        for img_path in tqdm(selected, desc=f"{cam_out}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            if img.shape[0] != args.valid_h or img.shape[1] != args.valid_w:
                continue

            strips, widths, split_ranges = fb_split_multi_strips(img)
            if not strips:
                continue

            strips = strips[: args.max_strips]
            orig_stem = os.path.splitext(os.path.basename(img_path))[0]
            n_parts = args.split_vertical_parts

            for s_idx, strip in enumerate(strips):
                if strip is None:
                    continue
                H, W = strip.shape[:2]
                if H < 10:
                    continue

                # 1. 获取整条带钢灰度
                gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip.copy()
                L_strip, R_strip = split_ranges[s_idx][0], split_ranges[s_idx][1]

                # 2. 【先】对整条带钢做全局预处理（裁边、FFT、拍平），保证 FFT 频率连续、中值滤波无切缝伪影
                preprocessed_strip = preprocess_like_inference(gray_strip, bias=args.bias)
                prep_H = preprocessed_strip.shape[0]

                # 3. 【后】再竖直切块保存
                if n_parts <= 1:
                    save_name = f"{orig_stem}_strip{s_idx+1}_x{L_strip}-{R_strip}_part0.png"
                    if cv2.imwrite(os.path.join(out_dir, save_name), preprocessed_strip):
                        count_written += 1
                else:
                    h_step = prep_H // n_parts
                    for part_id in range(n_parts):
                        y0 = part_id * h_step
                        y1 = (part_id + 1) * h_step if part_id < n_parts - 1 else prep_H
                        part_final = preprocessed_strip[y0:y1, :]
                        save_name = f"{orig_stem}_strip{s_idx+1}_x{L_strip}-{R_strip}_y{y0}-{y1}_part{part_id}.png"
                        if cv2.imwrite(os.path.join(out_dir, save_name), part_final):
                            count_written += 1

        print(f"  {cam_out}: 写出 {count_written} 张 -> {out_dir}")

    print("\n完成。训练时请使用:")
    print(f"  python det_model/train.py --data_root {args.out_root} --exp_name {args.exp_name} --preprocessed")


if __name__ == "__main__":
    main()
