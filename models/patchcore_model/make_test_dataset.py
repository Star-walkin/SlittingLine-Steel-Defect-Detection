"""
根据 detectoutline02 / det_model 的预处理流程，为 PatchCore 生成推理用测试数据集。

思路：
- 输入：bad_test/cam*_bad 下的原始 4096x4096 条带整幅图
- 预处理步骤与 SimpleAD 滑窗检测保持一致：
  1) split_multi_strips 切多条带
  2) 每条带灰度；调用 det_model.prepare_dataset_det.preprocess_like_inference：
     - 左右裁边 bias 像素
     - FFT 去纹 (apply_fft_deripple)
     - 纵向滤波 (apply_vertical_filter)
     - 背景拍平 (median - bg + mean)
  3) 再纵向等分成若干块保存，用于 PatchCore 的小图推理

输出目录结构（与 det_model 的 train 数据类似但用于 test）：
  <out_root>/<exp_name>/<CAMx>/test/images/*.png
"""

import argparse
import glob
import os
import random
from typing import List

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from function_bank import split_multi_strips as fb_split_multi_strips
from det_model.prepare_dataset_det import preprocess_like_inference, list_images


def build_cam_image_list(raw_root: str, cam_names: List[str], target_per_cam: int) -> dict:
    cam2files = {}
    for cam_name in cam_names:
        in_dir = os.path.join(raw_root, cam_name)
        if not os.path.isdir(in_dir):
            print(f"[WARN] 跳过 {cam_name}: 目录不存在 {in_dir}")
            continue
        files = list_images(in_dir)
        if not files:
            print(f"[WARN] 跳过 {cam_name}: 无图片 {in_dir}")
            continue
        if target_per_cam > 0:
            k = min(target_per_cam, len(files))
            files = random.sample(files, k)
        cam2files[cam_name] = files
        print(f"[INFO] {cam_name}: 选取 {len(files)} 张")
    return cam2files


def generate_test_patches(
    raw_root: str,
    out_root: str,
    exp_name: str,
    cam_names,
    cam_output,
    target_per_cam: int,
    max_strips: int,
    split_vertical_parts: int,
    valid_h: int,
    valid_w: int,
    bias: int,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    cam2files = build_cam_image_list(raw_root, cam_names, target_per_cam)

    for cam_folder, cam_out in zip(cam_names, cam_output):
        img_files = cam2files.get(cam_folder, [])
        if not img_files:
            continue

        out_dir = os.path.join(out_root, exp_name, cam_out, "test", "images")
        os.makedirs(out_dir, exist_ok=True)

        count_written = 0
        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            if img.shape[0] != valid_h or img.shape[1] != valid_w:
                # 仅处理与线上一致尺寸的图，避免预处理参数不匹配
                continue

            strips, widths, split_ranges = fb_split_multi_strips(img)
            if not strips:
                continue

            strips = strips[:max_strips]
            orig_stem = os.path.splitext(os.path.basename(img_path))[0]
            n_parts = split_vertical_parts

            for s_idx, strip in enumerate(strips):
                if strip is None:
                    continue
                H, W = strip.shape[:2]
                if H < 10:
                    continue

                gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip.copy()
                L_strip, R_strip = split_ranges[s_idx][0], split_ranges[s_idx][1]

                # 与离线检测一致的预处理（FFT + 纵向滤波 + 中值拍平）
                preprocessed_strip = preprocess_like_inference(gray_strip, bias=bias)
                prep_H = preprocessed_strip.shape[0]

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
                        save_name = (
                            f"{orig_stem}_strip{s_idx+1}_x{L_strip}-{R_strip}_y{y0}-{y1}_part{part_id}.png"
                        )
                        if cv2.imwrite(os.path.join(out_dir, save_name), part_final):
                            count_written += 1

        print(f"[TEST] {cam_out}: 写出 {count_written} 张 -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="为 PatchCore 生成基于 detectoutline02 预处理的测试数据集")
    p.add_argument("--raw_root", type=str, default=r"D:\pycharm_project\steeldefect\bad_test")
    p.add_argument(
        "--cam_names",
        type=str,
        nargs="+",
        default=["cam1_bad", "cam2_bad", "cam3_bad", "cam4_bad"],
        help="原始 bad 图根目录下的相机子文件夹名",
    )
    p.add_argument(
        "--cam_output",
        type=str,
        nargs="+",
        default=["CAM1", "CAM2", "CAM3", "CAM4"],
        help="输出目录中使用的相机名字",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default=r"D:\pycharm_project\steeldefect\patchcore_model\test_data",
    )
    p.add_argument("--exp_name", type=str, default="patchcore_test")
    p.add_argument("--target_per_cam", type=int, default=0, help="每相机最多使用多少原图，0 表示全部")
    p.add_argument("--max_strips", type=int, default=3)
    p.add_argument("--split_vertical_parts", type=int, default=3)
    p.add_argument("--valid_h", type=int, default=4096)
    p.add_argument("--valid_w", type=int, default=4096)
    p.add_argument("--bias", type=int, default=15, help="左右裁边像素，与 SlidingWindowDetector 一致")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.cam_names) != len(args.cam_output):
        raise ValueError("cam_names 与 cam_output 数量须一致")
    generate_test_patches(
        raw_root=args.raw_root,
        out_root=args.out_root,
        exp_name=args.exp_name,
        cam_names=args.cam_names,
        cam_output=args.cam_output,
        target_per_cam=args.target_per_cam,
        max_strips=args.max_strips,
        split_vertical_parts=args.split_vertical_parts,
        valid_h=args.valid_h,
        valid_w=args.valid_w,
        bias=args.bias,
        seed=args.seed,
    )
    print("\n完成。用于 PatchCore 推理测试的目录示例：")
    print(f"  {os.path.join(args.out_root, args.exp_name, 'CAM1', 'test', 'images')}")


if __name__ == "__main__":
    main()

