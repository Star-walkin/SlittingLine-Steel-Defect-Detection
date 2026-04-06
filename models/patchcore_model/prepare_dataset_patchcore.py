"""
为 PatchCore 生成训练数据集，确保：

- 训练输入图像的预处理流程
- 与 bad_test 图像在推理前的预处理流程

完全一致（同一套「切带 + FFT 去纹 + 纵向滤波 + 中值拍平 + 竖直切块」）。

数据来源与规则：
- 上游从 recv_* 目录中筛选样本并做一次 FFT 去纹，
  输出到: D:/pycharm_project/steeldefect/img_raw_0228/cam*_origin, cam*_filter
- 本脚本默认读取 cam*_filter 目录（已经过 img_select 的筛选与一次 FFT 去纹）
- 然后对整幅 4096x4096 图做：
    1. split_multi_strips 切多条带
    2. preprocess_like_inference (灰度 -> 裁边 -> FFT 去纹 -> 纵向滤波 -> 中值拍平)
    3. 竖直等分成若干 patch 保存

输出目录结构（兼容 patchcore_model/train.py 所需接口）：
  <out_root>/<exp_name>/<CAMx>/train/good/*.png

PatchCore 训练脚本默认读取：
  data_root=<out_root>, exp_name=<exp_name>, CAM1~4/train/good
"""

import argparse
import os
import random
from typing import List

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from function_bank import split_multi_strips as fb_split_multi_strips  # noqa: E402
from det_model.prepare_dataset_det import preprocess_like_inference, list_images  # noqa: E402


def build_cam_image_list(raw_root: str, cam_names: List[str], target_per_cam: int, seed: int) -> dict:
    """从 img_select 生成的 cam*_filter 中抽样图像列表。"""
    random.seed(seed)
    np.random.seed(seed)

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


def generate_patchcore_train_dataset(
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
    """
    与 patchcore_model/make_test_dataset.py 使用相同的预处理逻辑，
    但输出到 CAMx/train/good，供 PatchCore 训练使用。
    """
    cam2files = build_cam_image_list(raw_root, cam_names, target_per_cam, seed)

    for cam_folder, cam_out in zip(cam_names, cam_output):
        img_files = cam2files.get(cam_folder, [])
        if not img_files:
            continue

        out_dir = os.path.join(out_root, exp_name, cam_out, "train", "good")
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

        print(f"[TRAIN] {cam_out}: 写出 {count_written} 张 -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="为 PatchCore 生成与 bad_test 推理一致预处理的训练数据集")
    # 上游 img_select 输出目录：其中包含 cam1_origin/cam1_filter 等
    p.add_argument(
        "--raw_root",
        type=str,
        default=r"D:\pycharm_project\steeldefect\img_raw_0228",
        help="img_select.py 的 DST_ROOT，默认使用 cam*_filter 作为输入",
    )
    p.add_argument(
        "--cam_names",
        type=str,
        nargs="+",
        default=["cam1_filter", "cam2_filter", "cam3_filter", "cam4_filter"],
        help="raw_root 下的相机子目录名（已做过一次 FFT 去纹）",
    )
    p.add_argument(
        "--cam_output",
        type=str,
        nargs="+",
        default=["CAM1", "CAM2", "CAM3", "CAM4"],
        help="输出目录中使用的相机名字（须与 PatchCore 的 CAMx 一致）",
    )
    # 输出给 PatchCore 训练使用：结构 <out_root>/<exp_name>/<CAMx>/train/good
    p.add_argument(
        "--out_root",
        type=str,
        default=r"D:\pycharm_project\steeldefect\image_all",
        help="PatchCore 训练数据根目录，对应 train.py 的 data_root",
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default="image_data_patchcore_0228",
        help="实验名，对应 PatchCore train.py 的 exp_name",
    )
    p.add_argument(
        "--target_per_cam",
        type=int,
        default=0,
        help="每相机最多使用多少原图 (0 = 全部)",
    )
    p.add_argument("--max_strips", type=int, default=3)
    p.add_argument("--split_vertical_parts", type=int, default=3)
    p.add_argument("--valid_h", type=int, default=4096)
    p.add_argument("--valid_w", type=int, default=4096)
    p.add_argument(
        "--bias",
        type=int,
        default=15,
        help="左右裁边像素，与 SlidingWindowDetector / preprocess_like_inference 一致",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.cam_names) != len(args.cam_output):
        raise ValueError("cam_names 与 cam_output 数量须一致")

    generate_patchcore_train_dataset(
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

    print("\n完成。PatchCore 训练时请使用：")
    print(f"  data_root = {args.out_root}")
    print(f"  exp_name  = {args.exp_name}")
    print("例如：")
    print("  python patchcore_model/train.py --data_root {0} --exp_name {1}".format(args.out_root, args.exp_name))


if __name__ == "__main__":
    main()

