import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

def generate_unet_dataset(input_dir, output_dir, target_size=256, num_segments=3):
    """
    从切割后的钢带图像生成U-Net训练数据集
    1. 沿纵向(高度)等分切割为 num_segments 段
    2. 每段保持原始宽度，裁剪成近似正方形
    3. 直接resize成标准正方形(target_size x target_size)
    4. 保存到输出目录
    """
    os.makedirs(os.path.join(output_dir, "train", "good"), exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) +
                         glob.glob(os.path.join(input_dir, "*.png")))

    print(f"[INFO] Found {len(image_paths)} steel strip images to process...")

    counter = 0
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read {img_path}, skip.")
            continue

        h, w, _ = img.shape
        segment_height = h // num_segments  # 每段高度

        for i in range(num_segments):
            y_start = i * segment_height
            y_end = h if i == num_segments - 1 else (y_start + segment_height)
            patch = img[y_start:y_end, :]  # 保留整条宽度

            patch_resized = cv2.resize(
                patch,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR
            )

            save_path = os.path.join(output_dir, "train", "good", f"strip_{counter:05d}.png")
            cv2.imwrite(save_path, patch_resized)
            counter += 1

    print(f"[INFO] Dataset generation complete! Total patches: {counter}")


if __name__ == "__main__":
    # 你需要修改以下路径：
    input_dir = r"D:\detect result\2025-09-11\0000\cam01\1\defect_images"
    output_dir = r"D:\steel_dataset\unet_dataset"
    generate_unet_dataset(input_dir, output_dir, target_size=256, num_segments=3)
