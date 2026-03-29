import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path

# ========== 配置区 ==========
SRC_ROOT = Path(r"D:\\")  # 源目录
# 输出的主目录 (脚本会在这个目录下自动创建 cam1_origin, cam1_filter 等子文件夹)
DST_ROOT = Path(r"img_raw_0228")

RECV_DIRS = [

    "recv_0130_01",
]

CAMS = ["cam1", "cam2", "cam3", "cam4"]
SAMPLE_PER_RECV_PER_CAM = 70

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 20260227

# --- 文件夹命名后缀配置 ---
# 结果将生成: cam1_origin (原图) 和 cam1_filter (处理图)
SUFFIX_ORIGIN = "_origin"
SUFFIX_FILTER = "_filter"

# --- 滤波参数配置 ---
MASK_WIDTH = 10  # 切除的垂直条带宽度
CENTER_PROTECT_SIZE = 15  # 中心低频保护区大小


# ===========================


def list_images(folder: Path):
    """列出文件夹下的图片"""
    if not folder.exists():
        return []
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def next_index_in_dst(dst_origin_dir: Path, cam_name: str) -> int:
    """
    只需扫描 _origin 文件夹来确定下一个编号
    因为 origin 和 filter 是成对生成的，编号应该同步
    """
    max_idx = -1
    prefix = cam_name + "_"
    if not dst_origin_dir.exists():
        return 0

    for p in dst_origin_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if stem.startswith(prefix):
            tail = stem[len(prefix):]
            if tail.isdigit():
                max_idx = max(max_idx, int(tail))
    return max_idx + 1


def process_and_save_image(src_path: Path, dst_path: Path):
    """
    读取源图片 -> FFT滤波去波纹 -> 保存到目标路径
    """
    # 1. 读取图像 (以灰度模式)
    img = cv2.imread(str(src_path), 0)

    if img is None:
        print(f"[警告] 无法读取图片用于滤波: {src_path}")
        return False

    rows, cols = img.shape

    # 2. FFT变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 3. 构建掩模 (Mask)
    mask = np.ones((rows, cols, 2), np.uint8)
    center_row, center_col = rows // 2, cols // 2

    # A. 切除垂直条带
    mask[:, center_col - MASK_WIDTH: center_col + MASK_WIDTH] = 0

    # B. 保护中心直流分量
    mask[center_row - CENTER_PROTECT_SIZE: center_row + CENTER_PROTECT_SIZE,
    center_col - CENTER_PROTECT_SIZE: center_col + CENTER_PROTECT_SIZE] = 1

    # 4. 应用掩模并逆变换
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 5. 归一化并转回 uint8
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # 6. 保存图片
    try:
        success = cv2.imwrite(str(dst_path), img_back)
        return success
    except Exception as e:
        print(f"[错误] 保存图片失败: {e}")
        return False


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 1. 创建 8 个目标文件夹 (4个相机 x 2种状态)
    for cam in CAMS:
        dir_origin = DST_ROOT / (cam + SUFFIX_ORIGIN)
        dir_filter = DST_ROOT / (cam + SUFFIX_FILTER)
        dir_origin.mkdir(parents=True, exist_ok=True)
        dir_filter.mkdir(parents=True, exist_ok=True)

    # 2. 获取初始编号 (只需要查 origin 文件夹)
    counters = {}
    for cam in CAMS:
        dir_origin = DST_ROOT / (cam + SUFFIX_ORIGIN)
        counters[cam] = next_index_in_dst(dir_origin, cam)

    total_pairs = {cam: 0 for cam in CAMS}

    for recv_name in RECV_DIRS:
        recv_path = SRC_ROOT / recv_name
        if not recv_path.exists():
            print(f"[跳过] 源目录不存在: {recv_path}")
            continue

        print(f"\n=== 处理 {recv_path} ===")
        for cam in CAMS:
            src_cam_dir = recv_path / cam

            # 定义两个目标文件夹路径
            dst_dir_origin = DST_ROOT / (cam + SUFFIX_ORIGIN)
            dst_dir_filter = DST_ROOT / (cam + SUFFIX_FILTER)

            imgs = list_images(src_cam_dir)

            if not imgs:
                continue

            k = min(SAMPLE_PER_RECV_PER_CAM, len(imgs))
            picked = random.sample(imgs, k)

            success_count = 0
            for p in picked:
                idx = counters[cam]
                # 统一文件名，保证两个文件夹里的文件名完全一致
                filename = f"{cam}_{idx:06d}{p.suffix.lower()}"

                path_origin = dst_dir_origin / filename
                path_filter = dst_dir_filter / filename

                # --- 步骤 A: 复制原图 ---
                try:
                    shutil.copy2(p, path_origin)
                except Exception as e:
                    print(f"  [失败] 复制原图出错: {p.name} -> {e}")
                    continue

                # --- 步骤 B: 生成滤波图 ---
                if process_and_save_image(p, path_filter):
                    # 只有当两步都成功时，才增加计数器
                    counters[cam] += 1
                    total_pairs[cam] += 1
                    success_count += 1
                else:
                    print(f"  [失败] 滤波处理出错: {p.name}")
                    # 如果滤波失败，把刚才复制过去的“原图”也删掉，确保一一对应
                    if path_origin.exists():
                        os.remove(path_origin)

            print(f"  {cam}: 选中 {k} 张，成功生成 {success_count} 对对比图")

    print("\n===== 完成统计 =====")
    print(f"结果保存在: {DST_ROOT}")
    for cam in CAMS:
        print(f"{cam}: 共生成 {total_pairs[cam]} 对图片")
        print(f"   |-- 原图: .../{cam}{SUFFIX_ORIGIN}")
        print(f"   |-- 滤波: .../{cam}{SUFFIX_FILTER}")


if __name__ == "__main__":
    main()