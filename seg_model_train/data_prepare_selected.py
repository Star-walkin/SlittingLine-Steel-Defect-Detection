import os
import glob
import cv2
import random
from tqdm import tqdm
import sys
import numpy as np
import itertools
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from function_bank import split_multi_strips
import time

# ----------------- 配置 -----------------
RAW_ROOT = r"img_raw_26A"
OUT_ROOT = r"image_all\image_data_01_26B"

CAM_NAMES = ["cam1", "cam2", "cam3", "cam4"]
CAM_OUTPUT = ["CAM1", "CAM2", "CAM3", "CAM4"]

TARGET_PER_CAM = 100          # 每个相机抽多少张原图)
MAX_STRIPS = 3               # 每张图最多按 3 条钢带输出
SPLIT_VERTICAL_PARTS = 3      # 每条钢带沿高度竖直等分几段

VALID_H = 4096
VALID_W = 4096

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")  # 更全一点
import numpy as np
import cv2

# ---------------- 1D 平滑 ----------------
def _smooth_1d(x, k=51):
    k = int(k) | 1
    return cv2.GaussianBlur(x.astype(np.float32).reshape(1, -1), (k, 1), 0).ravel()

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
def _trim_right_black_border(gray_strip, max_trim=160, alpha=0.25):
    """
    gray_strip: 2D 灰度 strip
    返回：需要从右侧裁掉多少列 trim_r（>=0）
    思路：在右端窗口里找“最后一个>=阈值”的位置，阈值用背景->平台相对比例。
    """
    H, W = gray_strip.shape
    if W < 10:
        return 0

    m = int(min(max_trim, W // 2))
    if m <= 5:
        return 0

    # 右端窗口的列均值
    prof = gray_strip[:, W - m:W].mean(axis=0).astype(np.float32)

    # 背景（右端窗口中较暗部分）+ 平台（整条 strip 的高灰部分）
    bg = float(np.percentile(prof, 10))
    col_mean_all = gray_strip.mean(axis=0).astype(np.float32)
    fg = float(np.percentile(col_mean_all, 90))

    # 阈值：越大裁得越“狠”
    T = bg + alpha * (fg - bg)

    idx = np.where(prof >= T)[0]
    if idx.size == 0:
        return 0

    last_good = int(idx[-1])  # 右窗口内最后一个属于带钢的平台列
    trim_r = (m - 1) - last_good
    return max(0, int(trim_r))

# ---------------- 主函数 ----------------
def split_multi_strips(img, fukuan_list_mm=None, standard_ratio_x=1,
                       min_peak_dist_px=30, search_margin_ratio=0.1):
    """
    关键改动：
    - 外边界：左=上升段结束点；右=下降段结束点（导数 run 的 end）
    - 新增：对每条 strip 做右侧 black-border 精修（把 R 往左收缩）
    """
    if fukuan_list_mm is None:
        fukuan_list_mm = [400, 400, 400]
    num_strips = len(fukuan_list_mm)

    if img.ndim == 3:
        gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_u8 = img
    H, W = gray_u8.shape[:2]

    # 1) 列投影（抽样行提速）
    step = max(1, H // 512)
    profile = gray_u8[0:H:step, :].mean(axis=0)

    # 2) 平滑
    profile_s = _smooth_1d(profile, k=max(31, (W // 80) | 1)).astype(np.float32)

    # 3) 找两条黑缝
    valleys = _find_gap_valleys_by_prominence(
        profile_s, topk=2, min_dist=max(300, W // 6),
        margin_ratio=0.06, win=max(200, W // 20)
    )

    if len(valleys) != 2 or num_strips != 3:
        splits = [(int(i * W / num_strips), int((i + 1) * W / num_strips)) for i in range(num_strips)]
        measured_mm = [float((r - l) * standard_ratio_x) for (l, r) in splits]
        strips = [img[:, L:R] for (L, R) in splits]
        return strips, measured_mm, splits

    v1, v2 = valleys
    if v1 > v2:
        v1, v2 = v2, v1

    # 4) 导数 + 导数平滑（抗缺陷尖刺）
    d = np.gradient(profile_s)
    d = _smooth_1d(d, k=max(15, (W // 200) | 1)).astype(np.float32)
    ad = np.abs(d)

    gap_pad = max(30, W // 120)

    # ---------- 左外边界：上升段结束点 ----------
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
        edge = _pick_run_end(runs, prefer="left")  # 上升段结束点
        if edge is not None:
            L0 = a0 + edge

    # ---------- 右外边界：下降段结束点 ----------
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

        # 从右往左挑“真正下坡”（累计下降量过滤）
        dyn = float(np.percentile(profile_s, 90) - np.percentile(profile_s, 10))
        min_drop = max(3.0, 0.10 * dyn)

        pickR = None
        for (L, R) in sorted(runs, key=lambda lr: lr[1], reverse=True):
            drop = float(np.sum(-roi_d[L:R]))
            if drop >= min_drop:
                pickR = R  # 下降段结束点
                break

        if pickR is None:
            edge = _pick_run_end(runs, prefer="right")
            if edge is not None:
                pickR = edge

        if pickR is not None:
            R0 = a0 + int(pickR)

    # ---------- 黑缝边界 ----------
    g1L, g1R = _gap_bounds_by_derivative(d, ad, v1, W, q=92, min_th=0.2)
    g2L, g2R = _gap_bounds_by_derivative(d, ad, v2, W, q=92, min_th=0.2)

    # ---------- 初始 splits ----------
    splits = [
        (int(np.clip(L0, 0, W - 2)), int(np.clip(g1L, 1, W))),
        (int(np.clip(g1R, 0, W - 2)), int(np.clip(g2L, 1, W))),
        (int(np.clip(g2R, 0, W - 2)), int(np.clip(R0, 1, W))),
    ]

    # ---------- 单调修正 + 最小宽度 ----------
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
    splits = fixed

    # ---------- ✅ 右侧黑边精修（关键新增） ----------
    refined = []
    for (L, R) in splits:
        # 用灰度图做精修判断，避免颜色影响
        strip_gray = gray_u8[:, L:R]
        trim_r = _trim_right_black_border(strip_gray, max_trim=160, alpha=0.25)
        R2 = max(L + 2, R - trim_r)  # 防止裁成空
        refined.append((L, R2))
    splits = refined

    # ---------- 输出 ----------
    measured_mm = [float((r - l) * standard_ratio_x) for (l, r) in splits]
    strips = [img[:, L:R] for (L, R) in splits]
    return strips, measured_mm, splits







def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(folder: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    return files


def generate_dataset():
    print("\n🚀 正在生成数据集 (同步生成 good 和 bad 子文件夹) ...")

    # 定义数据类别映射
    sub_types = [
        {"suffix": "", "folder_name": "good"},  # cam1 -> good
        {"suffix": "_bad", "folder_name": "bad"}  # cam1_bad -> bad
    ]

    for cam_idx, (cam_folder_base, cam_out) in enumerate(zip(CAM_NAMES, CAM_OUTPUT)):

        for sub_t in sub_types:
            # 拼接输入路径，例如: D:\...\img_raw_26A\cam1_bad
            current_in_name = cam_folder_base + sub_t["suffix"]
            in_dir = os.path.join(RAW_ROOT, current_in_name)

            # 拼接输出路径，例如: D:\...\image_all\CAM1\train\bad
            out_dir = os.path.join(OUT_ROOT, cam_out, "train", sub_t["folder_name"])
            ensure_dir(out_dir)

            img_files = list_images(in_dir)
            if len(img_files) == 0:
                print(f"⚠️ 路径 {in_dir} 下没有找到图片，跳过")
                continue

            # 随机抽取（如果是 bad 样本，建议有多少取多少，或者设定不同的阈值）
            # 这里默认依然使用 TARGET_PER_CAM
            k = min(TARGET_PER_CAM, len(img_files))
            selected = random.sample(img_files, k)

            print(f"\n📸 {cam_out} [{sub_t['folder_name']}]: 发现 {len(img_files)} 张 -> 抽取 {len(selected)} 张")

            # 计数器和统计
            idx_save = {1: 1, 2: 1, 3: 1}
            count_written = {1: 0, 2: 0, 3: 0}
            split_times = []

            for img_path in tqdm(selected, desc=f"Processing {cam_out}_{sub_t['folder_name']}"):
                img = cv2.imread(img_path)
                if img is None or img.shape[0] != VALID_H or img.shape[1] != VALID_W:
                    continue

                t0 = time.perf_counter()
                strips, _, _ = split_multi_strips(img)
                t1 = time.perf_counter()
                split_times.append(t1 - t0)

                if not strips:
                    continue

                strips = strips[:MAX_STRIPS]

                for s_idx, strip in enumerate(strips, start=1):
                    if strip is None: continue
                    H, W = strip.shape[:2]

                    # 竖切逻辑
                    h_step = H // SPLIT_VERTICAL_PARTS
                    for part_id in range(SPLIT_VERTICAL_PARTS):
                        y0 = part_id * h_step
                        y1 = (part_id + 1) * h_step if part_id < SPLIT_VERTICAL_PARTS - 1 else H
                        part = strip[y0:y1, :, :]

                        # 文件名：前缀区分好坏，方便排查
                        # 例如: good_strip1_000362_part0.png
                        save_name = f"{sub_t['folder_name']}_strip{s_idx}_{idx_save[s_idx] + 361:06d}_part{part_id}.png"
                        save_path = os.path.join(out_dir, save_name)

                        # 强制转灰度保存（对应你的单通道模型）
                        if part.ndim == 3:
                            part = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)

                        if cv2.imwrite(save_path, part):
                            idx_save[s_idx] += 1
                            count_written[s_idx] += 1

            # 打印每个子文件夹的统计结果
            total_written = sum(count_written.values())
            print(f"   - {sub_t['folder_name']} 总输出: {total_written} 张")

            # 如果是 good 文件夹且数量太少，给个警告
            if sub_t['folder_name'] == "good" and total_written < 200:
                print(f"   ⚠️ 警告：{cam_out} 训练图 < 200 张，可能会影响训练代码稳定性。")

    print("\n🌟 数据集生成完毕！已按 CAM1/train/good 和 CAM1/train/bad 结构存放。")


if __name__ == "__main__":
    generate_dataset()
