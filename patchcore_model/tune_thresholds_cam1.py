import os
import glob
import json
import cv2
import numpy as np
import torch

# 将 patchcore_model 目录作为根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 本脚本位于 patchcore_model 目录下，直接从同级模块导入
from online_detector import FeatureExtractor, build_patch_embeddings, knn_min_distance  # noqa: E402


class PatchCoreWrapper:
    """
    简化版 PatchCore 推理封装：
    - 读取 patchcore_memory.npz（memory_bank, img_size）
    - 使用 ResNet18 特征提取 + KNN 最近邻距离
    - 返回与原图尺寸一致的 2D 浮点热力图（不做伪彩、不归一化到 0-255）
    """

    def __init__(self, memory_path: str, device: torch.device):
        if not os.path.isfile(memory_path):
            raise FileNotFoundError(f"PatchCore 记忆库不存在: {memory_path}")

        ckpt = np.load(memory_path)
        memory_bank = ckpt["memory_bank"].astype(np.float32)
        self.img_size = int(ckpt["img_size"])

        self.device = device
        self.extractor = FeatureExtractor().to(self.device)
        self.bank_t = torch.from_numpy(memory_bank).to(self.device)

        from torchvision import transforms

        self.tf = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @torch.inference_mode()
    def infer_to_heatmap(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        输入 BGR 原图，输出与原图同分辨率的 2D 浮点热力图（越大越异常）。
        不做伪彩、不归一化到 0-255。
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        orig_h, orig_w = gray.shape

        gray_rs = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = self.tf(gray_rs).unsqueeze(0).to(self.device)

        feats = self.extractor(x)
        emb, (h, w) = build_patch_embeddings(feats)

        dist = knn_min_distance(emb, self.bank_t)
        score_map = dist.view(h, w).detach().cpu().numpy().astype(np.float32)

        score_map = cv2.GaussianBlur(score_map, (0, 0), sigmaX=1.0)
        score_map = cv2.resize(score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return score_map


def infer_image_to_heatmap(image_path, model: PatchCoreWrapper, device="cuda"):
    """
    读取原图，送入 PatchCore 模型，返回原始浮点数热力图矩阵 (2D numpy array)。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 设备兜底（模型内部已经绑定 device，这里主要保证字符串合法）
    _ = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    heatmap = model.infer_to_heatmap(img)
    if heatmap.ndim != 2:
        raise RuntimeError(f"heatmap 不是 2D 矩阵: {image_path}, shape={heatmap.shape}")
    return heatmap


def process_folder(folder_path, model, edge_crop, device="cuda"):
    """遍历文件夹，执行推理并返回所有热力图和有效区域掩膜"""
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        print(f"[WARN] 目录中没有图片: {folder_path}")
        return [], None, []

    print(f"正在处理 [{folder_path}]，共找到 {len(image_paths)} 张图片...")
    all_heatmaps = []
    valid_mask = None

    for path in sorted(image_paths):
        try:
            heatmap = infer_image_to_heatmap(path, model, device)
        except Exception as e:
            print(f"[ERROR] 推理失败 {os.path.basename(path)}: {e}")
            continue

        if valid_mask is None:
            h, w = heatmap.shape
            valid_mask = np.zeros((h, w), dtype=np.uint8)
            ec = int(edge_crop)
            ec = max(0, min(ec, h // 2 - 1, w // 2 - 1))
            valid_mask[ec : h - ec, ec : w - ec] = 1

        all_heatmaps.append(heatmap)

    return all_heatmaps, valid_mask, image_paths


def calculate_T_min(perfect_heatmaps, valid_mask, safety_margin=0.05):
    max_scores = []
    for hm in perfect_heatmaps:
        valid_scores = hm[valid_mask > 0]
        if len(valid_scores) > 0:
            max_scores.append(float(np.max(valid_scores)))
    if not max_scores:
        return 0.0
    absolute_max = float(np.max(max_scores))
    T_min = absolute_max * (1.0 + safety_margin)
    print(f"\n[T_min 推荐] 完美最高分 {absolute_max:.6f}，设定 T_min = {T_min:.6f}")
    return T_min


def calculate_K(textured_heatmaps, valid_mask):
    required_ks = []
    for hm in textured_heatmaps:
        valid_scores = hm[valid_mask > 0]
        if len(valid_scores) == 0:
            continue
        mu = float(np.median(valid_scores))
        sigma = float(np.std(valid_scores))
        max_val = float(np.max(valid_scores))
        if sigma > 0:
            required_ks.append((max_val - mu) / sigma)
    if not required_ks:
        return 3.0
    max_k = float(np.max(required_ks))
    recommended_K = max_k * 1.05
    print(f"\n[K 值推荐] 纹理最大波动需求 K {max_k:.6f}，设定 K = {recommended_K:.6f}")
    return recommended_K


def verify_thresholds(defect_heatmaps, defect_paths, valid_mask, T_min, K):
    print("\n--- 开始验证真实缺陷样本 ---")
    missed_count = 0
    for hm, path in zip(defect_heatmaps, defect_paths):
        valid_scores = hm[valid_mask > 0]
        if len(valid_scores) == 0:
            print(f"[WARN] 有效区域为空: {os.path.basename(path)}")
            continue

        mu = float(np.median(valid_scores))
        sigma = float(np.std(valid_scores))
        T_dynamic = mu + K * sigma

        cond_abs = valid_scores > T_min
        cond_dyn = valid_scores > T_dynamic
        defect_pixels = int(np.sum(cond_abs & cond_dyn))

        if defect_pixels == 0:
            print(f"❌ 漏检: {os.path.basename(path)} (T_min={T_min:.6f}, T_dyn={T_dynamic:.6f})")
            missed_count += 1
        else:
            print(
                f"✅ 检出: {os.path.basename(path)} "
                f"(异常像素数: {defect_pixels}, T_dyn={T_dynamic:.6f})"
            )
    if missed_count == 0:
        print("\n🎉 验证成功！当前阈值在给定样本上未发生漏检。")
    else:
        print(f"\n共有 {missed_count} 张缺陷图可能被漏检，建议适当调低 T_min 或 K。")


if __name__ == "__main__":
    # 数据目录放在 patchcore_model/data 下，便于集中管理
    dir_perfect = os.path.join(ROOT, "data", "01_perfect_normals")
    dir_textured = os.path.join(ROOT, "data", "02_textured_normals")
    dir_defects = os.path.join(ROOT, "data", "03_true_defects")

    edge_crop = 50  # 边缘舍弃像素数

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"使用设备: {device}")

    # 以 CAM1 的 PatchCore 记忆库为例
    memory_path = os.path.join(
        ROOT,
        "weights",
        "image_data_patchcore_0228",
        "CAM1",
        "patchcore_memory.npz",
    )
    print(f"加载 PatchCore 记忆库: {memory_path}")
    my_patchcore_model = PatchCoreWrapper(memory_path, device)

    # 1) 完美正常样本 → 估计 T_min
    perfect_hms, valid_mask, _ = process_folder(dir_perfect, my_patchcore_model, edge_crop, device_str)
    best_T_min = calculate_T_min(perfect_hms, valid_mask) if perfect_hms else 0.0

    # 2) 粗糙纹理正常样本 → 估计 K
    textured_hms, _, _ = process_folder(dir_textured, my_patchcore_model, edge_crop, device_str)
    best_K = calculate_K(textured_hms, valid_mask) if textured_hms else 3.0

    # 3) 含缺陷样本 → 验证 (T_min, K)
    defect_hms, _, defect_paths = process_folder(dir_defects, my_patchcore_model, edge_crop, device_str)
    if defect_hms and perfect_hms and textured_hms:
        verify_thresholds(defect_hms, defect_paths, valid_mask, best_T_min, best_K)

    # 4) 将推荐参数保存到 JSON 文件，便于写回 config.yaml 或其他配置
    out_cfg = {
        "T_min": float(best_T_min),
        "K": float(best_K),
        "edge_crop": int(edge_crop),
        "memory_path": memory_path,
    }
    out_path = os.path.join(ROOT, "patchcore_thresholds_cam1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)

    print(f"\n推荐阈值已保存到: {out_path}")

