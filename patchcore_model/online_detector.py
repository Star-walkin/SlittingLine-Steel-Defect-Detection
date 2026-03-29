import os
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# 确保可以从项目根目录导入 det_model / function_bank 等模块
import sys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from det_model.prepare_dataset_det import preprocess_like_inference  # noqa: E402
from det_model.infer import preprocess_like_inference_gpu             # noqa: E402
from function_bank import test_one_image  # noqa: E402


class FeatureExtractor(torch.nn.Module):
    """与 patchcore_model/train.py 中一致的 ResNet18 特征提取 backbone。"""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x3 = x.repeat(1, 3, 1, 1)
        x = self.stem(x3)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return {"layer2": f2, "layer3": f3}


def build_patch_embeddings(features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """与 patchcore_model/train.py / infer.py 中保持一致的特征拼接方式。"""
    f2 = features["layer2"]
    f3 = features["layer3"]
    h, w = f2.shape[-2:]
    f3_up = F.interpolate(f3, size=(h, w), mode="bilinear", align_corners=False)
    emb = torch.cat([f2, f3_up], dim=1)  # [B, C, H, W]
    emb = emb.permute(0, 2, 3, 1).contiguous().view(-1, emb.shape[1])  # [H*W, C]
    return emb, (h, w)


def knn_min_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """对每个 query patch 计算到记忆库的最小 L2 距离。"""
    out: List[torch.Tensor] = []
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk]
        d = torch.cdist(q, bank)
        out.append(d.min(dim=1).values)
    return torch.cat(out, dim=0)


def apply_soft_edge_suppression(
    score_map: np.ndarray,
    img_size: int,
    soft_border: int,
    strength: float = 1.0,
    weight_profile: str = "ease_out_cubic",
) -> np.ndarray:
    """
    在单块正方形热力图上做边缘抑制。

    Args:
        score_map: float32 2D，通常为 img_size x img_size
        img_size: 训练/推理边长
        soft_border: 边缘过渡带宽（像素）。<=0 表示不做抑制，返回副本。
        strength: 抑制强度，建议范围 [0, 1]。
            - 0：不抑制（与 soft_border 无关，直接返回副本）
            - 1：在带宽内从最外缘向内抬升权重，最外缘一行/列热力乘权为 0（硬边置零效果）
        weight_profile: 过渡曲线形状。
            - "linear": 线性 ramp（旧行为，过渡末端易与后续 diff 叠加产生边界尖峰）
            - "ease_out_cubic": 默认。w(t)=1-(1-t)^3，靠外缘变化快、靠内缘趋近 1 更平缓，
              减轻「边缘抑制结束处」在 H_diff = H - blur(H) 上的突变伪高亮。

    中间值在「原热力」与「强度为 1 的乘性掩膜结果」之间按 strength 插值。

    Returns:
        抑制后的热力图（float32）
    """
    out = np.asarray(score_map, dtype=np.float32).copy()
    s = float(np.clip(strength, 0.0, 1.0))
    if soft_border <= 0 or s <= 0.0:
        return out
    sb = min(int(soft_border), max(1, img_size // 2 - 1))
    Hm, Wm = out.shape

    def _ramp_weight(i: int, denom: float) -> float:
        t = i / denom  # 0 .. 1
        t = float(np.clip(t, 0.0, 1.0))
        if weight_profile == "linear":
            return t
        # ease-out cubic: 导数在 t=0 最大，在 t=1 附近趋于 0，过渡更平滑
        if weight_profile in ("ease_out_cubic", "ease_out"):
            return 1.0 - (1.0 - t) ** 3
        return 1.0 - (1.0 - t) ** 3

    # 与最外缘距离相关的权重：最外缘为 0，向内增至 band 内缘为 1
    wy = np.ones(Hm, dtype=np.float32)
    wx = np.ones(Wm, dtype=np.float32)
    denom = float(max(sb - 1, 1))
    for i in range(sb):
        w_edge = _ramp_weight(i, denom)
        wy[i] = min(wy[i], w_edge)
        wy[-i - 1] = min(wy[-i - 1], w_edge)
        wx[i] = min(wx[i], w_edge)
        wx[-i - 1] = min(wx[-i - 1], w_edge)

    weight2d = np.outer(wy, wx)
    # s=0 原图；s=1 为 out * weight2d（边缘为 0）
    out = out * (1.0 - s + s * weight2d)
    return out.astype(np.float32)


class PatchCoreDetector:
    """
    面向 detectoutline02 的 PatchCore 在线检测封装。

    - 与 SimpleAD / UNet 一样，提供 detect_ano 与 obtain_anomaly_location 接口
    - detect_ano: 条带图像 -> (test_cut, rec_cut, amap_cut)
    - obtain_anomaly_location: 直接复用 function_bank.test_one_image 中的实现
    """

    def __init__(
        self,
        memory_path: str,
        conduct_id: str,
        fukuan0: List[float],
        cut_ratio: int,
        img_size: int,
        seg_anomaly_thres: float,
        standard_ratio_x: float,
        standard_ratio_y: float,
        steel_real_y0: float,
        device: torch.device | None = None,
        patchcore_k: float | None = None,
        use_gradient_detection: bool = False,
        grad_threshold: float = 1.5,
        blur_ksize: int = 5,
        edge_crop: int = 20,
        bg_ksize: int = 101,
        diff_threshold: float = 1.0,
        patchcore_edge_soft_border: int = 20,
        patchcore_edge_strength: float = 1.0,
        patchcore_edge_weight_profile: str = "ease_out_cubic",
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 读取 PatchCore 记忆库
        ckpt = np.load(memory_path)
        self.memory_bank = ckpt["memory_bank"].astype(np.float32)
        # 记忆库内部训练时使用的 img_size（与当前配置保持一致）
        self.img_size = int(ckpt["img_size"])
        self.patchcore_threshold = float(ckpt["threshold"])

        # FP16 加速：在 CUDA 上将 backbone 和记忆库转为半精度，推理约 2x 加速
        self.use_fp16: bool = (self.device.type == "cuda")

        # PCA 降维加速（train_v2.py 训练时可选生成，存于 npz 中）
        # 若 npz 含 pca_components 则启用；否则 self.pca_components_t = None
        self.pca_components_t: Optional[torch.Tensor] = None
        self.pca_mean_t: Optional[torch.Tensor] = None
        if "pca_components" in ckpt and "pca_mean" in ckpt:
            pca_comp = ckpt["pca_components"].astype(np.float32)   # [n_components, C]
            pca_mean = ckpt["pca_mean"].astype(np.float32)         # [C]
            self.pca_mean_t = torch.from_numpy(pca_mean).to(self.device)
            self.pca_components_t = torch.from_numpy(pca_comp).to(self.device)
            if self.use_fp16:
                self.pca_mean_t = self.pca_mean_t.half()
                self.pca_components_t = self.pca_components_t.half()
            print(f"[PatchCoreDetector] PCA 已加载: {pca_comp.shape[1]}维 -> {pca_comp.shape[0]}维")

        # PatchCore 推理组件
        self.extractor = FeatureExtractor().to(self.device)
        if self.use_fp16:
            self.extractor = self.extractor.half()

        self.bank_t = torch.from_numpy(self.memory_bank).to(self.device)
        if self.use_fp16:
            self.bank_t = self.bank_t.half()

        self.tf = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # ====== 复用 test_one_image 的缺陷定位与坐标映射逻辑 ======
        self.patchcore_k = float(patchcore_k) if patchcore_k is not None else 3.0
        dummy_model = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self._locator = test_one_image(
            model=dummy_model,
            conduct_id=conduct_id,
            fukuan0=fukuan0,
            cut_ratio=cut_ratio,
            img_size=img_size,
            seg_anomaly_thres=seg_anomaly_thres,
            standard_ratio_x=standard_ratio_x,
            standard_ratio_y=standard_ratio_y,
            steel_real_y0=steel_real_y0,
            patchcore_k=self.patchcore_k,
            use_gradient_detection=bool(use_gradient_detection),
            grad_threshold=float(grad_threshold),
            blur_ksize=int(blur_ksize),
            edge_crop=int(edge_crop),
            bg_ksize=int(bg_ksize),
            diff_threshold=float(diff_threshold),
        )

        # PatchCore 正方形 tile 边缘抑制：带宽（像素）与强度 [0,1]（0 关闭，1 最外缘为 0）
        self.patchcore_edge_soft_border = int(patchcore_edge_soft_border)
        self.patchcore_edge_strength = float(patchcore_edge_strength)
        self.patchcore_edge_weight_profile = str(patchcore_edge_weight_profile or "ease_out_cubic")
        # 最近一次 detect_ano 的拼接前/后热力图，供 debug 展示「推理原图 vs 抑制后」
        self._last_amap_raw: np.ndarray | None = None
        self._last_amap_suppressed: np.ndarray | None = None

        # 让外部可以直接访问这些属性（便于与 SimpleAD / UNet 行为一致）
        self.cut_ratio = cut_ratio
        self.img_size = img_size
        self.seg_anomaly_thres = seg_anomaly_thres
        self.standard_ratio_x = standard_ratio_x
        self.standard_ratio_y = standard_ratio_y
        self.steel_real_y0 = steel_real_y0
        self.fukuan0 = fukuan0

    @torch.inference_mode()
    def _infer_one_tile(self, gray_tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        对单个条带子块 (2D 灰度) 运行 PatchCore：
        - 返回 (raw_heatmap, suppressed_heatmap, max_score)
        - raw 为平滑+resize 后、未做边缘软抑制；suppressed 为在正方形上做边缘抑制后的结果。
        - max_score 取 suppressed 上的最大值（与下游 amap_cut 一致）。
        """
        # 先 resize 到 PatchCore 训练时使用的 img_size
        g_resized = cv2.resize(gray_tile, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = self.tf(Image.fromarray(g_resized)).unsqueeze(0).to(self.device)
        if self.use_fp16:
            x = x.half()

        feats = self.extractor(x)
        emb, (h, w) = build_patch_embeddings(feats)

        dist = knn_min_distance(emb, self.bank_t)
        score_map = dist.view(h, w).detach().cpu().numpy().astype(np.float32)

        # 与离线 PatchCore 推理一致的后处理：平滑 + resize（此时尚未做边缘抑制）
        score_map = cv2.GaussianBlur(score_map, (0, 0), sigmaX=1.0)
        score_raw = cv2.resize(score_map, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )

        score_suppressed = apply_soft_edge_suppression(
            score_raw,
            self.img_size,
            self.patchcore_edge_soft_border,
            strength=self.patchcore_edge_strength,
            weight_profile=self.patchcore_edge_weight_profile,
        )
        score = float(score_suppressed.max())
        return score_raw, score_suppressed, score

    @torch.inference_mode()
    def _infer_batch(self, gray_tiles: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        对多个 tile 进行批量 PatchCore 推理（单次 GPU forward pass）。

        相比逐个调用 _infer_one_tile，在 cut_ratio=3 时约 2-3x 加速：
          串行：3 × ~167ms = ~500ms；批量：1 × ~200ms = ~200ms

        原理：FeatureExtractor.forward 中 x.repeat(1,3,1,1) 以及
        build_patch_embeddings 的 .view(-1, C) 均天然支持 batch > 1，
        无需修改骨干网络。

        返回: 与 tiles 等长的列表，每项为 (raw_heatmap, suppressed_heatmap, max_score)
        """
        N = len(gray_tiles)
        if N == 0:
            return []

        # 构建 batch tensor [N, 1, H, W]
        tensors = []
        for tile in gray_tiles:
            g_resized = cv2.resize(tile, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            tensors.append(self.tf(Image.fromarray(g_resized)))
        x = torch.stack(tensors, dim=0).to(self.device)  # [N, 1, H, W]
        if self.use_fp16:
            x = x.half()

        # 单次批量 forward（FeatureExtractor 天然支持 batch > 1）
        feats = self.extractor(x)
        emb, (h, w) = build_patch_embeddings(feats)   # emb: [N*h*w, C]

        # 可选 PCA 降维（由 train_v2.py 生成，减小 KNN 距离计算量）
        if self.pca_components_t is not None:
            emb = (emb - self.pca_mean_t) @ self.pca_components_t.T  # [N*h*w, n_comp]

        dist = knn_min_distance(emb, self.bank_t)      # [N*h*w]
        dist_map = dist.view(N, h, w).detach().cpu().float().numpy()  # [N, h, w]

        results: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for i in range(N):
            score_map = dist_map[i].copy()
            score_map = cv2.GaussianBlur(score_map, (0, 0), sigmaX=1.0)
            score_raw = cv2.resize(
                score_map, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            score_suppressed = apply_soft_edge_suppression(
                score_raw,
                self.img_size,
                self.patchcore_edge_soft_border,
                strength=self.patchcore_edge_strength,
                weight_profile=self.patchcore_edge_weight_profile,
            )
            results.append((score_raw, score_suppressed, float(score_suppressed.max())))
        return results

    @torch.inference_mode()
    def detect_ano(self, image, left_edge: int = 0, right_edge: int | None = None):
        """
        条带级 PatchCore 推理。

        输入:
          - image: 条带整幅图 (H, W) 或 (H, W, 3)
        输出:
          - test_cut: (cut_ratio * img_size, img_size) 的 2D 灰度图
          - rec_cut: 同尺寸的灰度图（此处用 128 常数图占位）
          - amap_cut: 同尺寸浮点热力图（每块正方形上已做边缘抑制后再纵向拼接）。
            未抑制的拼接结果保存在 self._last_amap_raw，供 debug 对比。
        """
        if right_edge is None:
            if image.ndim == 3:
                right_edge = image.shape[1]
            else:
                right_edge = image.shape[1]

        # 1. 转灰度并做与训练/离线推理一致的预处理（裁边 + FFT 去纹 + 纵向滤波 + 中值拍平）
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.asarray(image, dtype=np.uint8)

        # 与 preprocess 内部一致：用于缺陷图从原条带映射坐标（裁边前列索引）
        H0, W0 = gray.shape[:2]
        bias = 15
        l_edge = max(bias, 0)
        r_edge = min(W0 - bias, W0)
        if r_edge <= l_edge:
            l_edge, r_edge = 0, W0

        # GPU 预处理（CUDA 可用时）：FFT+纵向滤波+背景拍平全部在 GPU 上完成，
        # 速度从 ~554ms 降至 ~28ms；CPU 环境自动回退到原版。
        if self.device.type == "cuda":
            gray_proc = preprocess_like_inference_gpu(gray, self.device, bias=15)
        else:
            gray_proc = preprocess_like_inference(gray, bias=15)
        H, W = gray_proc.shape

        # 2. 按 cut_ratio 在竖直方向均匀切分
        tile_h = H // self.cut_ratio
        tiles: List[np.ndarray] = []
        for i in range(self.cut_ratio):
            y0 = i * tile_h
            y1 = (i + 1) * tile_h if i < self.cut_ratio - 1 else H
            tile = gray_proc[y0:y1, :]
            if tile.size == 0:
                tile = gray_proc
            tiles.append(tile)

        # 供 obtain_anomaly_location 从原条带裁 256×256（与 UNet 路径一致，写入 _locator）
        self._last_tiles_u8 = [np.copy(t) for t in tiles]
        self._locator._last_tiles_u8 = self._last_tiles_u8
        self._locator._last_cut_l = l_edge
        self._locator._last_cut_r = r_edge
        self._locator._last_detect_mode = "tiled"

        # 3. 批量对所有 tile 跑 PatchCore（单次 GPU forward，比串行约 2-3x 快）
        test_tiles_rs: List[np.ndarray] = []
        heat_tiles_raw: List[np.ndarray] = []
        heat_tiles_sup: List[np.ndarray] = []
        batch_results = self._infer_batch(tiles)
        for tile, (heat_raw, heat_sup, _) in zip(tiles, batch_results):
            # 可视化输入：与 heat_map 同尺寸的灰度图
            tile_vis = cv2.resize(tile, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            test_tiles_rs.append(tile_vis)
            heat_tiles_raw.append(heat_raw)
            heat_tiles_sup.append(heat_sup)

        # 纵向拼接得到整幅 cut 图与热力图（业务与 obtain_anomaly_location 使用抑制后的条带）
        test_cut = np.vstack(test_tiles_rs).astype(np.uint8)
        amap_cut = np.vstack(heat_tiles_sup).astype(np.float32)
        amap_raw = np.vstack(heat_tiles_raw).astype(np.float32)
        self._last_amap_raw = amap_raw
        self._last_amap_suppressed = amap_cut

        # PatchCore 没有重构图，这里用常数图占位
        rec_cut = np.full_like(test_cut, 128, dtype=np.uint8)

        return test_cut, rec_cut, amap_cut

    # obtain_anomaly_location 直接复用 test_one_image 的实现
    def obtain_anomaly_location(self, amap, test_img, anomaly_save_path, image_id, fukuan=None, original_strip_for_crop=None):
        return self._locator.obtain_anomaly_location(
            amap, test_img, anomaly_save_path, image_id, fukuan,
            original_strip_for_crop=original_strip_for_crop,
        )


# ---------------------------------------------------------------------------
# InferEngine：每相机专用 GPU 推理线程
# ---------------------------------------------------------------------------
class InferEngine:
    """
    每相机独立的 GPU 推理线程。

    设计目标：
      - 将 detect_ano + obtain_anomaly_location 从 worker 的 cam_lock 中抽出，
        使两个 worker 线程可以在 GPU 处理上一帧时，同步做下一帧的 CPU 预处理（decode/split）。
      - 每相机仅 1 个 GPU 线程，避免多线程抢 CUDA 上下文带来的切换开销。
      - detect() 内部仍会自行加 cam_lock 保护 list 写入，不破坏原有线程安全性。

    使用方式：
        engine = InferEngine(cam_id)
        # 向引擎提交单条带任务，立即返回 Future；worker 可在提交后继续 CPU 工作
        future = engine.submit(detect_fn, strip_img, *detect_args)
        result = future.result()  # 阻塞直到推理完成
    """

    class _Future:
        """轻量 Future：包装单次推理任务的结果与完成事件。"""
        __slots__ = ("_event", "_result", "_exc")

        def __init__(self) -> None:
            self._event: threading.Event = threading.Event()
            self._result: Any = None
            self._exc: Optional[BaseException] = None

        def _set_result(self, value: Any) -> None:
            self._result = value
            self._event.set()

        def _set_exception(self, exc: BaseException) -> None:
            self._exc = exc
            self._event.set()

        def result(self) -> Any:
            """阻塞直到推理完成，若推理异常则重新抛出。"""
            self._event.wait()
            if self._exc is not None:
                raise self._exc
            return self._result

    def __init__(self, cam_id: int, maxsize: int = 8) -> None:
        self._cam_id = cam_id
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"InferEngine-CAM{cam_id + 1}",
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            fn, args, kwargs, future = item
            try:
                future._set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()
                future._set_exception(exc)

    def submit(self, fn, *args, **kwargs) -> "_Future":
        """
        提交推理任务，立即返回 Future（不阻塞）。
        调用 future.result() 等待结果。
        """
        future = InferEngine._Future()
        self._q.put((fn, args, kwargs, future))
        return future

    def stop(self) -> None:
        """优雅停止推理线程（发送哨兵 None）。"""
        self._q.put(None)

