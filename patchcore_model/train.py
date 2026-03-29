import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class GoodImageDataset(Dataset):
    def __init__(self, data_dir: str, img_size: int):
        pattern = os.path.join(data_dir, "train", "good", "*.*")
        self.images = sorted(glob.glob(pattern))
        if not self.images:
            raise FileNotFoundError(f"没有找到训练图像: {pattern}")
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        gray = Image.open(path).convert("L")
        return self.tf(gray), path


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(
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
    f2 = features["layer2"]
    f3 = features["layer3"]
    h, w = f2.shape[-2:]
    f3_up = F.interpolate(f3, size=(h, w), mode="bilinear", align_corners=False)
    emb = torch.cat([f2, f3_up], dim=1)  # [B, C, H, W]
    emb = emb.permute(0, 2, 3, 1).contiguous().view(-1, emb.shape[1])  # [H*W, C]
    return emb, (h, w)


def sample_coreset(memory_bank: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    n = memory_bank.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return memory_bank
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return memory_bank[idx]


def knn_min_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = []
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk]
        d = torch.cdist(q, bank)
        out.append(d.min(dim=1).values)
    return torch.cat(out, dim=0)


def _list_test_images(root: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, e)))
    return sorted(files)


@torch.inference_mode()
def _eval_on_testset(
    args,
    cam: str,
    device: torch.device,
    extractor: FeatureExtractor,
    memory_bank: np.ndarray,
    threshold: float,
    epoch: int,
    img_size: int,
    save_dir: str,
):
    if not args.test_root:
        return

    test_exp = args.test_exp_name or args.exp_name
    test_dir = os.path.join(args.test_root, test_exp, cam, "test", "images")
    if not os.path.isdir(test_dir):
        print(f"[PatchCore][{cam}] 未找到测试集目录: {test_dir}，跳过评估")
        return

    files = _list_test_images(test_dir)
    if not files:
        print(f"[PatchCore][{cam}] 测试集空目录: {test_dir}，跳过评估")
        return

    out_dir = os.path.join(save_dir, f"eval_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PatchCore][{cam}] 评估 epoch {epoch}，测试图像 {len(files)} 张 -> {out_dir}")

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    bank_t = torch.from_numpy(memory_bank).to(device)

    for p in files:
        gray = Image.open(p).convert("L")
        x = tf(gray).unsqueeze(0).to(device)
        feats = extractor(x)
        emb, (h, w) = build_patch_embeddings(feats)

        dist = knn_min_distance(emb, bank_t)
        score_map = dist.view(h, w).detach().cpu().numpy().astype(np.float32)

        # 先平滑再 resize 到输出尺寸
        score_map = cv2.GaussianBlur(score_map, (0, 0), sigmaX=1.0)
        score_map = cv2.resize(score_map, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # 在分数图上添加 20 像素左右的边缘掩膜（不改原始图，只抑制边缘响应）
        border = min(20, img_size // 4)
        score_map[:border, :] = 0.0
        score_map[-border:, :] = 0.0
        score_map[:, :border] = 0.0
        score_map[:, -border:] = 0.0

        # 掩膜后再计算整体分数
        score = float(score_map.max())
        norm = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        heat = np.uint8(norm * 255)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        g = np.array(gray.resize((img_size, img_size)))
        g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # 仅输出“原图 + 热力图”左右拼接
        vis = np.hstack([g3, heat])

        stem = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_vis.png"), vis)

        label = "NG" if score > threshold else "OK"
        print(f"[{cam}] {os.path.basename(p)} | score={score:.6f} | threshold={threshold:.6f} | {label}")


@torch.inference_mode()
def train_one_cam(args, cam: str, device: torch.device):
    data_dir = os.path.join(args.data_root, args.exp_name, cam)
    save_dir = os.path.join(args.out_root, args.exp_name, cam)
    os.makedirs(save_dir, exist_ok=True)

    ds = GoodImageDataset(data_dir, args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    extractor = FeatureExtractor().to(device)

    print(f"\n[PatchCore] 开始训练 {cam}")
    print(f"  data_dir={data_dir}")
    print(f"  images={len(ds)}")

    for epoch in range(1, args.epochs + 1):
        all_embeddings = []
        # 为了 coreset 随机采样具备“epoch 概念”，每轮使用不同随机种子
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + epoch)

        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            feats = extractor(images)
            emb, _ = build_patch_embeddings(feats)

            if args.max_patches_per_batch > 0 and emb.shape[0] > args.max_patches_per_batch:
                idx = torch.randperm(emb.shape[0], device=emb.device)[: args.max_patches_per_batch]
                emb = emb[idx]
            all_embeddings.append(emb.cpu())

        memory_bank = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
        memory_bank = sample_coreset(memory_bank, ratio=args.coreset_ratio, seed=args.seed + epoch)

        bank_t = torch.from_numpy(memory_bank).to(device)
        # 用记忆库内部最近邻距离统计默认阈值（无标签场景下的启发式阈值）
        dist = knn_min_distance(bank_t, bank_t)
        threshold = float(np.percentile(dist.detach().cpu().numpy(), args.threshold_percentile))

        # 保存 epoch 权重
        npz_epoch = os.path.join(save_dir, f"patchcore_memory_epoch_{epoch}.npz")
        np.savez_compressed(
            npz_epoch,
            memory_bank=memory_bank,
            img_size=np.int32(args.img_size),
            threshold=np.float32(threshold),
        )

        # 更新默认权重（始终指向最后一次）
        npz_path = os.path.join(save_dir, "patchcore_memory.npz")
        np.savez_compressed(
            npz_path,
            memory_bank=memory_bank,
            img_size=np.int32(args.img_size),
            threshold=np.float32(threshold),
        )

        meta = {
            "camera": cam,
            "data_dir": data_dir,
            "num_images": len(ds),
            "memory_bank_size": int(memory_bank.shape[0]),
            "embedding_dim": int(memory_bank.shape[1]),
            "img_size": int(args.img_size),
            "coreset_ratio": float(args.coreset_ratio),
            "threshold_percentile": float(args.threshold_percentile),
            "score_threshold": float(threshold),
            "epoch": int(epoch),
        }
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(
            f"[PatchCore][{cam}] epoch {epoch}/{args.epochs} "
            f"memory={memory_bank.shape}, threshold={threshold:.6f}"
        )

        # 训练结束后在测试集上跑一次推理并输出热力图
        if epoch == args.epochs:
            _eval_on_testset(
                args=args,
                cam=cam,
                device=device,
                extractor=extractor,
                memory_bank=memory_bank,
                threshold=threshold,
                epoch=epoch,
                img_size=args.img_size,
                save_dir=save_dir,
            )


def parse_args():
    p = argparse.ArgumentParser(description="PatchCore 训练（接口兼容 det_model 数据目录）")
    p.add_argument("--data_root", type=str, default=r"image_all")
    p.add_argument("--exp_name", type=str, default="image_data_patchcore_0228")
    p.add_argument("--cams", type=str, default="CAM1,CAM2,CAM3,CAM4")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--coreset_ratio", type=float, default=0.1)
    p.add_argument("--max_patches_per_batch", type=int, default=20000)
    p.add_argument("--threshold_percentile", type=float, default=99.5)
    p.add_argument(
        "--out_root",
        type=str,
        default=r"patchcore_model\weights",
    )
    # 训练“epoch”次数（主要影响 coreset 采样；结束后做一次测试集推理）
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval_interval", type=int, default=10, help="已弃用：现仅在最后一轮推理")
    # 测试集根目录与实验名（由 make_test_dataset.py 生成）
    p.add_argument(
        "--test_root",
        type=str,
        default=r"patchcore_model\test_data",
        help="PatchCore 测试集根目录（为空则跳过评估）",
    )
    p.add_argument(
        "--test_exp_name",
        type=str,
        default="patchcore_test",
        help="测试集实验名，默认使用 make_test_dataset.py 生成的 patchcore_test",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cam in [x.strip() for x in args.cams.split(",") if x.strip()]:
        train_one_cam(args, cam, device)


if __name__ == "__main__":
    main()
