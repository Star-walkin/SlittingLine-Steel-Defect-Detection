import argparse
import glob
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


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
    emb = torch.cat([f2, f3_up], dim=1)  # [1, C, H, W]
    emb = emb.permute(0, 2, 3, 1).contiguous().view(-1, emb.shape[1])  # [H*W, C]
    return emb, (h, w)


def knn_min_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = []
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk]
        d = torch.cdist(q, bank)
        out.append(d.min(dim=1).values)
    return torch.cat(out, dim=0)


def list_input_images(inp: str) -> List[str]:
    if os.path.isdir(inp):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(glob.glob(os.path.join(inp, ext)))
        return sorted(files)
    if os.path.isfile(inp):
        return [inp]
    raise FileNotFoundError(f"输入路径不存在: {inp}")


@torch.inference_mode()
def run_infer(args):
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = np.load(args.model_path)
    memory_bank = ckpt["memory_bank"].astype(np.float32)
    img_size = int(ckpt["img_size"])
    threshold = float(ckpt["threshold"])

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = FeatureExtractor().to(device)
    bank_t = torch.from_numpy(memory_bank).to(device)

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    files = list_input_images(args.input)
    print(f"[PatchCore] 待推理图像数量: {len(files)}")

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
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_vis.png"), vis)

        label = "NG" if score > threshold else "OK"
        print(f"{os.path.basename(p)} | score={score:.6f} | threshold={threshold:.6f} | {label}")


def parse_args():
    p = argparse.ArgumentParser(description="PatchCore 推理")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练输出的 patchcore_memory.npz 文件路径",
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="单张图片路径或图片目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=r"patchcore_model\output",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_infer(parse_args())
