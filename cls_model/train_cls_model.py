from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass(frozen=True)
class Sample:
    path: str
    y0: int  # 0-based label


def _build_samples(dataset_root: str, mapping: Dict[str, int]) -> Tuple[List[Sample], List[str]]:
    """
    mapping: {folder_name: 1based_id}
    return samples, warnings
    """
    warns: List[str] = []
    dataset_root = str(dataset_root or "").strip()
    if not dataset_root or not os.path.isdir(dataset_root):
        return [], ["dataset_root 不存在或不可访问。"]

    # ensure ids are 1..K continuous
    ids = sorted({int(v) for v in mapping.values()})
    if not ids or ids[0] != 1 or ids != list(range(1, ids[-1] + 1)):
        warns.append("mapping 的类别ID建议为连续 1..K（否则报告/允收理解困难）。")

    samples: List[Sample] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for folder_name, cid1 in mapping.items():
        class_dir = os.path.join(dataset_root, str(folder_name))
        if not os.path.isdir(class_dir):
            warns.append(f"类别目录不存在：{class_dir}")
            continue
        y0 = int(cid1) - 1
        for fn in os.listdir(class_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            samples.append(Sample(path=os.path.join(class_dir, fn), y0=y0))

    if len(samples) == 0:
        warns.append("未找到任何图片样本。")
    return samples, warns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="train_config.yaml from UI")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    dataset_root = str(cfg.get("dataset_root", "") or "")
    mapping = cfg.get("mapping") or {}
    epochs = int(cfg.get("epochs", 15))
    batch_size = int(cfg.get("batch_size", 32))
    img_size = int(cfg.get("img_size", 128))
    out_ckpt = str(cfg.get("out_checkpoint", "") or "").strip()
    out_metrics = str(cfg.get("out_metrics", "") or "").strip()
    out_classes_json = str(cfg.get("out_classes_json", "") or "").strip()
    seed = int(cfg.get("seed", 42))

    _seed_everything(seed)

    # heavy imports after validation
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    # repo root: .. (this file is cls_model/train_cls_model.py)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # ProjectionNet is at repo_root/cls_model.py
    import sys

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from cls_model import ProjectionNet  # type: ignore

    # mapping keys may be non-str; normalize
    norm_mapping: Dict[str, int] = {}
    for k, v in dict(mapping).items():
        try:
            cid = int(v)
        except Exception:
            continue
        if cid <= 0:
            continue
        norm_mapping[str(k)] = cid

    samples, warns = _build_samples(dataset_root, norm_mapping)
    for w in warns:
        print(f"[WARN] {w}", flush=True)
    if len(samples) == 0:
        print("[ERR] empty dataset", flush=True)
        return 2

    k = max(int(v) for v in norm_mapping.values())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class ImgDs(Dataset):
        def __init__(self, ss: List[Sample]):
            self.ss = ss

        def __len__(self) -> int:
            return len(self.ss)

        def __getitem__(self, idx: int):
            s = self.ss[idx]
            try:
                im = Image.open(s.path).convert("RGB")
                x = tfm(im)
            except Exception:
                # unreadable: return a black image but keep label; log once per batch is too noisy
                x = torch.zeros((3, img_size, img_size), dtype=torch.float32)
            return x, int(s.y0)

    random.shuffle(samples)
    ds = ImgDs(samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    head_layers = [512] * 1 + [128]
    model = ProjectionNet(pretrained=True, head_layers=head_layers, num_classes=k)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    best_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(xb.size(0))
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))

        acc = float(correct) / float(max(1, total))
        avg_loss = loss_sum / float(max(1, total))
        print(f"epoch {ep}/{epochs} loss={avg_loss:.4f} acc={acc:.4f}", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"done best_acc={best_acc:.4f} elapsed_sec={elapsed:.1f}", flush=True)

    if not out_ckpt:
        print("[ERR] out_checkpoint missing", flush=True)
        return 3
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, out_ckpt)
    print(f"saved checkpoint: {out_ckpt}", flush=True)

    # classes.json is written by UI, but training script can ensure it's present
    if out_classes_json and not os.path.exists(out_classes_json):
        id_to_name = {str(v): str(k) for k, v in sorted(norm_mapping.items(), key=lambda x: int(x[1]))}
        _atomic_write_json(out_classes_json, id_to_name)

    if out_metrics:
        _atomic_write_json(
            out_metrics,
            {
                "best_acc": best_acc,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "num_classes": k,
                "device": device,
                "elapsed_sec": elapsed,
                "dataset_size": len(samples),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

