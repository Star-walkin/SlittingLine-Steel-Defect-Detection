"""
det_model/train.py
按相机分别训练 SimpleAD 模型，保存到 det_model/train-result/<exp_name>/CAMx/last.pth
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from det_model.model import build_model
from det_model.dataset import get_dataloader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_cam(args, dataset_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    if getattr(args, "device", None) is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset_name}] 使用设备: {device}")

    torch.backends.cudnn.benchmark = True
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    data_root = getattr(args, "data_root", r"image_all")
    exp_name = getattr(args, "exp_name", "image_data_02_27")
    data_dir = os.path.join(data_root, exp_name, dataset_name)
    preprocessed = getattr(args, "preprocessed", False)

    train_loader = get_dataloader(
        data_dir, img_size=args.img_size, batch_size=args.batch_size, preprocessed=preprocessed
    )

    model = build_model(base_ch=32).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    mse = nn.MSELoss()

    log_path = os.path.join(save_dir, "loss.log")
    last_path = os.path.join(save_dir, "last.pth")

    print(f"\n==== Train {dataset_name} ====")
    print(f"save_dir: {save_dir}")

    for epoch in range(args.epochs):
        model.train()
        lr_now = args.lr if epoch < args.lr_decay_epoch else args.lr * 0.1
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        epoch_loss = 0.0
        n_batch = 0
        t0 = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"{dataset_name} Epoch {epoch+1}/{args.epochs} lr={lr_now:.1e}",
            leave=False,
        )

        for normal_img, anomaly_img, _ in pbar:
            normal_img = normal_img.to(device, non_blocking=True)
            anomaly_img = anomaly_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                rec = model(anomaly_img)
                loss = mse(normal_img, rec)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(1, n_batch)
        elapsed = time.perf_counter() - t0

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{dataset_name}] Epoch {epoch+1}/{args.epochs}  lr={lr_now:.2e}  loss={avg_loss:.6f}  time={elapsed:.2f}s\n"
            )

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
            try:
                model.eval()
                with torch.no_grad():
                    rec_vis = model(anomaly_img[:1])
                    vis_in = (anomaly_img[0].cpu() * 0.5 + 0.5).clamp(0, 1) * 255
                    vis_out = (rec_vis[0].cpu() * 0.5 + 0.5).clamp(0, 1) * 255
                    import cv2
                    debug = np.hstack([
                        vis_in[0].numpy().astype(np.uint8),
                        vis_out[0].numpy().astype(np.uint8),
                    ])
                    cv2.imwrite(
                        os.path.join(save_dir, f"vis_epoch_{epoch+1}.png"), debug
                    )
            except Exception as e:
                print(f"Vis error: {e}")
            model.train()

        torch.save(model.state_dict(), last_path)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"[{dataset_name}] epoch {epoch+1}/{args.epochs}  loss={avg_loss:.6f}")

    print(f"Done {dataset_name}. Checkpoint: {last_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=r"image_all")
    p.add_argument("--exp_name", type=str, default="image_data_02_28")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr_decay_epoch", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preprocessed", action="store_true",
                   help="数据已由 prepare_dataset_det.py 生成（与推理一致预处理），不再做 flatten/crop")
    p.add_argument("--device", type=str, default=None,
                   help="训练设备，如 cuda / cuda:0 / cpu；默认自动：有 GPU 用 cuda 否则 cpu")
    args = p.parse_args()

    det_model_root = os.path.dirname(os.path.abspath(__file__))
    result_root = os.path.join(det_model_root, "train-result", args.exp_name+"A")

    for cam in ["CAM1", "CAM2", "CAM3", "CAM4"]:
        save_dir = os.path.join(result_root, cam)
        train_one_cam(args, cam, save_dir)


if __name__ == "__main__":
    main()
