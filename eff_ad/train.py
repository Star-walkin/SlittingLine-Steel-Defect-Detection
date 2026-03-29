"""
eff_ad/train.py
按相机训练 Student-Teacher，保存到 eff_ad/train-result/<exp_name>/CAMx/
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, ROOT)

from eff_ad.model import build_eff_ad
from eff_ad.dataset import get_dataloader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_cam(args, dataset_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    data_root = getattr(args, "data_root", r"image_all")
    exp_name = getattr(args, "exp_name", "image_data_02_27")
    data_dir = os.path.join(data_root, exp_name, dataset_name)

    train_loader = get_dataloader(data_dir, img_size=args.img_size, batch_size=args.batch_size)
    model = build_eff_ad(feat_channels=128).to(device)
    optimizer = torch.optim.Adam(model.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = os.path.join(save_dir, "loss.log")
    last_path = os.path.join(save_dir, "last.pth")

    print(f"\n==== eff_ad Train {dataset_name} ====")
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

        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            s_feat, t_feat = model.forward_train(x)
            loss = F.mse_loss(s_feat, t_feat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=5.0)
            optimizer.step()

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
            ckpt = {
                "teacher": model.teacher.state_dict(),
                "student": model.student.state_dict(),
            }
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
            try:
                model.eval()
                with torch.no_grad():
                    amap = model.forward_anomap(x[:1])
                    amap_np = amap[0, 0].cpu().numpy()
                    amap_np = (amap_np - amap_np.min()) / (amap_np.max() - amap_np.min() + 1e-8) * 255
                    amap_u8 = np.uint8(amap_np)
                    amap_big = __import__("cv2").resize(amap_u8, (256, 256), interpolation=__import__("cv2").INTER_LINEAR)
                    inp_np = (x[0, 0].cpu().numpy() * 0.5 + 0.5) * 255
                    inp_u8 = np.uint8(np.clip(inp_np, 0, 255))
                    heat = __import__("cv2").applyColorMap(amap_big, __import__("cv2").COLORMAP_JET)
                    vis = np.hstack([__import__("cv2").cvtColor(inp_u8, __import__("cv2").COLOR_GRAY2BGR), heat])
                    __import__("cv2").imwrite(os.path.join(save_dir, f"vis_epoch_{epoch+1}.png"), vis)
            except Exception as e:
                print(f"Vis error: {e}")
            model.train()

        ckpt = {"teacher": model.teacher.state_dict(), "student": model.student.state_dict()}
        torch.save(ckpt, last_path)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"[{dataset_name}] epoch {epoch+1}/{args.epochs}  loss={avg_loss:.6f}")

    print(f"Done {dataset_name}. Checkpoint: {last_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=r"image_all")
    p.add_argument("--exp_name", type=str, default="image_data_02_27")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr_decay_epoch", type=int, default=80)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    eff_root = os.path.dirname(os.path.abspath(__file__))
    result_root = os.path.join(eff_root, "train-result", args.exp_name)

    for cam in ["CAM1", "CAM2", "CAM3", "CAM4"]:
        save_dir = os.path.join(result_root, cam)
        train_one_cam(args, cam, save_dir)


if __name__ == "__main__":
    main()
