"""
draem_model/train.py
DRAEM 双网络联合训练：
  - RecNet  : MSE 重建损失
  - DiscNet : BCE + Dice 分割损失（监督缺陷 mask）
训练完成后保存到 draem_model/train-result/<exp_name>/CAMx/last.pth
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from draem_model.model import build_draem, save_draem
from draem_model.dataset import get_dataloader


# ────────────────────────────── 损失函数 ─────────────────────────────────

def dice_loss(pred, target, eps=1e-6):
    """Soft Dice Loss。pred/target: [B,1,H,W]，值域 [0,1]。"""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    return 1.0 - (2.0 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)


def focal_bce(pred, target, gamma=2.0, alpha=0.75, eps=1e-6):
    """
    Focal BCE（解决正负样本极不平衡：缺陷像素仅占 1-5%）。
    alpha 控制正样本权重，gamma 控制难例聚焦程度。
    """
    pred_c = pred.clamp(eps, 1 - eps)
    ce = -(target * torch.log(pred_c) + (1 - target) * torch.log(1 - pred_c))
    pt = torch.where(target == 1, pred_c, 1 - pred_c)
    focal_w = alpha * target + (1 - alpha) * (1 - target)
    loss = focal_w * (1 - pt) ** gamma * ce
    return loss.mean()


def disc_loss(pred, target):
    """DiscNet 损失 = 0.5*Focal BCE + 0.5*Dice，数值更稳定。"""
    fl = focal_bce(pred, target)
    dc = dice_loss(pred, target).mean()
    return 0.5 * (fl + dc)


# ────────────────────────────── 工具函数 ─────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────── 单相机训练 ──────────────────────────────────

def train_one_cam(args, dataset_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    data_root = getattr(args, "data_root", r"image_all")
    exp_name = getattr(args, "exp_name", "image_data_02_27")
    data_dir = os.path.join(data_root, exp_name, dataset_name)

    loader = get_dataloader(data_dir, img_size=args.img_size, batch_size=args.batch_size)

    rec_net, disc_net = build_draem(rec_base_ch=32, disc_base_ch=16)
    rec_net.to(device)
    disc_net.to(device)

    optimizer = torch.optim.Adam(
        list(rec_net.parameters()) + list(disc_net.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    mse = nn.MSELoss()

    log_path = os.path.join(save_dir, "loss.log")
    last_path = os.path.join(save_dir, "last.pth")

    print(f"\n==== DRAEM Train {dataset_name} ====")
    print(f"device: {device}  |  data_dir: {data_dir}")

    for epoch in range(args.epochs):
        rec_net.train()
        disc_net.train()

        epoch_rec_loss = 0.0
        epoch_disc_loss = 0.0
        n_batch = 0
        t0 = time.perf_counter()

        pbar = tqdm(loader, desc=f"{dataset_name} E{epoch+1}/{args.epochs}", leave=False)
        for normal_img, anomalous_img, mask in pbar:
            normal_img = normal_img.to(device, non_blocking=True)
            anomalous_img = anomalous_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # 输入 NaN/Inf 防护，避免脏数据直接污染网络
            normal_img = torch.nan_to_num(normal_img, nan=0.0, posinf=1.0, neginf=0.0)
            anomalous_img = torch.nan_to_num(anomalous_img, nan=0.0, posinf=1.0, neginf=0.0)
            mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)

            # 只在前向阶段使用混合精度，损失在 FP32 中计算，更稳定
            with autocast(enabled=(device.type == "cuda")):
                rec = rec_net(anomalous_img)
                disc_in = torch.cat([anomalous_img, rec.detach()], dim=1)
                disc_out = disc_net(disc_in)

            # 损失计算统一用 float32，避免 FP16 下 log / 比例引起 NaN
            rec_f = rec.float()
            normal_f = normal_img.float()
            disc_out_f = disc_out.float()
            mask_f = mask.float()

            rl = mse(normal_f, rec_f)
            dl = disc_loss(disc_out_f, mask_f)
            loss = rl + dl

            # 如出现非有限 loss（NaN/Inf），跳过该 batch，保护权重
            if not torch.isfinite(loss):
                try:
                    rec_val = float(rl.detach().cpu())
                except Exception:
                    rec_val = float("nan")
                try:
                    disc_val = float(dl.detach().cpu())
                except Exception:
                    disc_val = float("nan")
                print(f"[WARN] non-finite loss detected, skip batch: rec={rec_val:.4f}, disc={disc_val:.4f}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(rec_net.parameters()) + list(disc_net.parameters()), max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_rec_loss += rl.item()
            epoch_disc_loss += dl.item()
            n_batch += 1
            pbar.set_postfix(rec=f"{rl.item():.4f}", disc=f"{dl.item():.4f}")

        scheduler.step()
        elapsed = time.perf_counter() - t0
        avg_rec = epoch_rec_loss / max(1, n_batch)
        avg_disc = epoch_disc_loss / max(1, n_batch)
        lr_now = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"E{epoch+1:04d}  rec={avg_rec:.5f}  disc={avg_disc:.5f}  "
                f"lr={lr_now:.2e}  t={elapsed:.1f}s\n"
            )

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            save_draem(rec_net, disc_net, ckpt_path)

            # ── 可视化：输入 | 重建 | 判别图 | mask GT
            try:
                rec_net.eval()
                disc_net.eval()
                import cv2 as cv
                with torch.no_grad():
                    rec_vis = rec_net(anomalous_img[:1])
                    di = torch.cat([anomalous_img[:1], rec_vis], dim=1)
                    disc_vis = disc_net(di)

                def _to_u8(t):
                    arr = (t[0, 0].cpu().float() * 0.5 + 0.5).clamp(0, 1).numpy()
                    return (arr * 255).astype(np.uint8)

                vis_inp = _to_u8(anomalous_img)
                vis_rec = _to_u8(rec_vis)
                vis_disc = (disc_vis[0, 0].cpu().float().numpy() * 255).astype(np.uint8)
                vis_disc = cv.applyColorMap(vis_disc, cv.COLORMAP_JET)
                vis_disc_gray = cv.cvtColor(vis_disc, cv.COLOR_BGR2GRAY)
                vis_mask = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)

                vis_row = np.hstack([vis_inp, vis_rec, vis_disc_gray, vis_mask])
                label_h = 24
                banner = np.zeros((label_h, vis_row.shape[1]), dtype=np.uint8) + 240
                font = cv.FONT_HERSHEY_SIMPLEX
                w = vis_inp.shape[1]
                for i, txt in enumerate(["Input", "Reconstruction", "DiscNet", "GT Mask"]):
                    cv.putText(banner, txt, (w * i + 5, 18), font, 0.5, 0, 1, cv.LINE_AA)
                vis_full = np.vstack([banner, vis_row])
                cv.imwrite(os.path.join(save_dir, f"vis_epoch_{epoch+1}.png"), vis_full)
            except Exception as e:
                print(f"Vis error: {e}")
            rec_net.train()
            disc_net.train()

        save_draem(rec_net, disc_net, last_path)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(
                f"[{dataset_name}] E{epoch+1}/{args.epochs}  "
                f"rec={avg_rec:.5f}  disc={avg_disc:.5f}  lr={lr_now:.2e}"
            )

    print(f"Done {dataset_name}. Saved: {last_path}")


# ─────────────────────────────── 入口 ────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                   default=r"image_all")
    p.add_argument("--exp_name", type=str, default="image_data_02_27")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cams", type=str, default="CAM1,CAM2,CAM3,CAM4",
                   help="逗号分隔的相机列表")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    result_root = os.path.join(here, "train-result", args.exp_name)

    for cam in args.cams.split(","):
        cam = cam.strip()
        save_dir = os.path.join(result_root, cam)
        train_one_cam(args, cam, save_dir)


if __name__ == "__main__":
    main()
