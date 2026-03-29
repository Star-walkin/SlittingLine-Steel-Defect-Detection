import os
import time
import random
import numpy as np
import cv2  # [新增] 用于保存可视化图

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from options2 import TrainOptions
from seg_model_NEW import UNet
from seg_dataset import Get_traindataloader
from loss import SSIMLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_lr(optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def train_one_cam(args, dataset_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 性能相关 =====
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    # ===== DataLoader =====
    # 确保 seg_dataset.py 里返回的是 (B, 1, H, W)
    train_loader = Get_traindataloader(args, dataset_name)

    # ===== Loss / Model / Optim =====
    ssim_loss = SSIMLoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    # [注意] 请确保 seg_model_NEW.py 里 UNet 默认是单通道 (in=1, out=1)
    # 否则这里要写成 UNet(in_channels=1, ...)
    model = UNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    def get_epoch_lr(ep: int):
        if ep < args.lr_decay_epoch:
            return args.lr
        else:
            return args.lr * 0.1

    log_path = os.path.join(save_dir, "loss.log")
    last_path = os.path.join(save_dir, "last.pth")

    print(f"\n==== Train {dataset_name} ====")
    print(f"save_dir: {save_dir}")
    print(f"device: {device}")

    for epoch in range(args.epochs):
        model.train()
        lr_now = get_epoch_lr(epoch)
        set_lr(optimizer, lr_now)

        epoch_loss_sum = 0.0
        epoch_batches = 0
        t0 = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"{dataset_name} | Epoch {epoch + 1}/{args.epochs} | lr={lr_now:.1e}",
                    leave=False)
        PRINT_EVERY = 10

        for step, (normal_image, anomaly_image) in enumerate(pbar, start=1):
            normal_image = normal_image.to(device, non_blocking=True)
            anomaly_image = anomaly_image.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                rec_img = model(anomaly_image)

                # 计算 Loss
                mse = mse_loss(normal_image, rec_img)
                ssim = ssim_loss(normal_image, rec_img)
                loss = mse + ssim  # 根据需要调整权重，例如 mse + 1.0*ssim

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.detach().cpu().item())
            epoch_loss_sum += loss_val
            epoch_batches += 1

            if step % PRINT_EVERY == 0:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

        epoch_time = time.perf_counter() - t0
        epoch_loss_avg = epoch_loss_sum / max(1, epoch_batches)

        # 写日志
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{dataset_name}] Epoch {epoch + 1}/{args.epochs}  lr={lr_now:.2e}  loss_avg={epoch_loss_avg:.6f}  time={epoch_time:.2f}s\n")

        # 保存权重 & 可视化监控
        if epoch % args.save_interval == 0:
            # 1. 保存权重
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

            # 2. [关键] 保存一张对比图，确认重构是否正常 (单通道灰度)
            try:
                model.eval()
                with torch.no_grad():
                    # 随便取 batch 里的一张
                    vis_in = (normal_image[0].cpu().float() * 0.5 + 0.5).clamp(0, 1) * 255
                    vis_out = (rec_img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1) * 255

                    vis_in_np = vis_in.squeeze().numpy().astype(np.uint8)
                    vis_out_np = vis_out.squeeze().numpy().astype(np.uint8)

                    # 左右拼接
                    debug_img = np.hstack([vis_in_np, vis_out_np])
                    cv2.imwrite(os.path.join(save_dir, f"vis_epoch_{epoch + 1}.png"), debug_img)
                model.train()
            except Exception as e:
                print(f"Vis error: {e}")

        # 保存 last
        torch.save(model.state_dict(), last_path)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"[{dataset_name}] epoch {epoch + 1}/{args.epochs}  loss={epoch_loss_avg:.6f}")

    print(f"Done {dataset_name}. last checkpoint: {last_path}")


if __name__ == "__main__":
    args = TrainOptions().parse()

    if not hasattr(args, "seed"):
        args.seed = 42

    # 请确认这个路径下有 CAM1/train/good, CAM2/train/good 等结构
    args.data_root = r"image_all"
    args.exp_name = "image_data_01_24"

    all_types = ["CAM1", "CAM2", "CAM3", "CAM4"]
    for dataset_name in all_types:
        save_dir = rf".\train-result\{args.exp_name}\{dataset_name}"
        train_one_cam(args, dataset_name, save_dir)