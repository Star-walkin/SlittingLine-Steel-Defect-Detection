"""
draem_model/model.py
DRAEM 双网络架构：
  RecNet  : 重建网络 — 输入含缺陷图像，重建出干净正常图（U-Net, 单通道）
  DiscNet : 判别网络 — 输入 [含缺陷图, 重建图] 2通道 → 逐像素异常概率（轻量 U-Net, Sigmoid输出）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── 共用组件 ──────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ──────────────────────────── RecNet（重建网络） ───────────────────────────

class RecNet(nn.Module):
    """
    单通道 U-Net，输入含缺陷图像（灰度，归一化 [-1,1]），输出重建干净图。
    Tanh 输出，对应 Normalize([0.5],[0.5]) 归一化。
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        self.enc0 = DoubleConv(1, ch[0])
        self.enc1 = Down(ch[0], ch[1])
        self.enc2 = Down(ch[1], ch[2])
        self.enc3 = Down(ch[2], ch[3])
        self.enc4 = Down(ch[3], ch[4])

        self.dec3 = Up(ch[4], ch[3])
        self.dec2 = Up(ch[3], ch[2])
        self.dec1 = Up(ch[2], ch[1])
        self.dec0 = Up(ch[1], ch[0])

        self.out = nn.Sequential(
            nn.Conv2d(ch[0], 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d = self.dec3(e4, e3)
        d = self.dec2(d, e2)
        d = self.dec1(d, e1)
        d = self.dec0(d, e0)
        return self.out(d)


# ──────────────────────────── DiscNet（判别网络） ─────────────────────────

class DiscNet(nn.Module):
    """
    轻量 U-Net，输入 2 通道（含缺陷图 + 重建图，均归一化），
    输出 1 通道像素级缺陷概率（Sigmoid，值域 [0,1]）。
    Sigmoid 值越接近 1 表示该像素越可能是缺陷。
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.enc0 = DoubleConv(2, ch[0])
        self.enc1 = Down(ch[0], ch[1])
        self.enc2 = Down(ch[1], ch[2])
        self.enc3 = Down(ch[2], ch[3])

        self.dec2 = Up(ch[3], ch[2])
        self.dec1 = Up(ch[2], ch[1])
        self.dec0 = Up(ch[1], ch[0])

        self.out = nn.Sequential(
            nn.Conv2d(ch[0], 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d = self.dec2(e3, e2)
        d = self.dec1(d, e1)
        d = self.dec0(d, e0)
        return self.out(d)


# ─────────────────────────────── 工厂函数 ────────────────────────────────

def build_draem(rec_base_ch: int = 32, disc_base_ch: int = 16):
    """构建 DRAEM 双网络，返回 (rec_net, disc_net)。"""
    return RecNet(base_ch=rec_base_ch), DiscNet(base_ch=disc_base_ch)


def save_draem(rec_net, disc_net, path: str):
    """将双网络打包保存到单个 .pth 文件。"""
    torch.save({"rec": rec_net.state_dict(), "disc": disc_net.state_dict()}, path)


def load_draem(path: str, device="cuda", rec_base_ch: int = 32, disc_base_ch: int = 16):
    """从单个 .pth 文件加载 DRAEM 双网络，返回 (rec_net, disc_net)。"""
    ckpt = torch.load(path, map_location=device)
    rec_net, disc_net = build_draem(rec_base_ch, disc_base_ch)
    rec_net.load_state_dict(ckpt["rec"])
    disc_net.load_state_dict(ckpt["disc"])
    rec_net.to(device).eval()
    disc_net.to(device).eval()
    return rec_net, disc_net
