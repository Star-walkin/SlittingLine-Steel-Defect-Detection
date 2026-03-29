"""
det_model/model.py
SimpleAD 风格编码器-解码器，单通道输入/输出，L2 重建用于异常检测。
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect"),
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = x + skip
        return self.conv(x)


class SimpleADNet(nn.Module):
    """单通道 U-Net，输出 Tanh[-1,1]，与训练时 Normalize([0.5],[0.5]) 对应。"""

    def __init__(self, base_ch=32):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        self.proj = nn.Conv2d(1, ch[0], 1, stride=1, padding=0)
        self.down1 = DownBlock(ch[0], ch[1])
        self.down2 = DownBlock(ch[1], ch[2])
        self.down3 = DownBlock(ch[2], ch[3])
        self.down4 = DownBlock(ch[3], ch[4])

        self.up1 = UpBlock(ch[4], ch[3])
        self.up2 = UpBlock(ch[3], ch[2])
        self.up3 = UpBlock(ch[2], ch[1])
        self.up4 = UpBlock(ch[1], ch[0])

        self.out = nn.Sequential(
            nn.Conv2d(ch[0], 1, 1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        x0 = self.proj(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out(x)


def build_model(base_ch=32):
    """构建单通道 SimpleAD 模型，用于训练与推理。"""
    return SimpleADNet(base_ch=base_ch)
