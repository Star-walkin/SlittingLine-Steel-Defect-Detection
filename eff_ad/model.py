"""
eff_ad/model.py
轻量级 Student-Teacher 异常检测：Teacher 冻结提取特征，Student 在正常样本上学习拟合。
推理时异常图 = Student 与 Teacher 特征的空间 L2 差异。
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """4 层下采样 CNN，256x256 -> 32x32 x C，单通道输入。"""

    def __init__(self, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.encoder(x)


class StudentTeacher(nn.Module):
    """
    Teacher 冻结；Student 可训练。
    forward_train(x): 返回 (student_feat, teacher_feat) 用于训练 loss = L2(student_feat, teacher_feat)。
    forward_anomap(x): 返回 anomaly map [B, 1, H, W] 用于推理。
    """

    def __init__(self, feat_channels=128):
        super().__init__()
        self.teacher = SmallCNN(out_channels=feat_channels)
        self.student = SmallCNN(out_channels=feat_channels)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.feat_channels = feat_channels

    def init_teacher(self):
        """Teacher 随机初始化并冻结（无 ImageNet）。"""
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward_train(self, x):
        with torch.no_grad():
            t_feat = self.teacher(x)
        s_feat = self.student(x)
        return s_feat, t_feat

    def forward_anomap(self, x):
        with torch.no_grad():
            t_feat = self.teacher(x)
        s_feat = self.student(x)
        diff = (s_feat - t_feat) ** 2
        amap = diff.sum(dim=1, keepdim=True)
        return amap


def build_eff_ad(feat_channels=128):
    """构建 Student-Teacher 模型。"""
    m = StudentTeacher(feat_channels=feat_channels)
    m.init_teacher()
    return m
