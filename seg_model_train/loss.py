import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from pytorch_wavelets import DWTForward


# --- Wavelet Loss 修改版 ---
class WaveletLoss(nn.Module):
    def __init__(self, wave='haar', J=1):
        super(WaveletLoss, self).__init__()
        # 移除 .cuda()，保持模块设备无关性
        # 只实例化一个 DWTForward，因为它是无状态的变换
        self.DWT = DWTForward(J=J, mode='zero', wave=wave)
        self.mse_loss = nn.MSELoss()

    def forward(self, img1, img2):
        # 确保 DWT 模块和输入在同一设备上 (pytorch_wavelets 有时需要手动处理)
        # 通常 pytorch_wavelets 的 DWTForward 是 nn.Module，会随主模型移动

        # 获取低频(yl)和高频(yh)
        # yh 是一个列表，包含 J 层的高频系数
        yl1, yh1 = self.DWT(img1)
        yl2, yh2 = self.DWT(img2)

        # 计算低频部分的 MSE
        loss_yl = self.mse_loss(yl1, yl2)

        # 计算高频部分的 MSE
        # yh[0] 是第一层高频，形状通常为 (N, C, 3, H, W)
        # 我们需要累加所有层的高频损失
        loss_yh = 0
        for i in range(len(yh1)):
            loss_yh += self.mse_loss(yh1[i], yh2[i])

        return loss_yl + loss_yh


# --- SSIM Loss 辅助函数 ---
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_func(img1, img2, window, window_size, channel, size_average=True, val_range=None):
    # 动态确定数值范围
    if val_range is None:
        # 为了数值稳定性，增加 detach，不影响梯度
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    # 这里的 map 计算是 SSIM 的核心公式
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# --- SSIM Loss 修改版 ---
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        # 不在 __init__ 中生成具体的 window tensor，避免设备冲突
        # 使用 register_buffer 也可以，但动态生成更灵活适应不同通道数输入
        self.window = None

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # 检查是否需要重新生成 window (通道数变了，或者设备变了，或者数据类型变了)
        if (self.window is None or
                self.window.size(1) != channel or
                self.window.device != img1.device or
                self.window.dtype != img1.dtype):
            real_window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            # 将 window 注册为属性，但不作为模型参数（不会被优化器更新）
            self.window = real_window
            self.channel = channel

        return 1.0 - ssim_func(img1, img2, self.window, self.window_size, channel, self.size_average, self.val_range)


# --- 综合损失函数示例 (可选) ---
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.wavelet = WaveletLoss()
        self.ssim = SSIMLoss()

    def forward(self, input, target):
        loss_w = self.wavelet(input, target)
        loss_s = self.ssim(input, target)
        return self.alpha * loss_w + self.beta * loss_s


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 实例化 Loss (不需要在这里 .cuda())
    wave_criterion = WaveletLoss(wave='haar').to(device)
    ssim_criterion = SSIMLoss().to(device)

    # 模拟数据
    image1 = torch.randn(2, 3, 256, 256, dtype=torch.float).to(device)
    image2 = torch.randn(2, 3, 256, 256, dtype=torch.float).to(device)

    # 计算 Loss
    loss1 = wave_criterion(image1, image2)
    loss2 = ssim_criterion(image1, image2)

    print(f"Wavelet Loss: {loss1.item()}")
    print(f"SSIM Loss: {loss2.item()}")