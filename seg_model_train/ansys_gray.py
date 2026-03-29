import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ========= 配置 =========
IMG_PATH = r"img_raw\cam1\cam1_000000.png"
SMOOTH_KERNEL = 51     # 灰度平滑窗口（奇数）
DERIV_SMOOTH = 21      # 导数平滑窗口（奇数，建议小一点）

# ========= 读图 =========
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {IMG_PATH}")

H, W = img.shape
print(f"图像尺寸: H={H}, W={W}")

# ========= 计算横向灰度剖面 =========
profile = img.mean(axis=0)   # shape: (W,)

# ========= 1D 平滑 =========
def smooth_1d(x, k):
    k = int(k) | 1
    return cv2.GaussianBlur(
        x.astype(np.float32).reshape(1, -1),
        (k, 1),
        0
    ).ravel()

profile_smooth = smooth_1d(profile, SMOOTH_KERNEL)

# ========= 计算一阶导数 =========
# np.gradient 比 np.diff 更适合可视化（长度不变）
deriv = np.gradient(profile_smooth)

# 再平滑一次导数，抑制噪声尖刺
deriv_smooth = smooth_1d(deriv, DERIV_SMOOTH)

# ========= 绘图 =========
x = np.arange(W)

fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

# --- 上图：灰度趋势 ---
axes[0].plot(x, profile, color="gray", alpha=0.35, label="Original column mean")
axes[0].plot(x, profile_smooth, color="red", linewidth=2, label="Smooth grayscale")
axes[0].set_ylabel("Grayscale value")
axes[0].set_title("Lateral grayscale profile")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- 下图：灰度导数 ---
axes[1].plot(x, deriv, color="steelblue", alpha=0.3, label="Raw derivative")
axes[1].plot(x, deriv_smooth, color="blue", linewidth=2, label="Smooth derivative")
axes[1].axhline(0, color="black", linewidth=1)

axes[1].set_xlabel("x (pixel column)")
axes[1].set_ylabel("d(grayscale)/dx")
axes[1].set_title("Grayscale derivative (change rate)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
