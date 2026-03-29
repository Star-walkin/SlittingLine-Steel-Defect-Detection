# scar_anomaly_safe.py / 改进版
import numpy as np
from random import randint, uniform
from skimage.util import random_noise
from skimage.draw import ellipse_perimeter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import cv2, math

def add_stain(img, size='0.1-6', color='150-255', irregularity=0.3, blur=0.01):
    """
    给图像 img 添加随机“污点”，返回 (合成图, mask)
    - img: BGR ndarray
    - size: 百分比范围，例如 '0.1-6' -> 相对于图像尺寸
    - color: 灰度范围，例如 '150-255'
    - irregularity: 椭圆扰动程度
    - blur: 高斯模糊强度
    """
    row, col, ch = img.shape

    # 随机颜色
    if '-' in str(color):
        min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
        color_val = randint(min_color, max_color)
    else:
        color_val = int(color)

    # 随机椭圆尺寸
    min_range, max_range = float(size.split('-')[0]), float(size.split('-')[1])
    a = randint(max(1, int(min_range/100.*col)), max(1, int(max_range/100.*col)))
    b = randint(max(1, int(min_range/100.*row)), max(1, int(max_range/100.*row)))

    # 随机中心
    cx = randint(a, max(a, col - a))
    cy = randint(b, max(b, row - b))
    rotation = uniform(0, 2*np.pi)

    # 安全生成椭圆周长点
    try:
        xy = ellipse_perimeter(cy, cx, a, b, rotation)
        if xy is None or len(xy) != 2:
            raise ValueError("ellipse_perimeter failed")
        x, y = xy
        if not hasattr(x, '__len__') or not hasattr(y, '__len__') or len(x) < 3:
            raise ValueError("ellipse points too few")
    except:
        # 如果失败，用单点代替
        x = np.array([cx, cx+1, cx+1])
        y = np.array([cy, cy, cy+1])

    # 构建轮廓
    contour = np.array([[i, j] for i, j in zip(x, y)])
    if irregularity > 0:
        contour = perturbate_ellipse(contour, cx, cy, (a+b)/2, irregularity)
    if contour.shape[0] < 3:
        # 至少3个点才能画
        return img.copy(), np.zeros((row, col), dtype=np.uint8)

    # mask
    mask = np.zeros((row, col), dtype=np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, 255, -1)
    if blur != 0:
        mask = gaussian_filter(mask, max(a, b) * blur)
        if mask.max() > 0:
            mask = (mask / mask.max() * 255).astype(np.uint8)

    # 合成图像
    rgb_mask = np.dstack([mask]*3)
    not_modified = np.subtract(np.ones(img.shape), rgb_mask/255.0)
    stain = 255 * random_noise(np.zeros(img.shape), mode='gaussian', mean=color_val/255., var=0.05/255.)
    result = np.add(np.multiply(img, not_modified), np.multiply(stain, rgb_mask/255.0))

    return result.astype(np.uint8), mask


def perturbate_ellipse(contour, cx, cy, diag, irregularity):
    if len(contour) < 20:
        pts = contour
    else:
        pts = contour[0::max(1, int(len(contour)/20))]
    for idx, pt in enumerate(pts):
        pts[idx] = [pt[0] + randint(-int(diag*irregularity), int(diag*irregularity)),
                    pt[1] + randint(-int(diag*irregularity), int(diag*irregularity))]
    pts = sorted(pts, key=lambda p: clockwiseangle(p, cx, cy))
    pts.append([pts[0][0], pts[0][1]])
    i = np.arange(len(pts))
    interp_i = np.linspace(0, i.max(), max(1, int(10*i.max())))
    xi = interp1d(i, np.array(pts)[:,0], kind='cubic')(interp_i)
    yi = interp1d(i, np.array(pts)[:,1], kind='cubic')(interp_i)
    return np.array([[int(i), int(j)] for i,j in zip(yi, xi)])


def clockwiseangle(point, cx, cy):
    refvec = [0,1]
    vector = [point[0]-cy, point[1]-cx]
    norm = math.hypot(vector[0], vector[1])
    if norm==0: return -math.pi
    normalized = [vector[0]/norm, vector[1]/norm]
    dotprod = normalized[0]*refvec[0]+normalized[1]*refvec[1]
    diffprod = refvec[1]*normalized[0]-refvec[0]*normalized[1]
    angle = math.atan2(diffprod, dotprod)
    if angle < 0: angle += 2*math.pi
    return angle

