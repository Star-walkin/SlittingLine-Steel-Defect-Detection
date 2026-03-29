import os
import cv2
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from seg_model_NEW import UNet
from util import mean_smoothing
from msgms import MSGMSLoss


# =================== 数据读取：对齐训练集的单通道处理 ===================

class SimpleFolderDataset(Dataset):
    def __init__(self, root_dir, dataset_name, img_size=256, max_items=None):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.img_size = img_size

        paths = []
        for r, _, fs in os.walk(root_dir):
            for f in fs:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    paths.append(os.path.join(r, f))
        self.paths = sorted(paths)[:max_items] if max_items else sorted(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        # 【关键修改】读取为灰度图 (IMREAD_GRAYSCALE)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))

        # 归一化到 [-1, 1] 匹配训练集的 Tanh 输出
        img_f = img.astype(np.float32) / 255.0
        img_f = (img_f - 0.5) / 0.5

        # 增加通道维度 [1, H, W]
        img_t = torch.from_numpy(img_f).unsqueeze(0).contiguous()

        basename = os.path.splitext(os.path.basename(p))[0]
        return self.dataset_name, basename, img_t


def Get_testdataloader_ours(args, dataset_name, sub_folder):
    test_dir = os.path.join(args.data_root, dataset_name, "train", sub_folder)
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(args.data_root, dataset_name, sub_folder)

    if not os.path.exists(test_dir):
        return None

    ds = SimpleFolderDataset(test_dir, dataset_name, args.img_size, args.max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False)


# =================== 验证函数：对齐模型架构 ===================

def valid_img_updated(args, test_dataloader, model_name, save_test_dir, thres):
    # 初始化单通道模型
    model = UNet().cuda()
    state_dict = torch.load(model_name, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()

    msgm = MSGMSLoss()  # 请确保 MSGMSLoss 支持单通道或已内部处理

    for class_name, basename, img in test_dataloader:
        test_img = img.cuda()

        with torch.no_grad():
            rec_img = model(test_img)

            # 计算异常图
            amap = msgm(test_img, rec_img, as_loss=False)
            amap_ori = mean_smoothing(amap)

            # 后处理：转为 Numpy
            # 因为是单通道，squeeze 后变为 [H, W]
            test_img_np = (test_img[0].cpu().squeeze().numpy() * 0.5 + 0.5) * 255
            rec_img_np = (rec_img[0].cpu().squeeze().numpy() * 0.5 + 0.5) * 255
            amap_np = amap_ori[0].cpu().squeeze().numpy()

            # 师姐的边缘屏蔽
            amap_np[0:15, 0:20] = 0.001
            amap_np[0:15, 216:256] = 0.001

            # 可视化准备
            test_img_8u = test_img_np.astype(np.uint8)
            rec_img_8u = rec_img_np.astype(np.uint8)

            # 伪彩色热力图 (see_img_heatmap 逻辑)
            # 即使原图是灰度，heatmap 也是 BGR，所以统一转 BGR 拼接
            test_bgr = cv2.cvtColor(test_img_8u, cv2.COLOR_GRAY2BGR)
            rec_bgr = cv2.cvtColor(rec_img_8u, cv2.COLOR_GRAY2BGR)

            # 生成热力图
            norm_amap = (amap_np - amap_np.min()) / (amap_np.max() - amap_np.min() + 1e-8)
            heatmap = cv2.applyColorMap((norm_amap * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # 阈值分割图 (Binary Mask)
            mask = np.zeros_like(amap_np)
            mask[amap_np >= thres] = 255
            mask_bgr = cv2.merge([mask.astype(np.uint8)] * 3)

            # 叠加图
            overlay = cv2.addWeighted(test_bgr, 0.7, heatmap, 0.3, 0)

            # 拼接展示: 原图, 重建图, 热力图, 叠加图, 分割图
            combined = np.hstack([test_bgr, rec_bgr, heatmap, overlay, mask_bgr])

            save_path = os.path.join(save_test_dir, f"{basename[0]}.png")
            cv2.imwrite(save_path, combined)


# =================== 主程序 ===================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r'image_all')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)  # 验证时建议设为1方便逐张保存
    parser.add_argument('--max_items', type=int, default=20)
    args = parser.parse_args()

    all_types = ["CAM1", "CAM2", "CAM3", "CAM4"]
    model_root = r'.\train-result\image_data_01_24'  # 对应你训练代码的 save_dir
    save_root = r"seg_valid_results"

    for dataset_name in all_types:
        epoch = 291
        model_name = os.path.join(model_root, dataset_name, f"epoch_{epoch}.pth")  # 注意文件名格式

        for sub_f in ['good', 'bad']:
            save_dir = os.path.join(save_root, dataset_name, f"epoch_{epoch}", sub_f)
            os.makedirs(save_dir, exist_ok=True)

            loader = Get_testdataloader_ours(args, dataset_name, sub_f)
            if loader:
                print(f"Testing {dataset_name} - {sub_f}")
                valid_img_updated(args, loader, model_name, save_dir, thres=0.035)