import time
import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
import yaml
from cls_model import ProjectionNet
import torch
from pathlib import Path
from function_bank import Anomaly_info_List,find_folders_with_id
import json
import torchvision.transforms as transforms
import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import json

class ImageDataset(Dataset):
    """ dataset name."""
    def __init__(self, data_dir,  img_size=128, ):

        self.img_size=img_size
        self.transform_train = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225]),
                                                   ])

        self.images = sorted(glob.glob(data_dir+ '/*.*'))
        print("loading images")


    def __getitem__(self, idx):
        #读取正常图像
        image_path = self.images[idx]
        filename = os.path.basename(image_path)[:-4]
        #x, y, area = filename.split("_")
        #area=area[:-4]
        try:
            image=Image.open(image_path).convert("RGB")
            image = self.transform_train(image)
        except Exception as e:
            # 捕获错误并跳过当前文件
            print(f"无法打开文件 {image_path}: {e}")
            #image=None

        return filename,image

    def __len__(self):
        return len(self.images)

def cls_anomalies(model_path, result_all_path, camrea_id, multi_strip=False):
    """
    缺陷分类函数（增强版）
    支持单带钢与多带钢模式，接口与原版完全兼容。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Step 1. 自动检测多带钢路径 ===
    cam_dir = os.path.join(result_all_path, str(camrea_id))
    if not os.path.exists(cam_dir):
        print(f"[警告] 未找到 {cam_dir}，跳过分类。")
        return

    # 若存在 strip_1 等文件夹，自动启用多带钢模式
    strip_paths = [os.path.join(cam_dir, d) for d in os.listdir(cam_dir)
                   if d.startswith("strip_") and os.path.isdir(os.path.join(cam_dir, d))]
    if len(strip_paths) > 0:
        multi_strip = True

    # 若无 strip 文件夹，保持原逻辑
    if not multi_strip:
        strip_paths = [cam_dir]

    # === Step 2. 加载模型 ===
    # torchvision 新版对 pretrained 参数会产生弃用警告，这里统一忽略，避免报告中心终端刷屏
    warnings.filterwarnings(
        "ignore",
        message=r".*parameter 'pretrained' is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Arguments other than a weight enum.*for 'weights' are deprecated.*",
        category=UserWarning,
    )
    head_layers = [512] * 1 + [128]
    weights = torch.load(model_path, map_location=device)
    # raw = torch.load(model_path, map_location="cpu")
    # print("TOP KEYS:", list(raw.keys())[:50] if isinstance(raw, dict) else type(raw))
    # # 如果 raw 是 dict 且里面有 state_dict/model，就打印里面的 keys
    # sd = raw.get("state_dict") or raw.get("model") or raw
    # if isinstance(sd, dict):
    #     print("STATE_DICT sample:", list(sd.keys())[:50])

    classes = weights["out.weight"].shape[0]
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    # === Step 2.5 读取“模型输出->报告类别ID”的对齐映射（可选）===
    # 用于：模型类别顺序与 rptcfg.class_labels 顺序不一致时，仍能写出正确的 1-based 类别编号。
    remap_modelidx_to_rptid = None
    try:
        runtime_state_path = os.path.join(os.path.dirname(__file__), "config", "runtime_state.json")
        if os.path.exists(runtime_state_path):
            with open(runtime_state_path, "r", encoding="utf-8") as f:
                st = json.load(f) or {}
            if str(st.get("cls_model_path", "") or "").strip() == str(model_path or "").strip():
                rr = st.get("cls_label_remap_modelidx_to_rptid_1based")
                if isinstance(rr, list) and len(rr) == int(classes):
                    remap_modelidx_to_rptid = [int(x) for x in rr]
    except Exception:
        remap_modelidx_to_rptid = None

    # === Step 3. 遍历每条带钢进行分类 ===
    for strip_path in strip_paths:
        defect_dir = os.path.join(strip_path, "defect_images")
        if not os.path.exists(defect_dir):
            print(f"[提示] {strip_path} 下未检测到 defect_images，跳过。")
            continue

        test_dataloader = DataLoader(
            ImageDataset(defect_dir),
            batch_size=64, shuffle=False, num_workers=1, drop_last=False
        )

        if len(test_dataloader) == 0:
            print(f"[提示] {defect_dir} 内无图像，跳过。")
            continue

        print(f"开始分类：{defect_dir}")

        logits = []
        coords_list = []

        with torch.no_grad():
            for idx, (filename, img) in enumerate(test_dataloader):
                img = img.to(device)
                logit = model(img)
                logit = torch.softmax(logit, axis=1)
                logits.extend(logit.cpu())
                coords_list.extend(filename)

        logits = torch.stack(logits)
        predict_cla = torch.argmax(logits, axis=1).tolist()
        # 报告与 rptcfg 使用 1-based 类别编号（1..N）。
        # 默认：模型 argmax 为 0-based，写盘前 +1；
        # 若存在 remap（模型类别顺序与 rptcfg 不同），则按 remap 把 0-based 映射到 rptcfg 的 1-based id。
        if isinstance(remap_modelidx_to_rptid, list) and remap_modelidx_to_rptid:
            predict_cla_report = [
                int(remap_modelidx_to_rptid[int(c)]) if 0 <= int(c) < len(remap_modelidx_to_rptid) else int(c) + 1
                for c in predict_cla
            ]
        else:
            predict_cla_report = [int(c) + 1 for c in predict_cla]
        anomalies_info = [[f"{a}_{b}" for a, b in zip(predict_cla_report, coords_list)]]

        # === Step 4. 保存分类结果 ===
        result_cls_path = strip_path
        anomaly_info_process = Anomaly_info_List(filepath=result_cls_path)
        anomaly_info_process.save_anomaly_info("anomaly_info_result.json", anomalies_info)

        print(f"分类完成：{strip_path}")

    print(f"相机 {camrea_id} 分类已全部完成。")





if __name__ == '__main__':
    model_path=os.path.join(_REPO_ROOT, "cls_model", "cls_checkpoint.pth")
    with open('D:\dahenggrab\mainui/config0.yaml', 'r', encoding='utf-8') as f:
        config0 = yaml.safe_load(f)
    conduct_id = config0["conduct_id"]
    result_all_path = find_folders_with_id(base_path="D:\\detect result", product_id=conduct_id)[-1]+"/"
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    cls_model_path=config["cls_model_path"]
    camrea_id_up_cls = config["camrea_id_up_cls"]
    camrea_id_down_cls= config["camrea_id_down_cls"]
    cls_anomalies(model_path,result_all_path,camrea_id_up_cls)
    print("上表面缺陷分类已完成.")
    cls_anomalies(model_path, result_all_path, camrea_id_down_cls)
    print("下表面缺陷分类已完成.")