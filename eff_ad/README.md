# eff_ad：轻量级 Student-Teacher 带钢缺陷检测

## 模型说明
- **思路**：类 EfficientAD，仅用正常样本训练。Teacher 为冻结小 CNN，Student 学习拟合 Teacher 特征；异常 = Student 与 Teacher 特征的空间 L2 差异。
- **特点**：无需 ImageNet 预训练，单通道灰度输入，推理快，与现有切带 + 位置计算流程兼容。

## 目录结构
- `model.py`：SmallCNN + StudentTeacher
- `dataset.py`：NormalOnlyDataset（train/good + 背景拍平）
- `train.py`：按 CAM1–CAM4 分别训练
- `infer.py`：EffADDetector，接口与 detectoutline02 一致

## 1. 训练（需先有 image_data_02_27）
确保存在：`<data_root>/image_data_02_27/CAM1/train/good/*.png`（CAM2/CAM3/CAM4 同理）。  
若没有，请先运行 `seg_model_train/data_prepare.py` 生成。

```bash
# 在项目根目录
python -m eff_ad.train --data_root image_all --exp_name image_data_02_27 --epochs 200 --batch_size 8 --save_interval 10
```

权重与可视化保存到：`eff_ad/train-result/image_data_02_27/CAMx/last.pth` 与 `vis_epoch_*.png`。

## 2. 离线检测（bad_test）
训练完成后：

```bash
python detectoutline_effad.py
```

- 输入：`bad_test/cam1_bad`、`cam2_bad`、`cam3_bad`、`cam4_bad`
- 结果：`detect result/<日期>/<conduct_id>_EFFAD/1/`、`2/`、`3/`、`4/`（含 `debug_visuals`、`defect_images` 等）

## 3. 配置
- `config/config.yaml` 中可选配置 `cam1_eff_ad_ckpt`、`cam2_eff_ad_ckpt` 等指向各相机 `last.pth`；未配置时默认使用 `eff_ad/train-result/image_data_02_27/CAMx/last.pth`。
- 检测阈值使用现有 `cam*_seg_anomaly_thres`，可根据 bad_test 结果在 config 中微调。
