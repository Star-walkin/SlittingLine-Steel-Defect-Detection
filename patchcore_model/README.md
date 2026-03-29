# PatchCore 模块说明

本目录用于训练和推理带钢缺陷的 PatchCore 模型，数据接口与 `det_model` 保持一致。

## 数据目录接口（与 det_model 相同）

训练读取目录：

`<data_root>/<exp_name>/<CAMx>/train/good/*.png`

例如：

`image_all/image_data_02_27/CAM1/train/good/*.png`

## 目录结构

- `train.py`：训练并保存每个相机的 PatchCore 记忆库
- `infer.py`：加载记忆库做图像推理并输出热力图
- `weights/`：模型权重与记忆库输出目录

## 训练示例

```bash
python patchcore_model/train.py --data_root image_all --exp_name image_data_02_27 --cams CAM1,CAM2,CAM3,CAM4
```

## 推理示例

```bash
python patchcore_model/infer.py --model_path patchcore_model/weights/image_data_02_27/CAM1/patchcore_memory.npz --input image_all/image_data_02_27/CAM1/train/good --output_dir patchcore_model/output/CAM1
```
