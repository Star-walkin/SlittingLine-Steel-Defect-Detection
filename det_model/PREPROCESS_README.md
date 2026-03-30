# 训练 vs 离线检测：预处理一致性说明

## 对比结论（修改前）

| 环节           | 训练（data_prepare + dataset）     | 离线检测（detectoutline02 → infer）   |
|----------------|-------------------------------------|---------------------------------------|
| 数据来源       | img_raw_0228/cam*_filter（可能已 FFT） | 原图 → split_multi_strips → 条带      |
| 裁边           | dataset 内 crop 10px（左右）        | infer 内 crop 15px（左右）             |
| FFT 去纹       | 无（若用 filter 图则源文件已做）   | 有（apply_fft_deripple）              |
| 背景拍平       | 高斯模糊 -bg+128（dataset）         | 中值滤波 -bg+mean(bg)（infer）        |

**结论**：训练与推理的预处理不一致（拍平方式、裁边、FFT 是否再做），易导致推理时热力图整体偏高、误报多。

---

## 使用与推理一致的训练数据（推荐）

1. **生成数据（与离线检测相同预处理）**
   ```bash
   python det_model/prepare_dataset_det.py --raw_root D:\pycharm_project\steeldefect\img_raw_0228 --exp_name image_data_02_27 --out_root D:\pycharm_project\steeldefect\image_all
   ```
   会按 `split_multi_strips` 切带、分段，对每块做：**裁 15px → FFT 去纹 → 中值拍平**，写入 `out_root/exp_name/CAMx/train/good/`。

2. **训练时标明“已预处理”**
   ```bash
   python det_model/train.py --data_root D:\pycharm_project\steeldefect\image_all --exp_name image_data_02_27 --preprocessed
   ```
   加 `--preprocessed` 后，dataset 不再做 crop 10 与 flatten，只做 normal/anomaly 合成与 resize/ToTensor/Normalize，与推理输入分布一致。

---

## 文件说明

- **prepare_dataset_det.py**：原始图 → 与 infer 一致的预处理 → CAMx/train/good，供 SimpleAD 训练。
- **dataset.py**：`preprocessed=True` 时跳过 flatten 与 10px crop，直接使用 prepare_dataset_det 生成的数据。
- **train.py**：新增 `--preprocessed`，传给 `get_dataloader(..., preprocessed=True)`。
