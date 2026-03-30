# 可移植副本 — 运行依赖清单

## 主程序入口

- `ui/main.py`（PyQt5 主界面）

## 主界面同目录模块

- `ui/mainui.py`（UI 生成代码）`ui/para.py` `ui/report_change.py` `ui/report_center.py`
- `ui/cls_config.py` `ui/report.py` `ui/make_standard.py` `ui/theme.qss` `ui/repo_root_for_ui.py`
- `ui/requirements.txt`

## 由主界面启动的子进程

- `python -m app.online.detect_anomalies_online`（在线检测，工作目录为仓库根）
- `python -m app.report.gen_report_cls`（报告；由 `ui/report_change.py` 与 `ui/report_center.py` 调用）
- `ui/make_standard.py`（生成 `table.json`）
- C# 采集：`MULTICAM_DEMO_EXE` 或 `external/MultiCamDemo/MultiCamDemo.exe`（可选）

## 检测管线 Python 依赖（根目录）

- `function_bank.py` `util.py` `speed_monitor.py` `cls_anomalies.py` `cls_model.py`

## 算法包目录

- `patchcore_model/`（仅保留 PatchCore 推理链路）
- `det_model/`（仅保留 PatchCore 复用的预处理代码：`infer.py` / `prepare_dataset_det.py`）
- `cls_model/`（分类模型；用于报告/分类功能）

## 配置与数据

- `config/`（`config.yaml` `config0.yaml` `auth.yaml` `rptcfg.yaml` 等）
- `table.json`（允收矩阵，可由 make_standard 生成）
- `detect result/`（运行时输出，仓库内仅 `.gitkeep`）

## 再生本副本

- 在原工程根目录执行：`python build_portable_distribution.py`
