# SlittingLine 钢带缺陷检测（PatchCore 运行版）

本项目用于分条线钢带表面缺陷在线检测与报告生成。软件界面为 **Windows 桌面程序（Python + PyQt5）**。

> 重要说明（面向操作人员）：你只需要会做两件事：\n+> 1) 打开软件；\n+> 2) 点击 **开始 / 停止**。\n+
---

## 一、最简单的使用方法（工人 30 秒上手）

### 1) 开机前检查（必须）

- **相机采集程序是否准备好**：需要 C# 程序 `MultiCamDemo.exe`（由工程师安装/提供）。\n+- **磁盘空间**：`detect result/` 会持续写入 JSON 与缺陷图片，空间不足会导致卡顿或异常。\n+
### 2) 启动软件（双击/命令行二选一）

#### 方法 A：命令行启动（推荐给工程师/维护人员）

在仓库根目录打开命令行，执行：

```bash
cd ui
python main.py
```

#### 方法 B：双击启动（可选）

如果工程师给你做了桌面快捷方式，就直接双击快捷方式启动。

### 3) 运行检测（操作员日常流程）

1. 软件打开后，确认界面显示正常。\n+2. 点击 **开始**：\n+   - 软件会自动启动“在线检测进程”（Python），并尝试启动“相机采集程序”（C#）。\n+3. 正常运行时：\n+   - 界面会持续刷新幅宽曲线、缺陷点位等。\n+4. 需要停机/换卷/异常时：点击 **停止**。\n+
---

## 二、工程师安装与首次配置（必须做一次）

### 1) Python 环境与依赖

建议使用与产线一致的 Python 环境。\n+安装依赖（示例）：

```bash
pip install -r ui/requirements.txt
```

### 2) 关键目录说明（不要随便改名）

- `ui/`：主界面程序（入口 `ui/main.py`）\n+- `app/online/`：在线检测入口（界面点击“开始”后运行 `python -m app.online.detect_anomalies_online`）\n+- `app/report/`：报告生成入口（界面内生成报告时运行 `python -m app.report.gen_report_cls`）\n+- `models/patchcore_model/`：PatchCore 推理代码与权重目录（重点）\n+- `config/`：配置文件\n+- `detect result/`：运行输出（检测结果、JSON、缺陷图片、报告等）\n+- `external/`：外部程序目录（建议放相机采集程序）\n+
### 3) 相机采集程序（C#）位置

软件会尝试启动 C# 采集程序。你可以用两种方式指定：

- **方式 1（推荐）**：把程序放到\n+  - `external/MultiCamDemo/MultiCamDemo.exe`\n+- **方式 2**：使用环境变量\n+  - `MULTICAM_DEMO_EXE`：指向 `MultiCamDemo.exe`\n+  - `MULTICAM_DEMO_CWD`：程序工作目录\n+
### 4) 配置文件（最重要）

#### `config/config0.yaml`（产线当前卷/条带配置）

常用字段：\n+- `conduct_id`：生产卡号（会影响输出目录）\n+- `strip_count`：条带数量（1~4）\n+- `fukuan_list` 或 `fukuan_1..4`：每条带的设定幅宽（mm）\n+
#### `config/config.yaml`（检测参数）

本版本只保留 **PatchCore 运行链路**。\n+你需要确认：\n+- `patchcore_weights_root`：PatchCore 权重根目录名（在 `models/patchcore_model/` 下）\n+- 各相机 `camX_*` 的 PatchCore 参数（阈值/裁边/背景核等）\n+
---

## 三、PatchCore 权重放置规则（非常重要）

在线检测会按如下规则寻找权重（以 CAM1 为例）：

- `models/patchcore_model/<patchcore_weights_root>/image_data_patchcore_0228/CAM1/patchcore_memory.npz`

如果提示“未找到 PatchCore 权重文件”，说明权重目录不对或没拷贝过来。\n+请工程师把训练得到的 `patchcore_memory.npz` 放到上述路径。\n+
---

## 四、报告生成与查看

软件内有“报告中心/报告打印”等入口。\n+报告生成会读取：\n+- `detect result/<日期>/<卡号>/...` 下的检测结果\n+- `config/rptcfg.yaml`（类别名称、允收矩阵、打印项等）\n+- `table.json`（允收矩阵缓存；可由 `ui/make_standard.py` 生成）\n+
---

## 五、常见故障排查（按这个顺序看）

### 1) 点了“开始”没反应

- 是否安装了 Python 依赖（`pip install -r ui/requirements.txt`）\n+- 是否有权限写入 `detect result/`（建议放在非系统盘/有写权限目录）\n+- 查看控制台输出（工程师用）\n+
### 2) 提示找不到 PatchCore 权重

- 按“第三部分：权重放置规则”检查 `patchcore_memory.npz` 路径是否存在。\n+
### 3) 相机无图/无数据

- C# 采集程序是否能独立运行\n+- 是否设置了 `MULTICAM_DEMO_EXE` / `MULTICAM_DEMO_CWD`\n+- 相机/网线/交换机是否正常\n+
### 4) 软件卡顿/变慢

- 检查磁盘剩余空间\n+- `config/config.yaml` 的 `debug_io` 必须为 `false`（生产环境）\n+- `detect result/` 是否过大（可按日期归档到移动硬盘/服务器）\n+
---

## 六、维护与备份建议（工程师）

- 定期把 `detect result/` 按日期归档（例如每班/每天压缩）\n+- 版本更新建议：先在测试机验证，再上产线\n+- GitHub 仓库不包含大文件（权重/训练结果/检测结果），需要单独备份\n+
---

## 七、仓库与版本管理

GitHub：`https://github.com/Star-walkin/SlittingLine-Steel-Defect-Detection.git`\n+
