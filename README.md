# SlittingLine 钢带缺陷检测（PatchCore 运行版）

本项目用于分条线钢带表面缺陷在线检测与报告生成。软件界面为 **Windows 桌面程序（Python + PyQt5）**。

> 重要说明（面向操作人员）：你只需要会做两件事：
> 1) 打开软件；
> 2) 点击 **开始 / 停止**。

---

## 一、最简单的使用方法（工人 30 秒上手）

### 1) 开机前检查（必须）

- **相机采集程序是否准备好**：需要 C# 程序 `MultiCamDemo.exe`（由工程师安装/提供）。
- **磁盘空间**：`detect result/` 会持续写入 JSON 与缺陷图片，空间不足会导致卡顿或异常。

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

1. 软件打开后，确认界面显示正常。
2. 点击 **开始**：
   - 软件会自动启动“在线检测进程”（Python），并尝试启动“相机采集程序”（C#）。
3. 正常运行时：
   - 界面会持续刷新幅宽曲线、缺陷点位等。
4. 需要停机/换卷/异常时：点击 **停止**。

### 4) 暂停 / 继续（不关程序）

新版支持“暂停/继续”（不杀进程、不关相机连接），适合临时停线/换卷。

- **暂停**：界面点击“暂停/停止”后，软件会写入 `config/runtime_state.json`，在线检测进程按该状态暂停处理。
- **继续**：再次点击“开始/继续”，软件会将暂停状态切回继续。

如果你是操作员：只需要记住 **暂停=临时停一下**、**停止=真正结束本次检测**。

---

## 二、工程师安装与首次配置（必须做一次）

### 1) Python 环境与依赖

建议使用与产线一致的 Python 环境。
安装依赖（示例）：

```bash
pip install -r ui/requirements.txt
```

### 2) 关键目录说明（不要随便改名）

- `ui/`：主界面程序（入口 `ui/main.py`）
- `app/online/`：在线检测入口（界面点击“开始”后运行 `python -m app.online.detect_anomalies_online`）
- `app/report/`：报告生成入口（界面内生成报告时运行 `python -m app.report.gen_report_cls`）
- `models/patchcore_model/`：PatchCore 推理代码与权重目录（重点）
- `config/`：配置文件
- `detect result/`：运行输出（检测结果、JSON、缺陷图片、报告等）
- `external/`：外部程序目录（建议放相机采集程序）

### 3) 相机采集程序（C#）位置

软件会尝试启动 C# 采集程序。你可以用两种方式指定：

- **方式 1（推荐）**：把程序放到：
  - `external/MultiCamDemo/MultiCamDemo.exe`
- **方式 2**：使用环境变量：
  - `MULTICAM_DEMO_EXE`：指向 `MultiCamDemo.exe`
  - `MULTICAM_DEMO_CWD`：程序工作目录

### 4) 配置文件（最重要）

#### `config/config0.yaml`（产线当前卷/条带配置）

常用字段：
- `conduct_id`：生产卡号（会影响输出目录）
- `strip_count`：条带数量（1~4）
- `fukuan_list` 或 `fukuan_1..4`：每条带的设定幅宽（mm）

#### `config/config.yaml`（检测参数）

本版本只保留 **PatchCore 运行链路**。
你需要确认：
- `patchcore_weights_root`：PatchCore 权重根目录名（在 `models/patchcore_model/` 下）
- 各相机 `camX_*` 的 PatchCore 参数（阈值/裁边/背景核等）

---

## 三、PatchCore 权重放置规则（非常重要）

在线检测会按如下规则寻找权重（以 CAM1 为例）：

- `models/patchcore_model/<patchcore_weights_root>/image_data_patchcore_0228/CAM1/patchcore_memory.npz`

如果提示“未找到 PatchCore 权重文件”，说明权重目录不对或没拷贝过来。
请工程师把训练得到的 `patchcore_memory.npz` 放到上述路径。

---

## 四、报告生成与查看

软件内有“报告中心/报告打印”等入口。
报告生成会读取：
- `detect result/<日期>/<卡号>/...` 下的检测结果
- `config/rptcfg.yaml`（类别名称、允收矩阵、打印项等）
- `table.json`（允收矩阵缓存；可由 `ui/make_standard.py` 生成）

---

## 五、常见故障排查（按这个顺序看）

### 1) 点了“开始”没反应

- 是否安装了 Python 依赖（`pip install -r ui/requirements.txt`）
- 是否有权限写入 `detect result/`（建议放在非系统盘/有写权限目录）
- 查看控制台输出（工程师用）

### 2) 提示找不到 PatchCore 权重

- 按“第三部分：权重放置规则”检查 `patchcore_memory.npz` 路径是否存在。

### 3) 相机无图/无数据

- C# 采集程序是否能独立运行
- 是否设置了 `MULTICAM_DEMO_EXE` / `MULTICAM_DEMO_CWD`
- 相机/网线/交换机是否正常

### 4) 软件卡顿/变慢

- 检查磁盘剩余空间
- `config/config.yaml` 的 `debug_io` 必须为 `false`（生产环境）
- `detect result/` 是否过大（可按日期归档到移动硬盘/服务器）

---

## 六、维护与备份建议（工程师）

- 定期把 `detect result/` 按日期归档（例如每班/每天压缩）
- 版本更新建议：先在测试机验证，再上产线
- GitHub 仓库不包含大文件（权重/训练结果/检测结果），需要单独备份

---

## 七、仓库与版本管理

GitHub：`https://github.com/Star-walkin/SlittingLine-Steel-Defect-Detection.git`
