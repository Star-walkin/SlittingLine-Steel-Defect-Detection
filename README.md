# SlittingLine 钢带缺陷检测系统（PatchCore 便携运行版）

## 文档说明

本文档面向 **现场操作员、班组长、工艺/电气维护、软件开发/部署人员**，对当前 GitHub 便携仓库版本做**尽可能细**的功能说明与操作说明。

**重要约束（本仓库）**

- **在线异常检测只使用 PatchCore**：检测子进程为 `python -m app.online.detect_anomalies_online`，不依赖 UNet / eff_ad / draem 等其它检测模型；`config.yaml` 中已删除仅用于旧链路的权重路径键，避免误导。
- **主界面**仍为桌面程序：默认 **PyQt5**，若本机未安装 PyQt5，源码会自动 **回退到 PySide6**（需自行安装对应依赖）。
- **相机图像**一般由 **C# 采集程序（MultiCamDemo）** 通过网络送入检测端；本仓库不包含相机厂商 SDK，仅说明如何放置/启动可执行文件。

---

## 一、系统是干什么的（一句话 + 架构）

系统用于 **分条线钢带表面缺陷在线检测与质量管理**：相机拍到的钢带表面图像进入检测程序，输出 **缺陷位置、缺陷图、幅宽（钢带宽度）变化曲线** 等；主界面负责 **启停产线、人机交互、幅宽与缺陷可视化、报告与类别配置**；数据落在 `detect result/` 目录，可按 **日期 + 质保书号（conduct_id）+ 相机 + 条带** 归档。

**典型数据流**

1. **C# MultiCamDemo**：采集图像 → 按协议发往后端（Socket）。  
2. **Python `app.online.detect_anomalies_online`**：收图 → PatchCore 推理与后处理 → 写 `detect result/`（JSON / jsonl / 图片 / 幅宽曲线数据等）。  
3. **Python `ui/main.py`（主界面）**：读 `detect result/` 与 `config/`，刷新曲线与缺陷图；必要时拉起报告子进程。  

---

## 二、主界面功能清单（对照 `ui/main.py`）

下列功能均来自主入口及其引用的窗口类，便于你对照界面按钮理解与培训工人。

### 1. 产线启停与运行态

| 功能 | 典型控件/行为 | 说明 |
|------|----------------|------|
| **开始检测** | `pushButton_start` → `button_start_click` | 启动/恢复在线检测：通常先写 `config/runtime_state.json` 为继续状态，再启动子进程 `python -m app.online.detect_anomalies_online`（工作目录为**仓库根目录**），然后启动或 **召回** C# 采集程序。若子进程已在运行，可能 **不重复启动**（避免断连）。 |
| **停止** | `pushButton_stop` → `button_stop_click` | 停止产线相关逻辑；可能结束子进程并写暂停/停止相关状态（以当前代码为准）。**与「暂停」策略不同**：停止更接近结束当次运行。 |
| **暂停 / 继续（不关 Socket）** | 读写 `config/runtime_state.json` | 字段形如 `{"paused": true/false}`。检测端读到暂停后按策略 **少处理或不处理业务**，但**不一定**断开相机连接；适合 **换卷、短暂停线**。 |
| **产线是否在动** | `config/line_heartbeat.json` | 检测端在收到图像时刷新时间戳；主界面可据此判断 **静止/运行** 等，并与 `config.yaml` 中「静止追平」等参数配合（若启用）。 |

### 2. 生产参数区（当前卷 / 条带）

主界面顶部有 **质保书号、条带数量、各条带设定幅宽、带钢卡号、产品型号** 等输入；并做了 **栅格布局** 与 **Tab 顺序**，避免四条幅宽与产品型号重叠，并符合「从左到右、先幅宽后卡号」的录入习惯。



| 数据项 | 作用 |
|--------|--------|
| **质保书号 / conduct_id** | 与 `config0.yaml` 中 `conduct_id` 对应，决定 `detect result/<日期>/<卡号>/…` 的结果根路径（以实际代码拼接为准）。 |
| **条带数量（strip_count）** | 1～4 条；影响界面显示多少条幅宽/卡号输入，以及检测端建几条带目录。 |
| **幅宽 1…N（mm）** | **设定幅宽**（工艺目标/标定宽度）；与实测幅宽对比用于 **预警/报警**。 |
| **带钢卡号 1…N** | 每条带的「名称」；经 `strip_result_paths.py` **清洗非法 Windows 字符**并 **消重** 后，作为 `detect result` 下 **子文件夹名**（与旧版 `strip_1` 兼容）。卷根会有 `config0_snapshot.yaml` 的 `strip_dir_list` 记录最终目录名列表。 |
| **产品型号** | 与分类/报告配置 `rptcfg.yaml`、`table.json`、类别配置等联动（以你现场 data1… 配置为准）。 |

**条带左右与图像左右的对应关系（重要）**



主界面存在 **UI 槽位 → 物理条带序号** 的映射（`_truth_strip_index_1based`）：界面 **从左到右第 k 个** 输入，对应图像上 **从右数第 k 条带**，以满足「从左到右录入顺序」与产线表述一致，而 **不重写检测端几何定义**。



### 3. 幅宽监测（曲线 + 右侧状态 + 报警逻辑）



| 功能 | 说明 |
|------|------|
| **幅宽曲线** | 使用 **QPainter 自绘折线 + 外置坐标轴 QLabel**，避免 Matplotlib 文字挤占；与右侧 **幅宽状态** 面板配合布局（像素级常量如 `FUKUAN_WAVE_HOST_X` 等可在源码中调）。 |
| **允许带（对称容差）** | 代码中 **偏窄 / 偏宽均算「超出范围」**：下阈 = 设定 − max(绝对容差, 设定×相对比例)；上阈 = 设定 + 同带宽。默认常量见源码：`FUKUAN_NARROW_ABS_MM`、`FUKUAN_NARROW_REL`。 |
| **报警** | **最近 `FUKUAN_ALARM_WINDOW` 帧全部超出允许范围** → 报警（源码默认如 32 帧，以你同步版本为准）。 |
| **预警** | **当前序列末尾**连续超出允许范围达到 `FUKUAN_WARN_TAIL_STREAK` 帧，且未满足报警条件 → 预警（源码默认如 6 帧）。 |
| **双轨幅宽信息** | UI 可展示 **Stable** 主显示与 **Raw/Meta** 追溯；检测端在条带子目录写 `fukuan.json`、`fukuan_raw.json`、`fukuan_meta.json`（初始化时空列表占位）。 |
| **内存保护** | `TOTAL_DATA_MAX_SAMPLES` 限制幅宽历史采样点数，避免长时间运行 `fukuan.json` 极长导致 UI 内存与 CPU **线性爆炸**。 |
| **刷新频率** | 定时器约 **12.5 FPS** 量级刷新曲线（代码注释），在流畅度与读盘负载间折中。 |
**给操作员的口语解释**：绿色/正常表示宽度在「允许带」内；预警表示**最近连续几帧**偏出；报警表示**更长一串帧**都偏出——需要班长或工程师看是产线真有问题还是标定/相机问题。



### 4. 缺陷分布与缺陷图



| 功能 | 说明 |
|------|------|
| **缺陷点显示** | `ImageLoaderThread` 按相机、按条带异步读取 `detect result` 下坐标文件；支持 **json / jsonl** 等（取决于检测端写盘格式与配置）。 |
| **长度轴映射** | `_defect_length_mm_to_px`：缺陷与幅宽共用「物理长度 → 像素」映射；可在 `config.yaml` 打开 **非线性轴**（尾部物理区间压到更少像素）。相关键如 `ui_defect_window_target_m`、`ui_defect_axis_nonlinear`、`ui_defect_axis_tail_phys_ratio` 等。 |
| **防空白滑动** | 缓存每个系统上/下表面 **最近一次缺陷出现的长度锚点**，避免窗口滑动后长时间无点可看。 |
| **查看原图** | 双击（或事件过滤器）判断是否为缺陷/实时图控件，调用 `_open_image_path_with_system_viewer`：Windows 下优先 `os.startfile`，否则 `QDesktopServices.openUrl`。 |
| **多检测系统** | 界面按 **上表面/下表面** 与 **条带索引** 多窗显示；具体 QLabel 绑定见 `_label_kind_for_defect_host`。 |
### 5. 报告与质量管理



| 功能 | 入口 | 说明 |
|------|------|------|
| **报告（修改/打印类）** | `pushButton_report` → `pushButton_report_click` | 打开 `ReportWindow`（`ui/report_change.py`）。内部通过子进程执行：`python -m app.report.gen_report_cls`（并传日期、质保书号、条带 id 等参数）。 |
| **报告中心** | `ReportCenterWindow` | 集中查看/管理报告相关流程（以界面为准）。 |
| **细节优化矩阵** | `report_change` 内置轻量统计 | 为减少 **打包 exe 时 torch DLL 问题**，部分统计从 `function_bank` 剥离为本地实现（见文件中 `_DetailStatisticLite` 注释）。 |
**报告依赖文件**



- `detect result/...`：检测结果与缺陷坐标。  

- `config/rptcfg.yaml`：报告抬头、类别显示、打印筛选等。  

- `table.json`：允收矩阵等（通常由 `ui/make_standard.py` 在类别流程中生成）。  



### 6. 类别配置与分类训练



| 按钮 | 窗口 | 密码/权限 | 说明 |
|------|------|-----------|------|
| **类别配置** | `ClsConfigWindow` | 通常需 `auth.yaml` 中 `cls_config` 密码 | 维护产品型号（如 dataN）、缺陷类别名称、允收矩阵等；与报告、检测后分类统计相关。 |
| **分类训练** | `ClsTrainWindow` | 视实现而定 | 打开训练界面；构造时传入 `is_detect_running_fn`，避免检测运行时误操作（以弹窗提示为准）。 |
| **缺陷分类向导** | `ClsWizardWindow` | **不需要密码**（代码注释「工人小白模式」） | 面向一线 **分步骤** 完成：类型配置、训练准备等。 |
### 7. 参数与配置保存



| 功能 | 说明 |
|------|------|
| **参数设置** | `pushButton_para` → `ParaWindow` | 集中编辑 `config.yaml` / `config0.yaml` 中大量键（具体项见 `ui/para.py` 定义）。 |
| **保存 config01（界面按钮）** | `para_config01` → `save_config01` | 将界面输入写回 `config/config0.yaml` 等（以函数内逻辑为准），用于开卷前保存质保书号、条数、幅宽、卡号等。 |
### 8. 其它交互与工程项



| 功能 | 说明 |
|------|------|
| **交换/换卷（exchange）** | `button_exchange` → `exchangeNEWONE` | 用于切换卷、重置部分 UI 状态（详见源码）。 |
| **报警关闭** | `pushButton` → `baojing_close` | 通过 **串口 COM4** 写固定字节；现场必须确认串口设备存在，否则可能抛错。 |
| **主题** | `_apply_app_theme` | 加载同目录 `theme.qss`；若 `PyInstaller` 打包，qss 可能在 `_MEIPASS`。 |
| **Matplotlib** | `main.py` 仍 `import matplotlib` | 部分图表/坐标相关能力保留（与 QPainter 幅宽曲线并存，以实际界面为准）。 |
| **PyInstaller** | `getattr(sys, "frozen", False)` 分支 | 主工程考虑 onefile 部署；便携 Git 版一般 `frozen=False`。报告脚本中 `STEELDEFECT_ROOT` 可指向数据目录。 |
### 9. 在线检测端（子进程）要点



子进程模块：`app/online/detect_anomalies_online.py`（务必从仓库根 `-m` 运行）。



| 能力 | 说明 |
|------|------|
| **PatchCore** | 从 `models/patchcore_model/<patchcore_weights_root>/image_data_patchcore_0228/CAMx/patchcore_memory.npz` 载入内存库。 |
| **条带子目录** | 与主界面/快照一致：`strip_result_paths.build_strip_dir_names` + `resolve_strip_dir_basename`。 |
| **幅宽多点 JSON** | 条带目录中 `fukuan*.json` 等初始化；Stable 给 UI，Raw/Meta 给自己/工艺追溯。 |
| **缺陷图写盘限流** | `save_defect_images_*` 键控制峰值，保护磁盘与 CPU；jsonl 坐标流一般不限。 |
| **速度监控** | `app.common.speed_monitor` 输出各阶段耗时与 FPS（见配置 `speed_report_interval_sec` 等）。 |
---

## 三、目录结构（现场需要认识的文件夹）

| 路径 | 内容 |
|------|------|
| `ui/main.py` | **主程序入口** |
| `ui/*.py` | 界面逻辑：`para.py`、`report_change.py`、`report_center.py`、`cls_*.py` 等 |
| `app/online/` | 在线检测 |
| `app/report/` | 报告生成 `gen_report_cls.py` |
| `app/common/` | `function_bank.py`、`cls_model.py`、`speed_monitor.py` 等 |
| `models/patchcore_model/` | PatchCore **代码 + 权重目录** |
| `config/` | 全部配置文件 |
| `detect result/` | **运行产出**（建议在 `.gitignore` 中忽略大文件） |
| `strip_result_paths.py` | **条带结果目录名**解析（与主工程对齐） |
| `external/MultiCamDemo/` | **推荐**放 `MultiCamDemo.exe` |
| `DalsaGrabDemoTcp/…` | 可选：若你把 C# 工程也放进同一仓库并编译，`main.py` 也能自动找到 `bin/Release` 或 `bin/Debug` |
---

## 四、安装与部署（工程师按顺序做）

### 1. 克隆仓库

```powershell

git clone https://github.com/Star-walkin/SlittingLine-Steel-Defect-Detection.git

cd SlittingLine-Steel-Defect-Detection

```

### 2. 安装 Python 依赖



```powershell

pip install -r ui/requirements.txt

```



若你希望 **无 PyQt5 时走 PySide6**，请自行安装 PySide6（与主程序 `except ModuleNotFoundError` 分支一致）。



### 3. 放置 PatchCore 权重



见下文 **第九节**。



### 4. 放置/编译相机程序



**任选其一**（`main.py` 中 `_ensure_csharp_running` 的优先级）：



1. **环境变量（最灵活）**  

   - `MULTICAM_DEMO_EXE`：`MultiCamDemo.exe` 完整路径  

   - `MULTICAM_DEMO_CWD`：工作目录（可不设，默认可执行文件所在目录）



2. **`external/MultiCamDemo/MultiCamDemo.exe`**（推荐现场拷贝）



3. **仓库内编译输出**  

   - `DalsaGrabDemoTcp/MultiCamDemo/bin/Release/MultiCamDemo.exe`  

   - 或 `…\bin\Debug\…`



### 5. 指定 Python 解释器（可选）



- `STEEL_PYTHON_EXE`：当系统默认 `python` 不是运行环境时，用于 **子进程**（检测、报告、make_standard）。



### 6. 首次检查配置



- `config/config0.yaml`：`conduct_id`、`strip_count`、幅宽与卡号字段。  

- `config/config.yaml`：`patchcore_weights_root`、各 `camN_*`、幅宽/缺陷显示、`active_cam_ids`、静止追平、`debug_io: false`（生产环境）。  

- `config/auth.yaml`：改掉默认密码 `000`。  



---



## 五、操作员使用步骤（最细日常流程）



### 班前



1. 开机，登录 Windows。  

2. 确认 **磁盘空间**（`detect result` 会涨）。  

3. 确认 **相机、光源、网络** 正常；必要时先单独打开 C# 采集做硬件自检。  



### 打开软件



在仓库根目录：



```powershell

cd ui

python main.py

```



### 开卷前输入



1. **质保书号**（conduct_id）与 **条数**。  

2. 从左到右填 **幅宽 1…N**、**卡号 1…N**（与界面提示「从东向西」一致）。  

3. 选对 **产品型号**（若当班切换钢种/涂镀层）。  

4. 点击 **保存 config01**（或你现场约定的保存按钮），确认无校验错误提示。  



### 启动检测



1. 点 **开始**。  

2. 观察：幅宽曲线是否刷新；缺陷缩略图是否更新；若长时间无图，通知工程师看 **C# 与检测子进程控制台**。  

3. **暂停**：短停、换卷前可依赖 `runtime_state.json` 策略（界面具体操作以当前版本为准）。  

4. **停止**：结束当班或检修时点 **停止**。  



### 看缺陷图



- **双击** 缺陷小图区域，用系统看图打开大图路径（若不是有效文件会打不开）。  



### 打报告



1. 打开 **报告** 窗口。  

2. 填/确认 **日期**、**质保书号**、**条带号**。  

3. 执行打印或生成（以界面流程为准）。  



### 下班



1. **停止** 软件与采集。  

2. 备份或归档当日 `detect result`（移动硬盘/服务器）。  



---



## 六、班长 / 工艺员常见问题处置



| 现象 | 建议 |
|------|------|
| 幅宽频繁预警/报警 | 对比 Raw/Meta；检查标定、辊缝、算法 cam 的 times 与 speed；必要时调 `FUKUAN_*` 常量（需开发人员发版）。 |
| 缺陷点位置与实物不符 | 查 `standard_ratio_x`、`camN_times`、条带映射是否反了；查检测端写坐标格式。 |
| 条带 folder 名与卡号不一致 | 看卷根 `config0_snapshot.yaml` 的 `strip_dir_list`；属「清洗/消歧」正常行为。 |
| 报告类别统计为 0 | 查 `rptcfg.yaml` 与 `table.json`；类别 ID 是否与检测端输出一致。 |
---



## 七、工程师维护手册



### 1. 检测子进程手工启动（排障）



在仓库根：



```powershell

python -m app.online.detect_anomalies_online

```



用于对照 UI 启停逻辑，直接看控制台报错。



### 2. 报告子进程手工启动（排障）



```powershell

python -m app.report.gen_report_cls --help

```



### 3. 与主工程同步本仓库（开发人员）



```powershell

python tools/sync_from_main_project.py --source-root "d:\pycharm_project\steeldefect"

```



同步后本仓库仍应 **保持 PatchCore-only** 与可移植路径；若主工程新增根目录 `.py` 并被 UI `import`，请把文件加入 `tools/sync_from_main_project.py` 的 `ROOT_FILE_MAP` 或 `UI_FILES`。



### 4. 性能与稳定性



- 生产环境 **`debug_io` 必须为 false**。  

- `save_defect_images_*` 按磁盘与合规要求调节。  

- `line_idle_*`（若启用）避免静止时队列积压；调大 `line_idle_catchup_max_frames` 可能增加单次突发 CPU。  



---



## 八、配置文件详解



### `config/config0.yaml`（「当前卷」业务快照）



- **conduct_id**：质保书号/生产卡号。  

- **strip_count**：条数。  

- **fukuan_*** / **fukuan_list**：设定幅宽。  

- **strip_card_*** / **strip_card_list**：卡号原始字符串。  

- 开卷后卷根生成的 **`config0_snapshot.yaml`**：固化 **`strip_dir_list`**，供 UI/报告与检测端解析一致。  



### `config/config.yaml`（检测 + UI 行为）



**相机 PatchCore**



- `patchcore_weights_root`  

- `camN_seg_anomaly_thres`、`camN_patchcore_k`、`camN_use_gradient_detection`、`camN_grad_threshold`  

- `camN_patchcore_edge_*`、`camN_bg_ksize`、`camN_diff_threshold`、`camN_edge_crop` 等  



**缺陷与长度显示**



- `ui_defect_window_target_m`、`ui_defect_window_max_mm`、`ui_defect_backtrack_max_mm`、`ui_defect_lag_margin_mm`  

- `ui_defect_coord_poll_ms`  

- `ui_defect_axis_nonlinear`、`ui_defect_axis_tail_phys_ratio`、`ui_defect_axis_tail_pixel_ratio`  



**产线静止追平（若启用）**



- `line_idle_catchup_enable`、`line_idle_stale_sec`、`line_idle_catchup_max_frames`  



**相机启用**



- `active_cam_ids`：只跑部分相机时减少负载。  



**分类**



- `cls_model_path`：分类权重（与 `cls_model/` 目录配合）。  



### `config/auth.yaml`



```yaml

passwords:

  standard_report: "..."      # 标准报告相关

  parameter_settings: "..."   # 参数设置

  cls_config: "..."            # 类别配置

```



### `config/rptcfg.yaml`



报告展示名、屏蔽类、打印过滤、与 `conduct_id`/日期等字段的默认联动。



### `table.json`



允收矩阵缓存；与「类别配置 → make_standard」流程更新。



---



## 九、PatchCore 权重路径（必须背下来）



对每个相机 `CAMn`，默认：



`models/patchcore_model/<patchcore_weights_root>/image_data_patchcore_0228/CAMn/patchcore_memory.npz`



例如 `patchcore_weights_root: weights`：



`models/patchcore_model/weights/image_data_patchcore_0228/CAM1/patchcore_memory.npz`



---



## 十、故障排查 FAQ



| 现象 | 排查顺序 |
|------|-----------|
| 点了开始没曲线 | C# 是否启动；检测子进程是否启动；`detect result` 是否创建；`runtime_state` 是否 paused。 |
| 检测子进程秒退 | 控制台看报错；多为缺包、缺权重、CUDA/torch 问题。 |
| 幅宽全 0 或未激活 | `fukuan.json` 是否写；条带目录名是否错位；`strip_dir_list` 是否一致。 |
| 报告乱码/缺字体 | 安装「微软雅黑」或自行配置 matplotlib / Qt 字体（`config/simhei.ttf` 若存在）。 |
| GitHub 上没权重 | **正常**；权重与整卷 `detect result` 需离线拷贝。 |
---



## 十一、仓库地址



`https://github.com/Star-walkin/SlittingLine-Steel-Defect-Detection.git`



---



## 十二、附录：主工程里的其它脚本（本仓库不一定包含）



你在主工程里可能还有 **`simulate_cams.py`** 等 **仿真推流/离线回放** 工具，用于无相机调试；便携仓库以 `tools/sync_from_main_project.py` 同步列表为准。若需把仿真脚本也纳入 Git 便携版，请扩展同步配置并补充文档。


