# SlittingLine 钢带缺陷检测（PatchCore 便携运行版）

本项目面向 **分条线钢带表面缺陷在线检测**：相机采集端（通常为 C#）通过 **网络** 将图像帧送入 Python 检测端；主界面（PyQt5，可选回退 PySide6）负责 **启停产线、幅宽监测、缺陷分布与结果展示、报告与分类配置** 等。便携仓库内 **在线检测仅使用 PatchCore**（不包含 UNet / 其它检测回退链路）。

---

## 一、软件能做什么（功能总览）

### 1. 在线缺陷检测（PatchCore）

- 点击 **开始** 后，界面会启动子进程：`python -m app.online.detect_anomalies_online`（工作目录为仓库根目录）。
- 检测端按 `config/config.yaml` 与 `config/config0.yaml` 中的参数，对多路相机图像做 **PatchCore 异常检测**，并把结果写入 `detect result/`。
- 支持 **产线暂停/继续**：主界面写入 `config/runtime_state.json`，检测端读到 `paused: true` 后按策略暂停处理，**不强制杀进程、不关 Socket**（适合换卷、短暂停线）。

### 2. 幅宽（钢带宽度）监测与曲线

- 主界面为每条带、每个检测系统绘制 **幅宽随长度/时间变化的曲线**（内置 QPainter 绘制，外置坐标轴标签）。
- 支持 **设定幅宽** 与 **实测幅宽** 对比；对「显著偏窄」采用 **工程容差**（绝对 + 相对阈值），避免噪声导致误报。
- 状态区区分 **正常 / 预警 / 报警**（与连续多帧偏窄统计相关）；鼠标悬停可查看说明。
- 支持 **Stable / Raw** 等追溯信息（检测端在条带子目录写入 `fukuan.json`、`fukuan_raw.json`、`fukuan_meta.json` 等，供界面解释幅宽来源）。

### 3. 缺陷分布与缺陷图浏览

- 按配置的时间窗口，在「长度轴」上叠加显示缺陷点位；支持 **非线性长度轴**（尾部物理区间压缩到更少像素），可在 `config/config.yaml` 中调节。
- 线程异步读取 `detect result/` 下对应卷的坐标数据（支持 **json / jsonl** 等形态，以你当前配置为准）。
- 对缺陷缩略图支持 **双击用系统默认看图软件打开** 完整路径图片。

### 4. 条带目录与卡号（带钢身份）

- 每条带在结果目录下的文件夹名，可与 **带钢卡号** 对齐；非法 Windows 字符会被清洗，重名会自动加后缀。
- 卷根目录可写入 `config0_snapshot.yaml` 中的 `strip_dir_list`，与主界面、报告侧解析一致；解析逻辑见仓库根目录 **`strip_result_paths.py`**（与主工程同步）。

### 5. 报告与分类

- **报告**：通过界面入口生成/打印质保类报告（依赖 `detect result/`、`config/rptcfg.yaml`、`table.json` 等）。
- **类别配置**：维护缺陷类别、允收矩阵等（入口通常需密码，见 `config/auth.yaml`）。
- **分类模型训练 / 向导**：提供 `ClsTrainWindow`、`ClsWizardWindow` 等界面（具体使用以你现场流程为准）。
- 在线或后处理阶段可使用 **轻量分类网络**（权重路径见 `config/config.yaml` 中 `cls_model_path`）。

### 6. 参数与其它

- **参数设置**窗口：集中调整检测与界面相关参数（写入 `config.yaml` / `config0.yaml` 等，以界面实现为准）。
- **主题**：启动时加载 `ui/theme.qss`（若存在）。
- **外设**：主工程内保留串口关闭报警等接口（如 `COM4`），现场部署前请由工程师确认串口号与设备是否存在，避免误连。

### 7. 与采集程序（C#）的协同

- 主界面会尝试启动 **MultiCamDemo.exe**（或你放置的同名采集程序），默认路径为仓库下 `external/MultiCamDemo/MultiCamDemo.exe`，也可用环境变量覆盖（见下文）。

---

## 二、目录结构（你需要知道的）

| 路径 | 说明 |
|------|------|
| `ui/main.py` | **主程序入口**（操作员日常打开） |
| `app/online/detect_anomalies_online.py` | 在线检测（由界面以 `-m` 方式启动） |
| `app/report/gen_report_cls.py` | 报告生成模块（由报告界面子进程调用） |
| `app/common/` | 公共算法与工具（如 `function_bank.py`） |
| `models/patchcore_model/` | PatchCore 推理代码与 **权重根目录** |
| `config/` | `config.yaml`、`config0.yaml`、`auth.yaml`、`rptcfg.yaml` 等 |
| `detect result/` | 运行输出（按日期、卡号、相机、条带组织） |
| `external/MultiCamDemo/` | **建议**放置 `MultiCamDemo.exe` 及依赖 |
| `strip_result_paths.py` | 条带结果目录名解析（与主工程一致） |

---

## 三、环境要求与安装（工程师首次部署）

### 1. 操作系统与 Python

- **Windows 10/11**（当前界面与路径约定主要针对 Windows）。
- 安装 **Python 3**（与产线训练环境主版本一致），并勾选「Add Python to PATH」或使用虚拟环境。

### 2. 安装依赖

在仓库根目录执行：

```bash
pip install -r ui/requirements.txt
```

若本机未安装 PyQt5，主程序会尝试 **自动回退 PySide6**（需已安装 PySide6）。

### 3. 放置相机采集程序（C#）

**方式 A（推荐）**：将采集程序放到：

- `external/MultiCamDemo/MultiCamDemo.exe`

**方式 B**：设置环境变量后启动主界面（同一用户会话内有效）：

- `MULTICAM_DEMO_EXE`：`MultiCamDemo.exe` 的完整路径  
- `MULTICAM_DEMO_CWD`：该程序运行所需的工作目录  

### 4. 指定 Python 解释器（可选）

若系统 PATH 中的 `python` 不是运行环境，可设置：

- `STEEL_PYTHON_EXE`：指向实际用于检测子进程的 `python.exe`

---

## 四、PatchCore 权重放置（必须正确）

在线检测按下列规则查找（以 CAM1 为例，`patchcore_weights_root` 默认多为 `weights`）：

`models/patchcore_model/<patchcore_weights_root>/image_data_patchcore_0228/CAM1/patchcore_memory.npz`

若启动后日志或界面提示 **找不到权重**，请检查：

1. `config/config.yaml` 中 `patchcore_weights_root` 是否与磁盘文件夹名一致；  
2. `CAM1`～`CAM4` 目录名是否大写一致；  
3. `patchcore_memory.npz` 是否已拷贝到上述路径（大文件通常 **不入 Git**，需现场拷贝）。

---

## 五、配置文件说明（工程师必读）

### 1. `config/config0.yaml`（当前卷 / 条带业务数据）

常见字段（名称以你文件为准）：

- `conduct_id`：**质保书号 / 生产卡号**，会参与 `detect result/` 下目录命名。  
- `strip_count`：条带数量（1～4）。  
- 各条带 **设定幅宽**（如 `fukuan_list` 或 `fukuan_1`…）。  
- **带钢卡号**（如 `strip_card_*` / `strip_card_list`）：用于结果子目录命名（经 `strip_result_paths.py` 清洗）。  

修改后一般需在界面中 **保存** 或按你现场流程重启检测。

### 2. `config/config.yaml`（检测与界面参数）

与 PatchCore 强相关的项包括（示例，以实际文件为准）：

- `patchcore_weights_root`  
- 各相机 `camN_patchcore_k`、`camN_use_gradient_detection`、`camN_seg_anomaly_thres`、边缘抑制、背景差分参数等  
- **缺陷窗口 / 长度轴**：如 `ui_defect_window_target_m`、`ui_defect_axis_nonlinear` 等  
- **产线静止追平**（若启用）：如 `line_idle_catchup_enable`、`line_idle_stale_sec` 等  
- **缺陷图写盘限流**（若存在）：`save_defect_images_*` 系列键，用于保护磁盘与 CPU  

本便携版 **已移除** 仅用于 UNet / eff_ad / draem 等旧链路的配置键，避免误导。

### 3. `config/auth.yaml`（界面密码）

- 含 `standard_report`、`parameter_settings`、`cls_config` 等角色的口令。  
- **部署到生产前务必修改默认密码**，并限制该文件权限。

### 4. `config/runtime_state.json`（暂停开关）

- 由主界面写入：`{"paused": true/false}`。  
- 操作员无需手改；若需强制继续，可由工程师在停线状态下检查该文件是否为 `false`。

### 5. `config/line_heartbeat.json`（产线心跳，可选）

- 由检测端周期性刷新；界面可据此判断 **是否在收图 / 是否静止** 等（与 `config.yaml` 中相关项配合）。

### 6. `config/rptcfg.yaml` 与 `table.json`（报告）

- `rptcfg.yaml`：报告抬头、类别显示名、打印选项等。  
- `table.json`：允收矩阵等（可由 `ui/make_standard.py` 在类别配置流程中生成/更新）。

---

## 六、详细使用方法（按角色）

### A. 操作员（日常最短路径）

1. **开机**：确认采集电脑与相机、网络正常。  
2. **打开软件**：在仓库根目录打开命令行，执行：
   ```bash
   cd ui
   python main.py
   ```
3. **核对顶部信息**：质保书号、条数、幅宽、卡号等与当班计划一致（按班长/工艺员要求填写）。  
4. 点击 **开始**：  
   - 应出现检测子进程日志（若工程师打开了控制台）；  
   - 幅宽曲线、缺陷分布应随运行更新。  
5. **临时停一下**：使用界面上的 **暂停**（若有）或按现场规程操作；**不要**随意杀进程。  
6. **结束当班**：点击 **停止**；确认 `detect result` 当日目录已写入。  
7. **报告**：在「报告」相关按钮中按提示选择卷、条带，填写日期与质保书号后打印或导出（以界面为准）。

### B. 班长 / 工艺员（需要改卷参数时）

1. 在 **停止** 或确认无连续生产任务时，修改 `config0.yaml` 或通过界面 **保存** 当前卷参数（以你现场规定为准）。  
2. 修改 **条带数量、卡号、设定幅宽** 后，再 **开始** 新一卷。  
3. 若结果目录中条带文件夹名与卡号不一致，属 **非法字符清洗或重名消歧**，以 `config0_snapshot.yaml` 中 `strip_dir_list` 为准。

### C. 工程师（安装、调参、排障）

1. 完成 **第三节** 安装与 **第四节** 权重部署。  
2. 用 **测试模式** 或现场低速验证：确认 `detect result/<日期>/<conduct_id>/` 下各相机目录、条带子目录、`defect_images` 是否在写入。  
3. 调 `config.yaml`：  
   - 先保证 `debug_io: false`（生产环境）；  
   - 再按需调节 PatchCore 阈值与边缘抑制，观察误报/漏报。  
4. **卡顿**：检查磁盘空间、`save_defect_images_*` 限流、相机帧率与 `line_idle_*` 是否合理。  
5. **无图**：先独立启动 C# 采集；再查网络与检测端口；查看检测端控制台输出。  
6. **分类/报告异常**：查 `rptcfg.yaml`、`table.json` 与 `cls_model_path` 是否存在。

---

## 七、从主程序入口理解运行链路（`ui/main.py`）

1. 启动主窗口，加载 UI 与主题。  
2. 用户点击 **开始** → 写入 `runtime_state.json`（继续）→ 启动 `python -m app.online.detect_anomalies_online`（工作目录为仓库根）→ 再启动/召回 C# 采集程序。  
3. 用户点击 **停止/暂停** → 更新 `runtime_state.json`，必要时结束子进程（以当前主工程逻辑为准）。  
4. **幅宽线程 / 缺陷读取线程** 定时读 `detect result` 与配置文件，刷新曲线与图像。  
5. **报告 / 类别配置** 等子窗口通过密码门控打开，并可能拉起 `python -m app.report.gen_report_cls` 等子进程。

---

## 八、常见问题（FAQ）

**问：界面提示找不到 C# 程序？**  
答：检查 `external/MultiCamDemo/MultiCamDemo.exe` 是否存在，或设置 `MULTICAM_DEMO_EXE`。

**问：检测端提示找不到 PatchCore 权重？**  
答：按 **第四节** 检查路径与 `patchcore_weights_root`。

**问：报告里类别名不对？**  
答：改 `rptcfg.yaml` 后重新生成报告；允收矩阵改后需走 `make_standard` 更新 `table.json`。

**问：Git 克隆后没有权重和检测结果？**  
答：仓库刻意不包含大文件；权重与 `detect result` 需现场备份与拷贝。

---

## 九、仓库地址

GitHub：`https://github.com/Star-walkin/SlittingLine-Steel-Defect-Detection.git`

---

## 十、与主工程同步（开发人员）

在便携仓库根目录执行（只改本仓库内文件）：

```bash
python tools/sync_from_main_project.py --source-root "d:\pycharm_project\steeldefect"
```

同步后本仓库仍会 **强制保持 PatchCore-only 在线检测** 与可移植路径；若你增加了新的根目录级 `.py` 被 UI 引用，请同步更新 `tools/sync_from_main_project.py` 中的 `ROOT_FILE_MAP` 或 `UI_FILES`。
