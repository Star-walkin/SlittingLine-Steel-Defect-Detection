from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QApplication,
    QWidget,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QScrollArea,
    QFormLayout,
    QSizePolicy,
)
from configui import Ui_Parameter  # 引用生成的 ui_para.py 文件
import yaml
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import ast


class ParaWindow(QMainWindow, Ui_Parameter):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ======== 检测所需参数（来自 detect_anomalies_online.py） ========
        # 说明：参数设置窗口只管理 config/config.yaml 内的检测参数；config0.yaml（卡号/条数/幅宽）属于“生产/采集配置”，
        # 目前不在该窗口里改（避免混淆与误写）。
        self._detect_param_specs = self._build_detect_param_specs()
        self._param_controls = {}  # key -> control

        # 加载配置
        self._config_path = os.path.join(_REPO_ROOT, "config", "config.yaml")
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        # 重建界面布局（隐藏旧布局，构建滚动分组表单）
        self._rebuild_detect_params_ui()
        self._load_config_to_controls()

        # 预设 QComboBox 选项
        self.para_OK.clicked.connect(self.save_config)
        self.para_reset.clicked.connect(self.config_default)

    def _build_detect_param_specs(self):
        """
        返回 list[dict]，每项包含：
          key: config.yaml 键名
          name: 中文名
          type: "int"|"float"|"bool"|"str"|"choice"
          choices: 可选项（仅 type="choice"）
          group: 分组名
          help: 说明（可为空）
        """
        specs = []

        specs += [
            dict(key="Consecutive_Check", name="连续异常报警开关", type="bool", group="运行与报警", help="开启后在标定相机上按连续异常帧数触发报警"),
            dict(key="Consecutive_thres_num", name="连续异常帧数阈值", type="int", group="运行与报警", help="达到该连续帧数判定为连续异常"),
            dict(key="calibrat_cam_id", name="标定相机编号", type="choice", choices=["1", "2", "3", "4"], group="运行与报警", help="用于连续异常统计/标定的相机"),
            dict(key="steel_real_y0", name="钢带原始卷长度（km）", type="float", group="几何与坐标", help="用于长度坐标偏移/累计（未用可保持 0）"),
        ]

        # 相机公共项（1~4）
        for cam in range(1, 5):
            g_basic = f"相机{cam}（基础）"
            g_adv = f"相机{cam}（高级）"
            specs += [
                dict(key=f"cam{cam}_seg_anomaly_thres", name="异常分割阈值", type="float", group=g_basic, help="热力图/异常图二值化阈值"),
                dict(key=f"cam{cam}_standard_ratio_x", name="横向标定（mm/px）", type="float", group=g_basic, help="像素到毫米的比例（横向）"),
                dict(key=f"cam{cam}_times", name="纵向倍率（times）", type="float", group=g_basic, help="纵向比例系数，standard_ratio_y = standard_ratio_x * times"),
                dict(key=f"cam{cam}_model_name", name="检测模型路径", type="str", group=g_basic, help="UNet 或 det_model 的 checkpoint；脚本会自动适配"),
                dict(key=f"cam{cam}_cut_ratio", name="切分倍率（k）", type="int", group=g_basic, help="图像切成 k×k 做推理（与模型训练一致）"),
                dict(key=f"cam{cam}_img_size", name="推理输入尺寸（px）", type="int", group=g_basic, help="切块 resize 到该尺寸"),

                # PatchCore / 梯度 / 局部对比度等高级项（detect_anomalies_online.init_detect 使用）
                dict(key=f"cam{cam}_patchcore_k", name="PatchCore k", type="float", group=g_adv, help="PatchCore 异常分数放大/灵敏度系数"),
                dict(key=f"cam{cam}_use_gradient_detection", name="启用梯度缺陷检测", type="bool", group=g_adv, help="开启后会结合梯度/局部对比度进行缺陷定位"),
                dict(key=f"cam{cam}_grad_threshold", name="梯度阈值", type="float", group=g_adv, help="梯度检测阈值（越大越不敏感）"),
                dict(key=f"cam{cam}_blur_ksize", name="模糊核大小", type="int", group=g_adv, help="局部对比度/梯度前的模糊核尺寸（奇数）"),
                dict(key=f"cam{cam}_edge_crop", name="边缘裁剪（px）", type="int", group=g_adv, help="热力图边缘忽略像素，减少边缘伪检"),
                dict(key=f"cam{cam}_bg_ksize", name="背景估计核大小", type="int", group=g_adv, help="局部对比度背景滤波核（奇数，建议大一些）"),
                dict(key=f"cam{cam}_diff_threshold", name="对比度差分阈值", type="float", group=g_adv, help="局部对比度差分阈值"),
                dict(key=f"cam{cam}_patchcore_edge_soft_border", name="PatchCore 边缘过渡带（px）", type="int", group=g_adv, help="边缘抑制过渡带宽度"),
                dict(key=f"cam{cam}_patchcore_edge_strength", name="PatchCore 边缘抑制强度（0-1）", type="float", group=g_adv, help="越大越抑制边缘热力"),
                dict(
                    key=f"cam{cam}_patchcore_edge_weight_profile",
                    name="PatchCore 边缘权重曲线",
                    type="choice",
                    choices=["linear", "ease_out_cubic"],
                    group=g_adv,
                    help="边缘抑制的过渡曲线",
                ),
            ]
        return specs

    def _rebuild_detect_params_ui(self):
        # 隐藏旧版 frame（保留按钮）
        for nm in ("frame", "frame_2", "frame_3", "frame_4"):
            if hasattr(self, nm):
                getattr(self, nm).hide()

        # 调整按钮位置（靠右下）
        try:
            self.para_reset.setGeometry(QtCore.QRect(1020, 760, 300, 60))
            self.para_OK.setGeometry(QtCore.QRect(1020, 830, 300, 60))
        except Exception:
            pass

        # 新增滚动区域
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setGeometry(QtCore.QRect(20, 10, 980, 890))

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # 分组 -> QGroupBox + QFormLayout
        groups = {}
        for spec in self._detect_param_specs:
            groups.setdefault(spec["group"], []).append(spec)

        for group_name in groups.keys():
            gb = QGroupBox(group_name)
            form = QFormLayout(gb)
            form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            form.setFormAlignment(QtCore.Qt.AlignTop)
            form.setHorizontalSpacing(16)
            form.setVerticalSpacing(10)

            for spec in groups[group_name]:
                label = QLabel(spec["name"])
                label.setMinimumWidth(220)
                ctrl = self._make_control_for_spec(spec)
                self._param_controls[spec["key"]] = ctrl

                row = QWidget()
                row_l = QHBoxLayout(row)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(10)
                row_l.addWidget(ctrl, 1)
                if spec.get("help"):
                    hint = QLabel(spec["help"])
                    hint.setWordWrap(True)
                    hint.setStyleSheet("color: #666666;")
                    hint.setMinimumWidth(360)
                    row_l.addWidget(hint, 2)
                form.addRow(label, row)

            root_layout.addWidget(gb)

        root_layout.addStretch(1)
        self._scroll.setWidget(container)

    def _make_control_for_spec(self, spec):
        t = spec["type"]
        if t == "bool":
            cb = QCheckBox("启用")
            cb.setTristate(False)
            return cb
        if t == "choice":
            c = QComboBox()
            c.addItems([str(x) for x in spec.get("choices", [])])
            return c
        le = QLineEdit()
        le.setPlaceholderText("请输入数值" if t in ("int", "float") else "请输入")
        return le

    def _load_config_to_controls(self):
        cfg = self._config
        for spec in self._detect_param_specs:
            key = spec["key"]
            ctrl = self._param_controls.get(key)
            if ctrl is None:
                continue
            val = cfg.get(key)
            if spec["type"] == "bool":
                # 兼容 0/1/True/False
                ctrl.setChecked(bool(val))
            elif spec["type"] == "choice":
                if val is None:
                    continue
                txt = str(val)
                idx = ctrl.findText(txt)
                if idx >= 0:
                    ctrl.setCurrentIndex(idx)
                else:
                    # 允许配置中出现未知值
                    ctrl.addItem(txt)
                    ctrl.setCurrentText(txt)
            else:
                if val is None:
                    continue
                ctrl.setText(str(val))

    def _safe_parse_text(self, text: str, typ: str):
        text = (text or "").strip()
        if typ == "int":
            return int(float(text))
        if typ == "float":
            return float(text)
        if typ == "str":
            return text
        # 兜底：允许列表/字典的字面量（例如你未来扩展）
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    def _collect_controls_to_config(self):
        new_cfg = dict(self._config or {})
        for spec in self._detect_param_specs:
            key = spec["key"]
            ctrl = self._param_controls.get(key)
            if ctrl is None:
                continue
            t = spec["type"]
            if t == "bool":
                new_cfg[key] = 1 if ctrl.isChecked() else 0
            elif t == "choice":
                new_cfg[key] = str(ctrl.currentText()).strip()
                # 一些 choice 实际是 int（比如 calibrat_cam_id）
                if key == "calibrat_cam_id":
                    try:
                        new_cfg[key] = int(new_cfg[key])
                    except Exception:
                        pass
            else:
                raw = ctrl.text()
                if raw is None or str(raw).strip() == "":
                    continue
                new_cfg[key] = self._safe_parse_text(raw, t)
        return new_cfg

    def get_relative_files(self, folder_path, tag):
        """
        获取文件夹下的所有文件，并转换为相对路径（例如 './cam_model/xyz.pth'）
        """
        base_folder = folder_path
        relative_files = []

        # 遍历文件夹及子文件夹
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                # 计算相对路径
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_folder)
                # 添加 './' 前缀
                if tag == 10:
                    relative_files.append('cam1/' + relative_path.replace(os.sep, '/'))
                elif tag == 20:
                    relative_files.append('cam2/' + relative_path.replace(os.sep, '/'))
                elif tag == 30:
                    relative_files.append('cam3/' + relative_path.replace(os.sep, '/'))
                elif tag == 40:
                    relative_files.append('cam4/' + relative_path.replace(os.sep, '/'))
                else:
                    relative_files.append(relative_path.replace(os.sep, '/'))
        return relative_files

    #加载默认配置的函数
    def config_default(self):
        default_yaml_path = os.path.join(_REPO_ROOT, 'config', 'config_default.yaml')

        try:
            with open(default_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            QMessageBox.warning(self, "错误", f"找不到默认配置文件：{default_yaml_path}")
            return

        # 用默认配置覆盖当前配置并刷新控件（仅限检测所需项）
        try:
            self._config = config or {}
            self._load_config_to_controls()
            QMessageBox.information(self, "已恢复默认值", "已加载默认检测参数（尚未写入，点击“确认更改”保存）。")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"恢复默认值时出错: {str(e)}")
            return

    def save_config(self):
        # 收集并保存到 config.yaml（仅覆盖检测所需项，其它键保持不变）
        try:
            new_cfg = self._collect_controls_to_config()
            with open(self._config_path, "w", encoding="utf-8") as file:
                yaml.dump(new_cfg, file, default_flow_style=False, indent=4, allow_unicode=True, sort_keys=False)
            self._config = new_cfg
            QMessageBox.information(self, "保存成功", "检测参数已成功保存到 config.yaml。")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存配置文件时出错: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParaWindow()
    window.show()
    sys.exit(app.exec_())