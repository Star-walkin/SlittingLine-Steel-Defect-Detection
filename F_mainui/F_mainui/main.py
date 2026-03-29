from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QLineEdit, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import QRect, Qt, QTimer, QThread, pyqtSignal, QProcess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from mainui import Ui_MainWindow  # 导入pyuic生成的类
from para import ParaWindow
from report_change import ReportWindow
from report_center import ReportCenterWindow
from cls_config import (
    ClsConfigWindow,
    product_combo_entries,
    product_cls_key_from_combo_text,
)
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import time
import subprocess
import json
import serial
import yaml
import numpy as np
import math
from datetime import timedelta, datetime

_AUTH_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "auth.yaml")


def _read_auth_password(role: str) -> str:
    """从 auth.yaml 读取指定角色的密码，读取失败则回退到 '000'。"""
    try:
        with open(_AUTH_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("passwords", {}).get(role, "000"))
    except Exception:
        return "000"

# ---------- 幅宽偏窄判定（工程容差，整体偏宽松）----------
# 「显著偏窄」下阈值 = 设定幅宽 − max(绝对容差mm, 设定×相对比例)，仅当实测低于该阈值才计入预警/报警统计。
# 这样测量噪声、标定误差和正常工艺波动不会一碰就报警。
FUKUAN_NARROW_ABS_MM = 12.0
FUKUAN_NARROW_REL = 0.025
# 历史序列中连续「显著偏窄」采样数达到以下值 → 报警（原 10 帧、零容差；现加长链、带容差）
FUKUAN_ALARM_CONSEC = 16
# 当前序列末尾连续「显著偏窄」采样数达到以下值 → 预警（未达报警）
FUKUAN_WARN_TAIL_STREAK = 6


def fukuan_narrow_threshold_mm(baseline_mm: float) -> float:
    """低于此实测值视为「显著偏窄」（已扣掉允许带）。"""
    if baseline_mm is None or baseline_mm <= 0:
        return float("inf")
    band = max(FUKUAN_NARROW_ABS_MM, float(baseline_mm) * FUKUAN_NARROW_REL)
    return float(baseline_mm) - band


def fukuan_is_significantly_narrow(measured_mm: float, baseline_mm: float) -> bool:
    return measured_mm < fukuan_narrow_threshold_mm(baseline_mm)
# 幅宽波形与右侧状态列（按条带 frame 宽度动态计算，保证右侧留白、与波形区间隙）
FUKUAN_WAVE_HOST_X = 1210
FUKUAN_WAVE_GAP_TO_STATUS = 10
FUKUAN_RIGHT_MARGIN = 50
FUKUAN_STATUS_PANEL_W = 126
FUKUAN_STATUS_TITLE_H = 22
FUKUAN_STATUS_TITLE_PANEL_GAP = 6
FUKUAN_STATUS_PANEL_H = 118
FUKUAN_STRIP_DEFAULT_W = 1761


def _fukuan_layout_metrics(inner_width: int):
    """由条带内容区宽度推算：状态列左缘 sx、状态列宽 sw、波形宿主宽 wave_w。"""
    sw = FUKUAN_STATUS_PANEL_W
    sx = int(inner_width) - FUKUAN_RIGHT_MARGIN - sw
    wave_w = sx - FUKUAN_WAVE_HOST_X - FUKUAN_WAVE_GAP_TO_STATUS
    return sx, sw, max(260, wave_w)


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams


def _apply_app_theme(app: QtWidgets.QApplication) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qss_path = os.path.join(base_dir, "theme.qss")
    try:
        qss = ""
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                qss = f.read()
        app.setStyleSheet(qss)
    except Exception as e:
        print(f"加载主题失败: {e}")

    # 全局字体（尽量兼容中文显示）
    try:
        app.setFont(QFont("Microsoft YaHei UI", 10))
    except Exception:
        pass


class WaveformThread(QThread):
    # data, has_abnormal, detection_system_index, last_measured_mm, tail_narrow_streak
    update_signal = pyqtSignal(list, bool, int, float, int)

    def __init__(self, camid, detection_system_index):
        super(WaveformThread, self).__init__()
        self.data = []
        self.camid = camid
        self.detection_system_index = detection_system_index
        self._is_running = True

        # ---【关键修改】初始化变量 ---
        self.path = None
        self.last_file_path = None
        self.last_mtime = 0

    @staticmethod
    def _analyze_fukuan_series(values, baseline_width):
        """
        返回 (是否存在「连续多帧显著偏窄」报警, 最新实测mm, 末尾连续显著偏窄帧数)。
        显著偏窄：实测 < fukuan_narrow_threshold_mm(设定)，含绝对+相对容差。
        """
        if not values or baseline_width <= 0:
            return False, float("nan"), 0
        th = fukuan_narrow_threshold_mm(float(baseline_width))
        nums = [float(v) for v in values]
        last_mm = nums[-1]
        tail = 0
        for v in reversed(nums):
            if v < th:
                tail += 1
            else:
                break
        count = 0
        has_abnormal = False
        for v in nums:
            if v < th:
                count += 1
                if count >= FUKUAN_ALARM_CONSEC:
                    has_abnormal = True
                    break
            else:
                count = 0
        return has_abnormal, last_mm, tail

    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.fukuan)
        self.timer.start(2000)
        self.exec_()

    def find_folders_with_id0(self, base_path, product_id):
        now = datetime.now()
        today_date = now.strftime('%Y%m%d')
        yesterday_date = (now - timedelta(days=1)).strftime('%Y%m%d')
        folder_path = []
        today_folder_path = os.path.join(base_path, today_date, product_id)
        yesterday_folder_path = os.path.join(base_path, yesterday_date, product_id)

        if os.path.exists(today_folder_path):
            folder_path.append(today_folder_path)
            return folder_path

        if now.hour < 6 and os.path.exists(yesterday_folder_path):
            folder_path.append(today_folder_path)
            return folder_path

        return folder_path

    def read_json_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            return []

    def fukuan(self):
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)

            fukuan_key = f"fukuan_{self.detection_system_index}"
            baseline_width = config0.get(fukuan_key, 0)

            if baseline_width <= 0:
                self.update_signal.emit([], False, self.detection_system_index, float("nan"), 0)
                return

            id = config0['conduct_id']
            found_folders = self.find_folders_with_id0(os.path.join(_REPO_ROOT, 'detect result'), id)

            if len(found_folders) == 0:
                return

            root_path = found_folders[0]
            self.fukuan_path = root_path + "/" + str(self.camid) + "/" + "strip_" + str(
                self.detection_system_index) + "/"
            new_folder_name = "fukuan.json"

            self.path = os.path.join(self.fukuan_path, new_folder_name)

            if not os.path.exists(self.path):
                return

            # ---【新增关键修改：防止空文件报错】---
            if os.path.getsize(self.path) == 0:
                return
            # -----------------------------------

            current_mtime = os.path.getmtime(self.path)

            if self.path != self.last_file_path or current_mtime > self.last_mtime:
                new_data = self.read_json_data(self.path)

                self.last_file_path = self.path
                self.last_mtime = current_mtime

                if new_data != self.data:
                    self.data = new_data
                    has_abnormal, last_mm, tail_streak = self._analyze_fukuan_series(
                        new_data, baseline_width
                    )
                    self.update_signal.emit(
                        new_data, has_abnormal, self.detection_system_index, last_mm, tail_streak
                    )

        except Exception as e:
            print(f"检测系统{self.detection_system_index}读取数据失败: {e}")
            self.update_signal.emit([], False, self.detection_system_index, float("nan"), 0)

    def stop(self):
        self._is_running = False
        self.wait()
class ImageLoaderThread(QThread):
    # x: 宽度坐标(mm)，y: 长度坐标(mm)
    # c_m: 当前长度段索引（用于刷新X轴刻度）
    # pos: 读取到的累计位置计数
    # fukuan_for_point: 该长度段对应的幅宽(mm)，用于宽度方向自适应缩放
    image_loaded = pyqtSignal(float, float, int, int, float, str, int)

    def __init__(self, camid, pos, detection_system_index):
        super(ImageLoaderThread, self).__init__()
        self.camid = camid
        self.position = pos
        self.detection_system_index = detection_system_index
        self._is_running = True

        # 状态记录
        self.all_coordinates = []
        self.last_file_path = None
        self.last_mtime = 0

        # 幅宽文件状态（用于宽度方向自适应缩放）
        self.fukuan_file_path = None
        self.fukuan_data = []
        self.last_fukuan_mtime = 0
        self.last_fukuan_file_path = None
        self.fukuan_fallback = 0.0

    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_coordinates)
        self.timer.start(2000)
        self.exec_()

    def find_folders_with_id(self, base_path, product_id):
        now = datetime.now()
        today_date = now.strftime('%Y%m%d')
        yesterday_date = (now - timedelta(days=1)).strftime('%Y%m%d')
        folder_path = []
        today_folder_path = os.path.join(base_path, today_date, product_id)
        yesterday_folder_path = os.path.join(base_path, yesterday_date, product_id)

        if os.path.exists(today_folder_path):
            folder_path.append(today_folder_path)
            return folder_path

        if now.hour < 6 and os.path.exists(yesterday_folder_path):
            folder_path.append(today_folder_path)
            return folder_path

        return folder_path

    def read_coordinates(self):
        try:
            # 1. 读取配置
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)
                fukuan_key = f"fukuan_{self.detection_system_index}"
                baseline_width = config0.get(fukuan_key, 0)
                if baseline_width <= 0:
                    return
                self.fukuan_fallback = float(baseline_width)

            id = config0['conduct_id']

            # 2. 查找文件夹
            found_folders = self.find_folders_with_id(os.path.join(_REPO_ROOT, 'detect result'), id)
            if len(found_folders) == 0:
                return

            root_path = found_folders[0]
            self.folder_path = root_path + "/" + str(self.camid) + "/" + "strip_" + str(
                self.detection_system_index) + "/"
            new_folder_name = "defect_images"
            self.base_folder = os.path.join(self.folder_path, new_folder_name)

            # 目标坐标文件路径
            coord_file_path = os.path.join(self.folder_path, "image_anomaly_center.json")
            self.fukuan_file_path = os.path.join(self.folder_path, "fukuan.json")

            if not os.path.exists(coord_file_path):
                return

            # ---【新增关键修改：防止空文件报错】---
            if os.path.getsize(coord_file_path) == 0:
                # 文件存在但没内容，直接跳过，等待数据写入
                return
            # -----------------------------------

            # 读取/刷新幅宽文件（允许在坐标文件不变时仍持续增长）
            try:
                if os.path.exists(self.fukuan_file_path) and os.path.getsize(self.fukuan_file_path) > 0:
                    fuk_mtime = os.path.getmtime(self.fukuan_file_path)
                    if (
                        self.fukuan_file_path != self.last_fukuan_file_path
                        or fuk_mtime > self.last_fukuan_mtime
                    ):
                        with open(self.fukuan_file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        # fukuan.json 是 float 列表
                        self.fukuan_data = [float(v) for v in data] if isinstance(data, list) else []
                        self.last_fukuan_mtime = fuk_mtime
                        self.last_fukuan_file_path = self.fukuan_file_path
            except Exception:
                # 幅宽读取失败不影响坐标显示，退回到 fallback
                self.fukuan_data = []

            # 调试打印 (仅在路径变更时显示一次)
            if coord_file_path != self.last_file_path:
                print(f"-------------- 系统{self.detection_system_index} [相机{self.camid}] 路径监控 --------------")
                print(f"👉 锁定文件: ...{str(coord_file_path)[-40:]}")
                print("--------------------------------------------------")

            current_mtime = os.path.getmtime(coord_file_path)
            need_reload = False
            is_new_file = False

            if coord_file_path != self.last_file_path:
                need_reload = True
                is_new_file = True
            elif current_mtime > self.last_mtime:
                need_reload = True
                is_new_file = False

            if need_reload:
                try:
                    with open(coord_file_path, 'r') as file:
                        data = json.load(file)

                    self.all_coordinates = [coord for sublist in data for coord in sublist]

                    # ---【新增关键修改：输出读取状态】---
                    print(
                        f"✅ 系统{self.detection_system_index} [相机{self.camid}] 数据更新! 总坐标数: {len(self.all_coordinates)}")
                    # --------------------------------

                    if is_new_file:
                        self.position = 0
                        print(f"   🔄 新文件，计数器重置")

                    self.last_file_path = coord_file_path
                    self.last_mtime = current_mtime

                except Exception as e:
                    print(f"JSON读取失败: {e}")
                    return

            # 发送数据逻辑
            remaining = len(self.all_coordinates) - self.position
            if remaining > 0:
                read_count = min(10, remaining)
                for i in range(read_count):
                    if self.position < len(self.all_coordinates):
                        x, y = self.all_coordinates[self.position]
                        current_multiple = int(y // 1720)

                        # 根据长度段索引选择对应幅宽：fukuan.json 以“第1段写入”为索引0
                        fukuan_for_point = self.fukuan_fallback
                        idx0 = current_multiple - 1
                        if 0 <= idx0 < len(self.fukuan_data):
                            fukuan_for_point = self.fukuan_data[idx0]
                        elif 0 <= current_multiple < len(self.fukuan_data):
                            fukuan_for_point = self.fukuan_data[current_multiple]

                        self.position += 1
                        self.image_loaded.emit(x, y, current_multiple, self.position, fukuan_for_point, self.base_folder,
                                               self.detection_system_index)

        except Exception as e:
            print(f"缺陷显示线程{self.camid}号相机出错: {e}")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 调用生成的UI
        self.system_count = 3
        self._load_system_count_from_config()
        self.MAX_STRIPS = 4
        self.buttons = [[] for _ in range(self.MAX_STRIPS)] # 存储生成的按钮
        self.buttons2 = [[] for _ in range(self.MAX_STRIPS)]
        self.coordinate_queue = [[] for _ in range(self.MAX_STRIPS)]  # 队列用于存储待处理的坐标
        self.coordinate_queue2 = [[] for _ in range(self.MAX_STRIPS)]  # 队列用于存储待处理的坐标
        self.last_update_time = [time.time() for _ in range(self.MAX_STRIPS)]  # 上次更新的时间
        self.last_update_time2 = [time.time() for _ in range(self.MAX_STRIPS)] # 上次更新的时间
        self.last_multiple =  [-1 for _ in range(self.MAX_STRIPS)]  # 用于存储上一次的倍数
        self.last_multiple2 =  [-1 for _ in range(self.MAX_STRIPS)]
        self.pos = [0 for _ in range(self.MAX_STRIPS)]
        self.pos2 = [0 for _ in range(self.MAX_STRIPS)]
        self.cm = [0 for _ in range(self.MAX_STRIPS)]  # 系统的当前倍数
        self.base_folder = [None for _ in range(self.MAX_STRIPS)]
        self.cm2 = [0 for _ in range(self.MAX_STRIPS)]  # 系统的当前倍数
        self.base_folder2 = [None for _ in range(self.MAX_STRIPS)]
        self.fukuan_mm = [0.0 for _ in range(self.MAX_STRIPS)]   # 宽度方向刻度上限(mm) - 上表面
        self.fukuan_mm2 = [0.0 for _ in range(self.MAX_STRIPS)]  # 宽度方向刻度上限(mm) - 下表面
        # 缺陷点缓存：与波形窗口同时间轴滑动显示
        self.defect_points_up = [[] for _ in range(self.MAX_STRIPS)]
        self.defect_points_down = [[] for _ in range(self.MAX_STRIPS)]
        # 当前波形窗口的绝对长度范围（mm），用于缺陷图同步滑动
        self.wave_window_start_mm = [0.0 for _ in range(self.MAX_STRIPS)]
        self.wave_window_end_mm = [1720.0 for _ in range(self.MAX_STRIPS)]
        self.csharp_process = None
        self.python_process = None
        self.loader_thread =  [None for _ in range(self.MAX_STRIPS)]
        self.loader_thread2 =  [None for _ in range(self.MAX_STRIPS)]
        self.total_data = [[] for _ in range(self.MAX_STRIPS)]
        self.POINTS_TO_DISPLAY = 100
        self.abnormal_status = [False] * self.MAX_STRIPS
        self.fukuan_last_measured = [float("nan")] * self.MAX_STRIPS
        self.fukuan_tail_narrow = [0] * self.MAX_STRIPS
        self.display_end_indices = [0 for _ in range(self.MAX_STRIPS)]
        # 浮点游标：平滑追赶最新数据长度，避免整档跳跃
        self.display_smooth_end = [0.0 for _ in range(self.MAX_STRIPS)]

        self.waveform_threads = [None for _ in range(self.MAX_STRIPS)]  # 幅宽监测线程
        self.figures = [None for _ in range(self.MAX_STRIPS)]  # 波形图
        self.canvases = [None for _ in range(self.MAX_STRIPS)]  # 画布
        self.axes = [None for _ in range(self.MAX_STRIPS)]  # 坐标轴

        self.pushButton_start.clicked.connect(self.button_start_click)#界面上的按钮pushButton_start和自己定义的button_start_click函数连接起来
        self.pushButton_stop.clicked.connect(self.button_stop_click)
        self.pushButton_para.clicked.connect(self.pushButton_para_click)
        self.pushButton_old_report.clicked.connect(self.pushButton_old_report_click)
        self.pushButton_old_report.setToolTip(
            "报告打印与标准维护\n"
            "功能：维护缺陷类别/允收标准/打印优化项，并生成检测报告。\n"
            "适用角色：工艺工程师 / 质检人员（需密码验证）。\n"
            "提示：报告文件定位与浏览请使用右侧「报告生成」入口。"
        )
        self.pushButton_report.clicked.connect(self.pushButton_report_click)
        self.pushButton.clicked.connect(self.baojing_close)
        self.button_exchange.clicked.connect(self.exchangeNEWONE)
        self.para_config01.clicked.connect(self.save_config01)
        self.para_window = None#声明窗口实例变量，初始化为 None 表示窗口尚未创建
        self.report_window = None
        self.report_center_window = None

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_all_plots)
        # 约 30 FPS，配合浮点插值游标，滑动更顺滑
        self.render_timer.start(33)
        self._create_strip4_controls()
        self._init_strip_count_ui()
        self._init_scrollable_strip_layout()
        self._init_external_axis_canvases()
        self.apply_strip_layout(self.system_count)
        self._setup_fukuan_status_panel()

    def _fukuan_status_title_labels(self):
        titles = [self.label_title_15, self.label_title_7, self.label_title_10]
        if hasattr(self, "label_fukuan_status_4"):
            titles.append(self.label_fukuan_status_4)
        return titles

    def _fukuan_strip_content_width(self):
        try:
            w = int(self.strip_frames[0].width())
            if w > 100:
                return w
        except Exception:
            pass
        return FUKUAN_STRIP_DEFAULT_W

    def _fukuan_status_horizontal_metrics(self):
        return _fukuan_layout_metrics(self._fukuan_strip_content_width())

    def _layout_fukuan_waveform_host_width(self):
        """收窄幅宽波形图容器，与右侧状态列留出间隙，避免相互遮挡。"""
        _, _, wave_w = self._fukuan_status_horizontal_metrics()
        for host in self._fukuan_labels():
            if host is None:
                continue
            g = host.geometry()
            host.setGeometry(QRect(FUKUAN_WAVE_HOST_X, g.y(), wave_w, g.height()))

    def _fukuan_status_title_for_strip(self, idx):
        if idx == 0:
            return self.label_title_15
        if idx == 1:
            return self.label_title_7
        if idx == 2:
            return self.label_title_10
        if idx == 3 and hasattr(self, "label_fukuan_status_4"):
            return self.label_fukuan_status_4
        return None

    def _position_fukuan_status_column_for_strip(self, idx):
        """状态标题 + 卡片相对本条「波形图宿主」竖直居中。"""
        hosts = self._fukuan_labels()
        smalls = self._small_fukuan_labels()
        if idx >= len(hosts) or idx >= len(smalls):
            return
        title = self._fukuan_status_title_for_strip(idx)
        if title is None:
            return
        host = hosts[idx]
        small = smalls[idx]
        sx, sw, _ = self._fukuan_status_horizontal_metrics()
        fg = host.geometry()
        cy = fg.y() + fg.height() / 2.0
        block_h = FUKUAN_STATUS_TITLE_H + FUKUAN_STATUS_TITLE_PANEL_GAP + FUKUAN_STATUS_PANEL_H
        title_y = int(round(cy - block_h / 2.0))
        panel_y = title_y + FUKUAN_STATUS_TITLE_H + FUKUAN_STATUS_TITLE_PANEL_GAP
        title.setGeometry(QRect(sx, title_y, sw, FUKUAN_STATUS_TITLE_H))
        small.setGeometry(QRect(sx, panel_y, sw, FUKUAN_STATUS_PANEL_H))

    def _refresh_fukuan_status_layout(self):
        self._layout_fukuan_waveform_host_width()
        for i in range(self.MAX_STRIPS):
            self._position_fukuan_status_column_for_strip(i)
        for w in self._fukuan_status_title_labels():
            w.raise_()
        for lbl in self._small_fukuan_labels():
            lbl.raise_()

    def _setup_fukuan_status_panel(self):
        """右侧幅宽状态：标题 + 多行实测/设定/偏差与分级状态。"""
        title_txt = "幅宽状态"
        for w in self._fukuan_status_title_labels():
            w.setText(title_txt)
            w.setWordWrap(False)
            f = w.font()
            f.setPointSize(10)
            f.setBold(True)
            w.setFont(f)
            w.setStyleSheet("color: #2c3e50; background: transparent;")
        panel_style = (
            "background-color: #f9fafb; border: 1px solid #cfd8dc; "
            "border-radius: 6px; padding: 6px;"
        )
        for lbl in self._small_fukuan_labels():
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            lf = lbl.font()
            lf.setPointSize(9)
            lf.setBold(False)
            lbl.setFont(lf)
            lbl.setStyleSheet(panel_style)
        self._refresh_fukuan_status_layout()

        tip = (
            "幅宽状态说明（含工程容差，偏宽松）：\n"
            f"· 允许带：实测低于「设定−max({FUKUAN_NARROW_ABS_MM:.0f}mm, 设定×{FUKUAN_NARROW_REL:.1%})」才算显著偏窄；\n"
            f"· 报警：记录中曾连续≥{FUKUAN_ALARM_CONSEC}帧显著偏窄；\n"
            f"· 预警：当前末尾连续≥{FUKUAN_WARN_TAIL_STREAK}帧显著偏窄，且未触发报警；\n"
            "· 注意：最近一次显著偏窄，但末尾连续未达预警；\n"
            "· 正常：当前未显著偏窄，且无报警条件。"
        )
        for lbl in self._small_fukuan_labels():
            lbl.setToolTip(tip)

    def _format_fukuan_status_richtext(
        self, baseline, last_mm, tail_streak, has_alarm, active, has_waveform_points
    ):
        """生成右侧状态面板 HTML。"""
        if not active:
            return (
                '<span style="color:#7f8c8d;font-size:11px;">'
                "<b>未激活</b><br/>本通道幅宽为 0</span>"
            )
        if not has_waveform_points or last_mm is None or (
            isinstance(last_mm, float) and math.isnan(last_mm)
        ):
            return (
                f'<span style="color:#7f8c8d;font-size:11px;">'
                f"<b>等待数据</b><br/>设定 <b>{baseline:.0f}</b> mm</span>"
            )
        th = fukuan_narrow_threshold_mm(baseline)
        delta = last_mm - baseline
        delta_txt = f"{delta:+.1f}"
        if baseline:
            pct_txt = f" ({delta / baseline * 100.0:+.1f}%)"
        else:
            pct_txt = ""
        sub_lines = []
        narrow = fukuan_is_significantly_narrow(last_mm, baseline)
        if has_alarm:
            tier = "报警"
            tier_color = "#c0392b"
            sub_lines.append(
                f"历史曾连续≥{FUKUAN_ALARM_CONSEC}帧低于允许下限({th:.0f}mm)"
            )
        elif tail_streak >= FUKUAN_WARN_TAIL_STREAK and narrow:
            tier = "预警"
            tier_color = "#d35400"
            sub_lines.append(
                f"末尾已连续 {tail_streak} 帧低于允许下限({th:.0f}mm)"
            )
        elif narrow:
            tier = "注意"
            tier_color = "#b7950b"
            sub_lines.append(f"实测低于允许下限({th:.0f}mm)")
        else:
            tier = "正常"
            tier_color = "#1e8449"
        extra = f"<br/><span style='font-size:9px;color:#566573'>{sub_lines[0]}</span>" if sub_lines else ""
        return (
            f'<span style="color:{tier_color};font-size:12px;"><b>{tier}</b></span><br/>'
            f'<span style="color:#2c3e50;font-size:10px;">'
            f"实测 <b>{last_mm:.1f}</b> mm<br/>"
            f"设定 {baseline:.0f} mm<br/>"
            f"偏差 {delta_txt}{pct_txt} mm</span>"
            f"{extra}"
        )

    def _load_system_count_from_config(self):
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file) or {}
            raw_count = int(config0.get("strip_count", 3))
            self.system_count = min(4, max(2, raw_count))
        except Exception:
            self.system_count = 3
        if hasattr(self, "strip_count_combo"):
            self.strip_count_combo.blockSignals(True)
            self.strip_count_combo.setCurrentText(str(self.system_count))
            self.strip_count_combo.blockSignals(False)

    def _init_strip_count_ui(self):
        # 与生产卡号/幅宽配置放在同一配置区
        # 调整顶部输入区，为第4条幅宽输入留出位置
        input_font = QFont("Arial", 14)
        self.fukuan_1.setFont(input_font)
        self.fukuan_2.setFont(input_font)
        self.fukuan_3.setFont(input_font)
        self.fukuan_1.setPlaceholderText("幅宽1(mm)")
        self.fukuan_2.setPlaceholderText("幅宽2(mm)")
        self.fukuan_3.setPlaceholderText("幅宽3(mm)")
        # 左上角：固定文字「带钢产品号」为 QLabel，输入在下一行，避免被误认为整格都是输入框
        self.label_ID.setText("带钢产品号")
        self.label_ID.setGeometry(QRect(10, 10, 132, 31))
        self.label_ID.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_ID.setStyleSheet(
            "QLabel { background: transparent; color: #1a1a1a; border: none; }"
        )
        self.label_ID.setToolTip("固定说明文字；请在下方输入框填写带钢产品号（生产卡号）。")
        self.conduct_id.setGeometry(QRect(10, 55, 221, 41))
        self.conduct_id.setPlaceholderText("请输入带钢产品号")

        # 隐藏 mainui 生成的 QLineEdit（保留对象避免潜在引用报错）
        self.product_cls.hide()

        # 标签改为“产品型号”并附 Tooltip，说明各字段的职责边界
        # 与「带钢条数」右缘留约 8px，并与右侧「确认」拉开间距（确认按钮在 main 中右移）
        self.label_ID_7.setGeometry(QRect(668, 10, 70, 31))
        self.label_ID_7.setText("产品型号")
        self.label_ID_7.setToolTip(
            "【产品型号】\n"
            "下拉为「显示名称 [编号]」；保存到 config0 的仍为数字编号（对应 data{N}）。\n"
            "显示名称在「类别配置」中维护（rptcfg 的 product_cls_names）。\n"
            "允收矩阵在类别配置中编辑，经 make_standard 生成 table.json。"
        )

        # 用 QComboBox 替换原 QLineEdit，选项来自 rptcfg 中已有的 data{N} 键
        self.product_cls_combo = QComboBox(self.frame)
        self.product_cls_combo.setGeometry(QRect(668, 55, 200, 41))
        self.product_cls_combo.setFont(input_font)
        self.product_cls_combo.setEditable(True)
        self.product_cls_combo.setToolTip(
            "从列表选择可看到显示名称；也可直接输入编号。\n"
            "名称在「类别配置」中设置；底层仍用 data{编号}。"
        )
        self._refresh_product_cls_combo()
        self._sync_product_cls_combo_from_config0()

        # 「类别配置」与「产品型号」标签同排，下拉在下一行
        self.btn_cls_config = QPushButton("类别配置", self.frame)
        self.btn_cls_config.setGeometry(QRect(748, 10, 100, 31))
        font_btn = QFont("Arial", 11)
        font_btn.setBold(True)
        self.btn_cls_config.setFont(font_btn)
        self.btn_cls_config.setStyleSheet(
            "QPushButton { background-color: #E8F5E9; color: #2e7d32;"
            " border: 1px solid #a5d6a7; border-radius: 4px;"
            " padding-left: 16px; padding-right: 14px; }"
            "QPushButton:hover { background-color: #C8E6C9; }"
        )
        self.btn_cls_config.setToolTip(
            "打开类别配置窗口。\n"
            "在此维护产品型号（data{N}）、缺陷类别名称和允收矩阵，\n"
            "保存后可生成 table.json 供报告判定使用。"
        )
        self.btn_cls_config.clicked.connect(self._open_cls_config)
        self._cls_config_window = None

        self.fukuan_4 = QLineEdit(self.frame)
        self.fukuan_4.setGeometry(QRect(580, 55, 120, 41))
        self.fukuan_4.setFont(input_font)
        self.fukuan_4.setPlaceholderText("幅宽4(mm)")

        self.strip_count_label = QLabel("带钢条数", self.frame)
        self.strip_count_label.setGeometry(QRect(500, 10, 85, 31))
        self.strip_count_label.setStyleSheet("font: 16px 'Arial'; font-weight: bold;")
        self.strip_count_combo = QComboBox(self.frame)
        self.strip_count_combo.setGeometry(QRect(590, 10, 70, 31))
        self.strip_count_combo.addItems(["2", "3", "4"])
        self.strip_count_combo.blockSignals(True)
        self.strip_count_combo.setCurrentText(str(self.system_count))
        self.strip_count_combo.blockSignals(False)
        self.strip_count_combo.currentIndexChanged.connect(self.apply_strip_count_preview)

        # 与「类别配置」右缘留白（748+100=848）
        self.para_config01.setGeometry(QRect(878, 10, 75, 31))
        self.button_exchange.setGeometry(QRect(878, 60, 75, 31))

    def _current_product_cls_key_from_combo(self) -> str:
        """当前选中的型号编号（与 data{N}、config0 一致）。"""
        ix = self.product_cls_combo.currentIndex()
        if ix >= 0:
            d = self.product_cls_combo.itemData(ix)
            if d is not None and str(d).strip() != "":
                return str(d).strip()
        return product_cls_key_from_combo_text(self.product_cls_combo.currentText())

    def _sync_product_cls_combo_from_config0(self):
        """启动时根据 config0.product_cls 选中下拉项（展示名 + 编号）。"""
        try:
            with open(
                os.path.join(_REPO_ROOT, "config", "config0.yaml"),
                "r",
                encoding="utf-8",
            ) as f:
                c = yaml.safe_load(f) or {}
            key = str(c.get("product_cls") or "").strip()
            if not key:
                return
            self.product_cls_combo.blockSignals(True)
            for i in range(self.product_cls_combo.count()):
                if str(self.product_cls_combo.itemData(i)) == key:
                    self.product_cls_combo.setCurrentIndex(i)
                    self.product_cls_combo.blockSignals(False)
                    return
            self.product_cls_combo.setCurrentIndex(-1)
            self.product_cls_combo.setEditText(key)
            self.product_cls_combo.blockSignals(False)
        except Exception:
            pass

    def _refresh_product_cls_combo(self):
        """从 rptcfg 重新加载型号列表（含显示名称）。"""
        prev = self._current_product_cls_key_from_combo() if hasattr(self, "product_cls_combo") else ""
        entries = product_combo_entries()
        self.product_cls_combo.blockSignals(True)
        self.product_cls_combo.clear()
        for key, label in entries:
            self.product_cls_combo.addItem(label, key)
        if prev:
            for i in range(self.product_cls_combo.count()):
                if str(self.product_cls_combo.itemData(i)) == str(prev):
                    self.product_cls_combo.setCurrentIndex(i)
                    break
            else:
                self.product_cls_combo.setCurrentIndex(-1)
                self.product_cls_combo.setEditText(str(prev))
        self.product_cls_combo.blockSignals(False)

    def _open_cls_config(self):
        """打开类别配置窗口，关闭后刚新型号列表。"""
        if self._cls_config_window is None or not self._cls_config_window.isVisible():
            self._cls_config_window = ClsConfigWindow(self)
            self._cls_config_window.finished.connect(self._on_cls_config_closed)
        self._cls_config_window.show()
        self._cls_config_window.raise_()
        self._cls_config_window.activateWindow()

    def _on_cls_config_closed(self):
        """类别配置窗口关闭时刷新型号下拉列表。"""
        self._refresh_product_cls_combo()

    def apply_strip_count_preview(self, *_args):
        """切换带钢条数时立即刷新幅宽输入框与下方带钢显示区；不写 config0（检测前须点「确认」保存）。"""
        try:
            value = self.strip_count_combo.currentText()
            n = min(4, max(2, int(value)))
        except (ValueError, TypeError):
            return
        if n == self.system_count:
            return
        self.system_count = n
        self.apply_strip_layout(n)
        try:
            self.statusBar().showMessage(
                f"已切换为 {n} 条带钢显示（开始检测前请点击「确认」保存到配置）",
                5000,
            )
        except Exception:
            pass
        print(f"[预览] 带钢条数已切换为 {n} 条（尚未写入 config0）")

    def apply_strip_count_from_ui(self):
        """将当前条数写入 config0 并刷新布局（点「确认」时调用）。"""
        try:
            value = self.strip_count_combo.currentText()
            self.system_count = min(4, max(2, int(value)))
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file) or {}
            config0["strip_count"] = self.system_count
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'w', encoding='utf-8') as file:
                yaml.dump(config0, file, allow_unicode=True)
            self.apply_strip_layout(self.system_count)
            print(f"带钢条数已写入配置并刷新为 {self.system_count} 条")
        except Exception as e:
            QMessageBox.warning(self, "设置失败", f"条数更新失败: {e}")

    def _create_strip4_controls(self):
        # 第4条完整控件（与前三套对应）
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")

        self.label_up_show_4 = QtWidgets.QLabel(self.frame_6)
        self.label_up_show_4.setGeometry(QtCore.QRect(50, 10, 430, 240))
        self.label_up_show_4.setStyleSheet("background-color: white; border: 1px solid black;")
        self.label_up_show_4.setText("")

        self.label_down_show_4 = QtWidgets.QLabel(self.frame_6)
        self.label_down_show_4.setGeometry(QtCore.QRect(630, 10, 430, 240))
        self.label_down_show_4.setStyleSheet("background-color: white; border: 1px solid black;")
        self.label_down_show_4.setText("")

        self.label_up_pic4_1 = QtWidgets.QLabel(self.frame_6)
        self.label_up_pic4_1.setGeometry(QtCore.QRect(490, 10, 110, 110))
        self.label_up_pic4_1.setStyleSheet("border: 1px solid black;")
        self.label_up_pic4_1.setText("")
        self.label_up_pic4_2 = QtWidgets.QLabel(self.frame_6)
        self.label_up_pic4_2.setGeometry(QtCore.QRect(490, 140, 110, 110))
        self.label_up_pic4_2.setStyleSheet("border: 1px solid black;")
        self.label_up_pic4_2.setText("")

        self.label_down_pic4_1 = QtWidgets.QLabel(self.frame_6)
        self.label_down_pic4_1.setGeometry(QtCore.QRect(1070, 10, 110, 110))
        self.label_down_pic4_1.setStyleSheet("border: 1px solid black;")
        self.label_down_pic4_1.setText("")
        self.label_down_pic4_2 = QtWidgets.QLabel(self.frame_6)
        self.label_down_pic4_2.setGeometry(QtCore.QRect(1070, 140, 110, 110))
        self.label_down_pic4_2.setStyleSheet("border: 1px solid black;")
        self.label_down_pic4_2.setText("")

        sx4, sw4, wave_w4 = _fukuan_layout_metrics(FUKUAN_STRIP_DEFAULT_W)
        self.label_fukuan_4 = QtWidgets.QLabel(self.frame_6)
        self.label_fukuan_4.setGeometry(
            QtCore.QRect(FUKUAN_WAVE_HOST_X, 10, wave_w4, 240)
        )
        self.label_fukuan_4.setStyleSheet("background-color: white; border: 1px solid black;")
        self.label_fukuan_4.setText("")

        self.label_fukuan_status_4 = QtWidgets.QLabel(self.frame_6)
        self.label_fukuan_status_4.setGeometry(QtCore.QRect(sx4, 112, sw4, 22))
        self.label_fukuan_status_4.setText("幅宽状态")
        self.label_fukuan_status_4.setFont(QFont("Arial", 11, QFont.Bold))

        self.small_fukuan_4 = QtWidgets.QLabel(self.frame_6)
        self.small_fukuan_4.setGeometry(QtCore.QRect(1670, 160, 71, 41))
        self.small_fukuan_4.setStyleSheet("color:green; font: 16pt 'Arial'; font-weight: bold;")
        self.small_fukuan_4.setText("正常")

        self.label_strip4_title = QtWidgets.QLabel(self.frame_6)
        self.label_strip4_title.setGeometry(QtCore.QRect(10, 80, 30, 110))
        self.label_strip4_title.setText("钢\n带\n4")
        self.label_strip4_title.setStyleSheet("font: 14pt 'Arial'; font-weight: bold;")

    def _init_scrollable_strip_layout(self):
        # 将条带显示区域接入滚动容器，便于4条时扩展
        self.strip_frames = [self.frame_3, self.frame_4, self.frame_5, self.frame_6]
        self.strip_scroll_area = QtWidgets.QScrollArea(self.centralwidget)
        self.strip_scroll_area.setGeometry(QRect(0, 125, 1761, 900))
        self.strip_scroll_area.setWidgetResizable(True)
        self.strip_scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.strip_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.strip_scroll_content = QWidget()
        self.strip_scroll_area.setWidget(self.strip_scroll_content)

        # 重新挂载原有三个frame到滚动内容区
        for frame in self.strip_frames:
            frame.setParent(self.strip_scroll_content)

        self.frame_6.hide()

        # 关闭旧版固定图例
        for old_name in ("label_title_8", "label_title_9", "label_title_11"):
            if hasattr(self, old_name):
                getattr(self, old_name).hide()

    def apply_strip_layout(self, count):
        visible_count = min(max(2, int(count)), 4)
        if hasattr(self, "fukuan_4"):
            self.fukuan_4.setVisible(visible_count == 4)
        # 幅宽输入框数量与条数同步：2条显示前2个，3条显示前3个，4条显示前4个
        self.fukuan_1.setVisible(True)
        self.fukuan_2.setVisible(True)
        self.fukuan_3.setVisible(visible_count >= 3)
        if hasattr(self, "fukuan_4"):
            self.fukuan_4.setVisible(visible_count >= 4)
        row_h = 306
        row_gap = 16
        # 第一行下轴区域在视觉上最容易贴边/被压线，给第一行后额外留白
        first_row_extra_gap = 25
        y = 10
        shown = min(visible_count, 4)

        for idx, frame in enumerate(self.strip_frames):
            if idx < shown:
                frame.show()
                # 带钢1在 ui 原始文件中的显示框 y 起点更靠下（50），需要更高容器避免下轴数字被裁切
                frame_h = 336 if idx == 0 else 296
                frame.setGeometry(QRect(1, y, 1761, frame_h))
                self._sync_external_axis_geometry(idx)
                self.up_axis_left[idx].show()
                self.up_axis_bottom[idx].show()
                self.down_axis_left[idx].show()
                self.down_axis_bottom[idx].show()
                self.up_width_titles[idx].show()
                self.down_width_titles[idx].show()
                self.up_len_titles[idx].show()
                self.down_len_titles[idx].show()
                if idx == 0:
                    y += frame_h + row_gap + first_row_extra_gap
                else:
                    y += frame_h + row_gap
            else:
                frame.hide()
                self.up_axis_left[idx].hide()
                self.up_axis_bottom[idx].hide()
                self.down_axis_left[idx].hide()
                self.down_axis_bottom[idx].hide()
                self.up_width_titles[idx].hide()
                self.down_width_titles[idx].hide()
                self.up_len_titles[idx].hide()
                self.down_len_titles[idx].hide()

        self.strip_scroll_content.setMinimumSize(1761, max(900, y + 20))
        if hasattr(self, "label_title_15"):
            self._refresh_fukuan_status_layout()

    def _init_external_axis_canvases(self):
        # 将坐标轴放在显示框外（左侧 + 下侧）
        self.up_axis_left = []
        self.up_axis_bottom = []
        self.down_axis_left = []
        self.down_axis_bottom = []
        self.up_width_titles = []
        self.down_width_titles = []
        self.up_len_titles = []
        self.down_len_titles = []
        for frame in self.strip_frames:
            axl_up = QLabel(frame)
            axl_up.setGeometry(QRect(8, 10, 40, 240))
            axl_up.setStyleSheet("background-color: transparent;")
            self.up_axis_left.append(axl_up)

            axb_up = QLabel(frame)
            axb_up.setGeometry(QRect(50, 252, 430, 18))
            axb_up.setStyleSheet("background-color: transparent;")
            self.up_axis_bottom.append(axb_up)

            axl_down = QLabel(frame)
            axl_down.setGeometry(QRect(588, 10, 40, 240))
            axl_down.setStyleSheet("background-color: transparent;")
            self.down_axis_left.append(axl_down)

            axb_down = QLabel(frame)
            axb_down.setGeometry(QRect(630, 252, 430, 18))
            axb_down.setStyleSheet("background-color: transparent;")
            self.down_axis_bottom.append(axb_down)

            # 轴标题（补回“长度(m)”与“宽度(mm)”）
            tw_up = QLabel("宽度(mm)", frame)
            tw_up.setStyleSheet("font: 9pt 'Arial'; color: #1e2a39; background-color: transparent;")
            tw_up.setAlignment(Qt.AlignCenter)
            self.up_width_titles.append(tw_up)

            tw_down = QLabel("宽度(mm)", frame)
            tw_down.setStyleSheet("font: 9pt 'Arial'; color: #1e2a39; background-color: transparent;")
            tw_down.setAlignment(Qt.AlignCenter)
            self.down_width_titles.append(tw_down)

            tl_up = QLabel("长度 (m)", frame)
            tl_up.setStyleSheet("font: 8pt 'Arial'; color: #1e2a39; background-color: transparent;")
            tl_up.setAlignment(Qt.AlignCenter)
            self.up_len_titles.append(tl_up)

            tl_down = QLabel("长度 (m)", frame)
            tl_down.setStyleSheet("font: 8pt 'Arial'; color: #1e2a39; background-color: transparent;")
            tl_down.setAlignment(Qt.AlignCenter)
            self.down_len_titles.append(tl_down)

    def _sync_external_axis_geometry(self, idx):
        # 外轴几何与显示框动态对齐，避免首条/第四条出现偏移
        up_rect = self._up_show_labels()[idx].geometry()
        down_rect = self._down_show_labels()[idx].geometry()

        # 让外轴边线与显示框边线重合：34
        # 左轴右边线 == 显示框左边线；下轴上边线 == 显示框下边线
        self.up_axis_left[idx].setGeometry(QRect(max(0, up_rect.x() - 39), up_rect.y(), 40, up_rect.height()))
        self.up_axis_bottom[idx].setGeometry(QRect(up_rect.x(), up_rect.y() + up_rect.height() - 1, up_rect.width(), 20))

        self.down_axis_left[idx].setGeometry(QRect(max(0, down_rect.x() - 39), down_rect.y(), 40, down_rect.height()))
        self.down_axis_bottom[idx].setGeometry(QRect(down_rect.x(), down_rect.y() + down_rect.height() - 1, down_rect.width(), 20))

        # 轴标题位置（紧贴外轴，避免遮挡）
        # 让标题与刻度分层：宽度标题在左轴上方；长度标题在下轴数字下方
        self.up_width_titles[idx].setGeometry(QRect(max(0, up_rect.x() - 62), max(0, up_rect.y() - 18), 58, 14))
        self.down_width_titles[idx].setGeometry(QRect(max(0, down_rect.x() - 62), max(0, down_rect.y() - 18), 58, 14))
        self.up_len_titles[idx].setGeometry(QRect(up_rect.x() + up_rect.width() // 2 - 46, up_rect.y() + up_rect.height() + 22, 92, 14))
        self.down_len_titles[idx].setGeometry(QRect(down_rect.x() + down_rect.width() // 2 - 46, down_rect.y() + down_rect.height() + 22, 92, 14))
        # 提升层级，避免被同层控件覆盖
        self.up_axis_left[idx].raise_()
        self.up_axis_bottom[idx].raise_()
        self.down_axis_left[idx].raise_()
        self.down_axis_bottom[idx].raise_()
        self.up_width_titles[idx].raise_()
        self.down_width_titles[idx].raise_()
        self.up_len_titles[idx].raise_()
        self.down_len_titles[idx].raise_()

    def _visible_count(self):
        return min(max(2, int(self.system_count)), self.MAX_STRIPS)

    def _fukuan_labels(self):
        return [self.label_fukuan_1, self.label_fukuan_2, self.label_fukuan_3, self.label_fukuan_4]

    def _small_fukuan_labels(self):
        return [self.small_fukuan_1, self.small_fukuan_2, self.small_fukuan_3, self.small_fukuan_4]

    def _up_show_labels(self):
        return [self.label_up_show_1, self.label_up_show_2, self.label_up_show_3, self.label_up_show_4]

    def _down_show_labels(self):
        return [self.label_down_show_1, self.label_down_show_2, self.label_down_show_3, self.label_down_show_4]

    def _up_realtime_labels(self):
        return [self.label_up_pic1_1, self.label_up_pic2_1, self.label_up_pic3_1, self.label_up_pic4_1]

    def _down_realtime_labels(self):
        return [self.label_down_pic1_1, self.label_down_pic2_1, self.label_down_pic3_1, self.label_down_pic4_1]

    def _up_click_labels(self):
        return [self.label_up_pic1_2, self.label_up_pic2_2, self.label_up_pic3_2, self.label_up_pic4_2]

    def _down_click_labels(self):
        return [self.label_down_pic1_2, self.label_down_pic2_2, self.label_down_pic3_2, self.label_down_pic4_2]

    def exchangeNEWONE(self):
        self.conduct_id.clear()
        self.fukuan_1.clear()  # 清空第一个检测系统幅宽输入框
        self.fukuan_2.clear()  # 清空第二个检测系统幅宽输入框
        self.fukuan_3.clear()  # 清空第三个检测系统幅宽输入框
        self.fukuan_4.clear()  # 清空第四个检测系统幅宽输入框
        # product_cls 现为 QComboBox；换卷时保留型号选项，仅清空用户输入的自由文本
        if hasattr(self, 'product_cls_combo'):
            self.product_cls_combo.clearEditText()
        else:
            self.product_cls.clear()

        # 重置配置文件
        data = {
            'conduct_id': '',
            'strip_count': self.system_count,
            'fukuan_1': 0.0,
            'fukuan_2': 0.0,
            'fukuan_3': 0.0,
            'fukuan_4': 0.0,
            'product_cls': ''
        }
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'w', encoding='utf-8') as file:
                yaml.dump(data, file, allow_unicode=True)

            # 重置位置计数器
            self.pos = [0 for _ in range(self.MAX_STRIPS)]
            self.pos2 = [0 for _ in range(self.MAX_STRIPS)]

            QMessageBox.information(self, "输入信息",
                                    "已清空配置，请输入：\n"
                                    "• 带钢产品号\n"
                                    "• 需要的检测系统幅宽（不需要的设为0）\n"
                                    "• 产品型号")
        except Exception as e:
            QMessageBox.warning(self, "重置失败", f"重置配置文件时出错: {str(e)}")

    def save_config01(self):#通过界面输入创建configu文件并保存
        try:
            print("开始获取配置数据...")
            self.apply_strip_count_from_ui()

            # 获取所有输入数据
            conduct_id = self.conduct_id.text().strip()
            fukuan_1_text = self.fukuan_1.text().strip()
            fukuan_2_text = self.fukuan_2.text().strip()
            fukuan_3_text = self.fukuan_3.text().strip()
            fukuan_4_text = self.fukuan_4.text().strip()
            product_cls = (
                self._current_product_cls_key_from_combo()
                if hasattr(self, "product_cls_combo")
                else self.product_cls.text().strip()
            )

            print(
                f"输入数据 - 卡号:'{conduct_id}', 幅宽1:'{fukuan_1_text}', 幅宽2:'{fukuan_2_text}', 幅宽3:'{fukuan_3_text}', 类别:'{product_cls}'")

            # 验证必填字段
            if not conduct_id:
                QMessageBox.warning(self, "输入错误", "带钢产品号不能为空！")
                return

            if not product_cls:
                QMessageBox.warning(self, "输入错误", "产品型号不能为空！")
                return

            # 安全转换数值
            fukuan_1 = float(fukuan_1_text) if fukuan_1_text else 0.0
            fukuan_2 = float(fukuan_2_text) if fukuan_2_text else 0.0
            fukuan_3 = float(fukuan_3_text) if fukuan_3_text else 0.0
            fukuan_4 = float(fukuan_4_text) if fukuan_4_text else 0.0

            print(f"转换后数值 - 幅宽1:{fukuan_1}, 幅宽2:{fukuan_2}, 幅宽3:{fukuan_3}")

            # 创建字典
            data = {
                'conduct_id': conduct_id,
                'strip_count': self.system_count,
                'fukuan_1': fukuan_1,
                'fukuan_2': fukuan_2,
                'fukuan_3': fukuan_3,
                'fukuan_4': fukuan_4,
                'fukuan_list': [fukuan_1, fukuan_2, fukuan_3, fukuan_4][:self.system_count],
                'product_cls': product_cls
            }
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'w', encoding='utf-8') as file:
                yaml.dump(data, file, allow_unicode=True)

            print("配置保存成功！")

            # 校验 rptcfg 中是否存在对应的允收矩阵，缺失则非阻塞提示
            try:
                rptcfg_path = os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml')
                with open(rptcfg_path, 'r', encoding='utf-8') as _rf:
                    import yaml as _yaml
                    _rpt = _yaml.safe_load(_rf) or {}
                if f'data{product_cls}' not in _rpt:
                    QMessageBox.warning(
                        self, "允收标准缺失",
                        f"产品型号 {product_cls} 在 rptcfg.yaml 中没有对应的允收矩阵（data{product_cls}）。\n"
                        "请在「类别配置」中添加该型号并保存矩阵，否则报告判定可能异常。"
                    )
                else:
                    # 同步 rptcfg.product_cls，使二者保持一致
                    from cls_config import _rptcfg_set
                    _rptcfg_set('product_cls', product_cls)
                    print(f'rptcfg.product_cls 已同步为 {product_cls}')
            except Exception as _ve:
                print(f'rptcfg 校验时发生异常: {_ve}')

            QMessageBox.information(self, "保存成功", "配置文件已成功保存。")

        except ValueError as ve:
            error_msg = f"数值转换错误: {str(ve)}\n请确保幅宽输入框中只包含数字！"
            print(f"ValueError: {ve}")
            QMessageBox.warning(self, "输入错误", error_msg)

        except FileNotFoundError as fe:
            error_msg = f"文件路径错误: {str(fe)}\n请检查配置文件路径是否正确！"
            print(f"FileNotFoundError: {fe}")
            QMessageBox.critical(self, "文件错误", error_msg)

        except PermissionError as pe:
            error_msg = f"权限不足: {str(pe)}\n请以管理员身份运行程序！"
            print(f"PermissionError: {pe}")
            QMessageBox.critical(self, "权限错误", error_msg)

        except Exception as e:
            error_msg = f"未知错误: {str(e)}\n请联系技术支持！"
            print(f"Exception: {e}")
            print(f"异常类型: {type(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            QMessageBox.critical(self, "系统错误", error_msg)


    def baojing_close(self):
        ser = serial.Serial('COM4', 9600, timeout=1)
        hex_data = [0xa0, 0x01, 0x00, 0xa1]
        ser.write(hex_data)
        ser.close()
        #self.baojing_state.setText("正常")
        #self.baojing_state.setStyleSheet("color: green;")

    def pushButton_old_report_click(self):
        password, ok = QInputDialog.getText(
            self, '密码验证', '报告打印与标准维护（工艺/质检专用）\n请输入密码:',
            QLineEdit.Password
        )
        if ok:
            if password == _read_auth_password("standard_report"):
                if self.report_window is None:
                    self.report_window = ReportWindow()
                self.report_window.show()
                self.report_window.raise_()
                self.report_window.activateWindow()
            else:
                QMessageBox.warning(self, '密码错误', '请输入正确的密码！', QMessageBox.Ok)

    def pushButton_report_click(self):
        # 报告中心：选择现有检测结果（日期/卡号/钢带）并生成/查看/修改报告
        if self.report_center_window is None:
            self.report_center_window = ReportCenterWindow()
        self.report_center_window.show()
        self.report_center_window.raise_()
        self.report_center_window.activateWindow()


    def pushButton_para_click(self):
        password, ok = QInputDialog.getText(self, '密码验证', '请输入密码:', QLineEdit.Password)
        if ok:
            if password == _read_auth_password("parameter_settings"):
                if self.para_window is None:
                    self.para_window = ParaWindow()
                self.para_window.show()
                self.para_window.raise_()
                self.para_window.activateWindow()
            else:
                QMessageBox.warning(self, '密码错误', '请输入正确的密码！', QMessageBox.Ok)

    def closeEvent(self, event):
        self.button_stop_click()#在窗口关闭时自动调用 button_stop_click()方法执行清理操作
        super().closeEvent(event)

    def button_stop_click(self):
        # 1. 停止界面上的定时器
        self.render_timer.stop()

        # 2. 停止所有子线程
        # (注意：直接 terminate 线程比较暴力，最好是在线程里写一个 stop() 方法来修改标志位，但这里先沿用你的逻辑)
        for i in range(self._visible_count()):
            if self.waveform_threads[i] and self.waveform_threads[i].isRunning():
                self.waveform_threads[i].terminate()
                self.waveform_threads[i].wait()

        for i in range(self._visible_count()):
            if self.loader_thread[i] and self.loader_thread[i].isRunning():
                self.loader_thread[i].terminate()
                self.loader_thread[i].wait()
            if self.loader_thread2[i] and self.loader_thread2[i].isRunning():
                self.loader_thread2[i].terminate()
                self.loader_thread2[i].wait()

        # 3. 停止外部进程 (修复了 poll 报错的问题)
        try:
            # 停止 C# 进程
            if self.csharp_process:
                # QProcess 使用 state() != QProcess.NotRunning 来判断是否在运行
                if self.csharp_process.state() != QProcess.NotRunning:
                    print("正在停止 C# 程序...")
                    self.csharp_process.terminate()  # 发送终止信号
                    # 等待最多1秒，如果还在运行则强制杀死
                    if not self.csharp_process.waitForFinished(1000):
                        print("C# 程序未响应，强制杀死...")
                        self.csharp_process.kill()

            # 停止 Python 进程
            if self.python_process:
                if self.python_process.state() != QProcess.NotRunning:
                    print("正在停止 Python 图像处理程序...")
                    self.python_process.terminate()
                    if not self.python_process.waitForFinished(1000):
                        print("Python 程序未响应，强制杀死...")
                        self.python_process.kill()

            # 不需要再调用 terminate_processes 了，上面已经处理完了

            # 更新界面状态
            self.run_state.setText("暂停")
            self.run_state.setStyleSheet("color: red;")
            print("所有进程已停止")

        except Exception as e:
            print(f"停止进程发生错误：{e}")

    def terminate_processes(self):
        """
        这个函数用于窗口关闭事件 (closeEvent) 中的强制清理
        """
        try:
            # C# 进程清理
            if self.csharp_process:
                if self.csharp_process.state() != QProcess.NotRunning:
                    self.csharp_process.kill()  # 直接强制杀死
                    self.csharp_process.waitForFinished(500)

            # Python 进程清理
            if self.python_process:
                if self.python_process.state() != QProcess.NotRunning:
                    self.python_process.kill()
                    self.python_process.waitForFinished(500)

        except Exception as e:
            print(f"强制清理进程出错: {e}")



    def  button_start_click(self):
        self.start_programs()

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        camid = config["camrea_id_up_cls"]
        camid2 = config["camrea_id_down_cls"]

        with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
            config0 = yaml.safe_load(file)

            # 为三个检测系统分别创建上下表面的缺陷检测线程
        self._load_system_count_from_config()
        threads = []
        for i in range(self._visible_count()):
            detection_system_index = i + 1
            fukuan_key = f"fukuan_{detection_system_index}"
            baseline_width = config0.get(fukuan_key, 0)

            if baseline_width > 0:
                # 创建线程但不立即启动
                loader_thread = ImageLoaderThread(camid, self.pos[i], detection_system_index)
                loader_thread.image_loaded.connect(self.update_coordinates)

                loader_thread2 = ImageLoaderThread(camid2, self.pos2[i], detection_system_index)
                loader_thread2.image_loaded.connect(self.update_coordinates2)

                threads.append((loader_thread, loader_thread2, detection_system_index))

        # 批量启动所有缺陷检测线程
        for loader_thread, loader_thread2, system_index in threads:
            loader_thread.start()
            loader_thread2.start()
            self.loader_thread[system_index - 1] = loader_thread
            self.loader_thread2[system_index - 1] = loader_thread2
            print(f"检测系统{system_index}缺陷检测线程已启动")
            # 然后启动幅宽监测
        self.render_timer.start(33)
        self.showfukuan()


    def showfukuan(self):
        try:
            self._refresh_fukuan_status_layout()
            # 获取三个幅宽显示区域的控件
            fukuan_labels = self._fukuan_labels()
            small_fukuan_labels = self._small_fukuan_labels()

            # 读取配置
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)

            with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            calibrat_cam_id = config["calibrat_cam_id"]

            self._load_system_count_from_config()
            for i in range(self._visible_count()):
                detection_system_index = i + 1
                fukuan_key = f"fukuan_{detection_system_index}"
                baseline_width = config0.get(fukuan_key, 0)

                # 初始化状态显示（详细面板由定时刷新统一绘制）
                if baseline_width > 0:
                    small_fukuan_labels[i].setText(
                        self._format_fukuan_status_richtext(
                            baseline_width,
                            float("nan"),
                            0,
                            False,
                            True,
                            False,
                        )
                    )
                    print(f"检测系统{detection_system_index}已激活，基准幅宽: {baseline_width}mm")
                else:
                    small_fukuan_labels[i].setText(
                        self._format_fukuan_status_richtext(0, float("nan"), 0, False, False, False)
                    )
                    print(f"检测系统{detection_system_index}未激活（幅宽为0）")

                # 为所有系统都创建波形图显示区域
                if not self.figures[i] or not self.canvases[i]:
                    self.figures[i], self.axes[i] = plt.subplots(figsize=(9, 3))
                    self.canvases[i] = FigureCanvas(self.figures[i])

                    layout = QVBoxLayout(fukuan_labels[i])
                    layout.addWidget(self.canvases[i])
                    fukuan_labels[i].setLayout(layout)
                    self.figures[i].subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.2)

                # 为所有系统都启动线程（线程内部会根据幅宽值决定是否工作）
                self.waveform_threads[i] = WaveformThread(calibrat_cam_id, detection_system_index)
                #self.update_signal.emit(new_data, has_abnormal, self.detection_system_index)
                self.waveform_threads[i].update_signal.connect(self.plot_waveform)#将第i个波形线程的update_signal信号连接到主窗口的plot_waveform槽函数
                self.waveform_threads[i].start()

        except Exception as e:
            print(f"showfukuan发生错误：{e}")



    def non_blocking_information(self, parent, title, message):
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.show()  # 非阻塞显示
        return msg_box  # 返回消息框对象（如果需要进一步操作）

    def plot_waveform(self, data, has_abnormal, detection_system_index, last_mm, tail_streak):
        try:
            system_index = detection_system_index - 1
            self.total_data[system_index] = data
            self.abnormal_status[system_index] = has_abnormal
            self.fukuan_last_measured[system_index] = last_mm
            self.fukuan_tail_narrow[system_index] = tail_streak
        except Exception as e:
            print(f"plot_waveform系统{detection_system_index}发生错误：{e}")


    def update_all_plots(self):
        """
        统一渲染入口：由 QTimer 驱动，处理所有可见系统（2~4条）的渲染逻辑。
        合并了逻辑计算和 Matplotlib 绘图，代码量大，但函数数量最少。
        """

        # 字体路径（假设它不变）
        font_path = os.path.join(_REPO_ROOT, 'config', 'simhei.ttf')
        font_prop = fm.FontProperties(fname=font_path)

        # 横坐标窗口目标长度（单位：m）
        TARGET_X_WINDOW_M = 10.0
        # 浮点游标平滑参数：每帧至少前进 min_step 个采样点，追赶 gap 时按 alpha 比例 + 上限 max_step
        SMOOTH_MIN_STEP = 0.025
        SMOOTH_ALPHA = 0.14
        SMOOTH_MAX_STEP = 10.0

        # 循环处理可见系统（2~4条）
        for system_index in range(self._visible_count()):
            try:
                detection_system_index = system_index + 1
                small_fukuan_labels = self._small_fukuan_labels()

                # --- 1. 配置读取和初始检查 ---
                with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                    config0 = yaml.safe_load(file)
                baseline_width = config0.get(f"fukuan_{detection_system_index}", 0)

                raw_data = self.total_data[system_index]
                total_len = len(raw_data)

                if not self.axes[system_index]: continue  # 确保图表存在

                # --- 2. 状态标签和非活跃/无数据处理 ---
                if baseline_width <= 0 or total_len < 2:
                    self.display_smooth_end[system_index] = 0.0
                    # 标签更新
                    if baseline_width <= 0:
                        small_fukuan_labels[system_index].setText(
                            self._format_fukuan_status_richtext(
                                0, float("nan"), 0, False, False, False
                            )
                        )
                        self.axes[system_index].text(0.1, 0.5, '无钢带（幅宽设置为0)',
                                                     horizontalalignment='left',
                                                     verticalalignment='center',
                                                     transform=self.axes[system_index].transAxes,
                                                     fontsize=16,
                                                     color='gray',
                                                     fontproperties=font_prop)
                        self.canvases[system_index].draw()
                        continue
                    else:
                        small_fukuan_labels[system_index].setText(
                            self._format_fukuan_status_richtext(
                                baseline_width,
                                self.fukuan_last_measured[system_index],
                                self.fukuan_tail_narrow[system_index],
                                self.abnormal_status[system_index],
                                True,
                                total_len >= 2,
                            )
                        )

                    # 清空图表并返回
                    self.axes[system_index].clear()
                    self.canvases[system_index].draw()
                    continue

                # --- 3. 流动呈现逻辑（浮点游标 + 插值，窗口连续左移）---
                with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                ratio = config["cam1_standard_ratio_x"]

                x_factor = ratio * 4096 * 0.001
                x_step = x_factor if x_factor != 0 else 1.0
                mm_per_idx = x_step * 1000.0  # 与横轴物理长度一致，供缺陷图对齐
                # 四舍五入换算点数，让窗口长度更接近目标值
                N_window = int(TARGET_X_WINDOW_M / x_step + 0.5) + 1
                N_window = max(2, N_window)
                N_window = min(N_window, total_len)

                target_end = float(total_len)
                cur = self.display_smooth_end[system_index]
                if cur > target_end:
                    cur = target_end
                gap = target_end - cur
                if gap > 1e-9:
                    step = max(SMOOTH_MIN_STEP, min(gap * SMOOTH_ALPHA, SMOOTH_MAX_STEP))
                    cur = min(cur + step, target_end)
                self.display_smooth_end[system_index] = cur
                self.display_end_indices[system_index] = int(round(cur))

                end_f = cur
                start_f = max(0.0, end_f - float(N_window - 1))
                # 与缺陷图共用同一长度窗口（mm，与波形索引一致）
                self.wave_window_start_mm[system_index] = float(start_f * mm_per_idx)
                self.wave_window_end_mm[system_index] = float(end_f * mm_per_idx)
                current_multiple = int(self.wave_window_start_mm[system_index] // 1720)
                self.cm[system_index] = current_multiple
                self.cm2[system_index] = current_multiple

                # --- 4. 数据插值与 X 轴（连续滑动）---
                idx_axis = np.arange(total_len, dtype=float)
                raw_arr = np.asarray(raw_data, dtype=float)
                n_samples = max(64, min(512, int(max(2, (end_f - start_f) * 6)) + 1))
                xi = np.linspace(start_f, end_f, n_samples)
                xi = np.clip(xi, 0.0, float(total_len - 1))
                y_plot = np.interp(xi, idx_axis, raw_arr)
                x_plot = (xi - start_f) * x_step
                abs_x_offset = start_f * x_step

                # --- 5. Matplotlib 绘图和样式更新 ---

                self.axes[system_index].clear()

                # 绘制波形和基准线
                self.axes[system_index].plot(x_plot, y_plot, color='blue', linewidth=1.0)
                self.axes[system_index].axhline(y=baseline_width, color='green', linestyle='--', linewidth=2,
                                                label='基准线')

                # 设置标签
                self.axes[system_index].set_xlabel('长度 (单位：m)', fontsize=12, fontproperties=font_prop)
                self.axes[system_index].set_ylabel('宽度 (单位：mm)', fontsize=12, fontproperties=font_prop)
                self.axes[system_index].set_title(f'检测系统{detection_system_index}', fontsize=14,
                                                  fontproperties=font_prop)
                self.axes[system_index].tick_params(axis='both', which='both', direction='in', labelsize=10)

                # 固定 X 轴范围（窗口物理宽度 = 浮点窗长 * 步长）
                span_idx = max(1e-9, end_f - start_f)
                x_window = span_idx * x_step
                if x_window <= 0:
                    x_window = 1.0
                self.axes[system_index].set_xlim([0, x_window])
                # 通过重设刻度文字，让用户看到的横坐标是绝对坐标系下的位置
                # （避免直接让 xlim 随数据平移导致“看不到滑动”的观感退化）
                tick_count = 6
                ticks = np.linspace(0, x_window, tick_count)
                tick_labels = [f"{(t + abs_x_offset):.1f}" for t in ticks]
                self.axes[system_index].set_xticks(ticks)
                self.axes[system_index].set_xticklabels(tick_labels, fontproperties=font_prop)

                # 更新 Y 轴范围
                if len(y_plot) > 0:
                    y_min = np.min(y_plot)
                    y_max = np.max(y_plot)
                    self.axes[system_index].set_ylim([y_min - 10, y_max + 10])

                    # 这里不额外处理 label

                small_fukuan_labels[system_index].setText(
                    self._format_fukuan_status_richtext(
                        baseline_width,
                        self.fukuan_last_measured[system_index],
                        self.fukuan_tail_narrow[system_index],
                        self.abnormal_status[system_index],
                        True,
                        total_len >= 2,
                    )
                )

                self.figures[system_index].tight_layout()
                self.canvases[system_index].draw()

            except Exception as e:
                print(f"渲染核心错误 系统{detection_system_index}: {e}")

        # 幅宽窗口更新后再重绘缺陷与刻度，避免两路定时器乱序导致坐标轴与标点不同步
        try:
            for i in range(self._visible_count()):
                self.refresh_scale(1, i)
                self.refresh_scale(2, i)
            self.create_button_two()
        except Exception as e:
            print(f"缺陷显示同步重绘: {e}")

    def start_programs(self):
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            missing_configs = []

            # 检查基本必需配置（带钢产品号与产品型号）
            if not config.get('conduct_id') or config['conduct_id'] == '':
                missing_configs.append("带钢产品号")
            if not config.get('product_cls') or config['product_cls'] == '':
                missing_configs.append("产品型号")

            # 检查是否至少有一个检测系统被激活（支持2~4条）
            self._load_system_count_from_config()
            active_systems = []
            fukuan_values = []
            for i in range(1, 5):
                fukuan_values.append(float(config.get(f"fukuan_{i}", 0)))
            for i in range(self.system_count):
                if fukuan_values[i] > 0:
                    active_systems.append(f"检测系统{i + 1}")

            # 如果没有任何激活的系统，添加到缺失配置
            if not active_systems:
                missing_configs.append("至少一个检测系统的幅宽（当前所有系统幅宽都为0）")

            # 4条模式下，要求第4条幅宽必须有效，避免UI显示4条但算法仅有效3条
            if self.system_count == 4 and fukuan_values[3] <= 0:
                missing_configs.append("当前为4条带钢模式，fukuan_4 必须大于0")

            if missing_configs:
                missing_str = "、".join(missing_configs)
                QMessageBox.information(self, "配置不完整",
                                        f"请完善以下配置项：{missing_str}")
                return
            else:
                # =========================
                # 1) 启动 Python 检测程序（用 QProcess 统一管理）
                # =========================
                python_exe = os.environ.get('STEEL_PYTHON_EXE', sys.executable)
                py_script = os.path.join(_REPO_ROOT, "detect_anomalies_online.py")

                # 若之前启动过，先清理（避免重复启动）
                try:
                    if hasattr(self, "python_process") and self.python_process is not None:
                        if self.python_process.state() != QProcess.NotRunning:
                            self.python_process.kill()
                            self.python_process.waitForFinished(2000)
                except Exception:
                    pass

                self.python_process = QProcess(self)

                # 合并 stdout/stderr，避免你只连 stdout 结果报错信息丢了
                self.python_process.setProcessChannelMode(QProcess.MergedChannels)

                # 你原来只连了 StandardOutput，这里仍然走 handle_output 就够了
                self.python_process.readyReadStandardOutput.connect(self.handle_output)

                # 设置工作目录（强烈建议：相对路径/读取配置更稳定）
                self.python_process.setWorkingDirectory(os.path.join(_REPO_ROOT))

                # 启动
                # self.python_process.start(python_exe, [py_script])
                self.python_process.start(python_exe, ["-u", py_script])

                if not self.python_process.waitForStarted(3000):
                    QMessageBox.critical(self, "启动失败",
                                         "Python 检测程序启动失败，请检查解释器路径/脚本路径/环境依赖。")
                    return

                # =========================
                # 2) 启动 C# 多相机程序（MultiCamDemo.exe）
                # =========================
                csharp_exe = os.environ.get('MULTICAM_DEMO_EXE', os.path.join(_REPO_ROOT, 'external', 'MultiCamDemo', 'MultiCamDemo.exe'))
                csharp_workdir = os.environ.get('MULTICAM_DEMO_CWD', os.path.join(_REPO_ROOT, 'external', 'MultiCamDemo'))

                # 若之前启动过，先清理（避免重复启动）
                try:
                    if hasattr(self, "csharp_process") and self.csharp_process is not None:
                        if self.csharp_process.state() != QProcess.NotRunning:
                            self.csharp_process.kill()
                            self.csharp_process.waitForFinished(2000)
                except Exception:
                    pass

                # 检查 exe 是否存在（避免“启动了个空气”）
                if not os.path.exists(csharp_exe):
                    QMessageBox.critical(self, "启动失败", f"未找到 C# 程序：\n{csharp_exe}")
                    return

                self.csharp_process = QProcess(self)
                self.csharp_process.setProcessChannelMode(QProcess.MergedChannels)
                # 如果你也希望把C#输出显示到同一个 handle_output，可以接上：
                # self.csharp_process.readyReadStandardOutput.connect(self.handle_output)
                self.csharp_process.setWorkingDirectory(csharp_workdir)

                self.csharp_process.start(csharp_exe)

                if not self.csharp_process.waitForStarted(5000):
                    # 获取详细错误信息
                    error_state = self.csharp_process.error()
                    error_string = self.csharp_process.errorString()

                    QMessageBox.critical(self, "启动失败",
                                         f"C# 程序启动失败！\n\n"
                                         f"错误代码: {error_state}\n"
                                         f"错误详情: {error_string}\n\n"
                                         f"路径: {csharp_exe}")
                    return

                self.run_state.setText("运行")
                self.run_state.setStyleSheet("color: green;")
                active_str = "、".join(active_systems)
                print(f"系统启动成功，激活的系统: {active_str}")
                for i in range(self.system_count):
                    fw = fukuan_values[i]
                    print(f"系统{i+1}幅宽: {fw}mm {'(激活)' if fw > 0 else '(未激活)'}")
        except Exception as e:
            print(f"start_programs发生错误：{e}")

    def handle_output(self):
        data = self.python_process.readAllStandardOutput()
        if not data:
            return

        b = bytes(data)

        # Windows 上子进程很多是 GBK 输出
        try:
            text = b.decode("gbk")
        except UnicodeDecodeError:
            text = b.decode("utf-8", errors="replace")

        print(text, end="", flush=True)

    def update_coordinates(self, x, y, c_m, pos, fukuan_for_point, path, system_index):
        """处理上表面坐标（对应label_up_show）"""
        try:
            list_index = system_index - 1
            # 将坐标添加到对应系统的上表面缓存（用于滑动窗口渲染）
            file_name = self.find_file_with_coordinates(path, int(x), int(y))
            self.defect_points_up[list_index].append({
                "x": float(x),
                "y": float(y),
                "fukuan": float(fukuan_for_point) if fukuan_for_point else 0.0,
                "path": path,
                "file": file_name,
            })
            self.cm[list_index] = c_m
            self.pos[list_index] = pos
            self.base_folder[list_index] = path
            self.fukuan_mm[list_index] = fukuan_for_point
            # 轻量缓存：保留最近一段时间数据，避免无限增长
            if len(self.defect_points_up[list_index]) > 3000:
                self.defect_points_up[list_index] = self.defect_points_up[list_index][-2200:]
            # print(f"检测系统{system_index}上表面接收坐标: ({x}, {y})")
        except Exception as e:
            print(f"update_coordinates系统{system_index}发生错误：{e}")

    def update_coordinates2(self, x, y, c_m, pos, fukuan_for_point, path, system_index):
        """处理下表面坐标（对应label_down_show）"""
        try:
            list_index = system_index - 1
            # 将坐标添加到对应系统的下表面缓存（用于滑动窗口渲染）
            file_name = self.find_file_with_coordinates(path, int(x), int(y))
            self.defect_points_down[list_index].append({
                "x": float(x),
                "y": float(y),
                "fukuan": float(fukuan_for_point) if fukuan_for_point else 0.0,
                "path": path,
                "file": file_name,
            })
            self.cm2[list_index] = c_m
            self.pos2[list_index] = pos
            self.base_folder2[list_index] = path
            self.fukuan_mm2[list_index] = fukuan_for_point
            # 轻量缓存：保留最近一段时间数据，避免无限增长
            if len(self.defect_points_down[list_index]) > 3000:
                self.defect_points_down[list_index] = self.defect_points_down[list_index][-2200:]
            # print(f"检测系统{system_index }下表面接收坐标: ({x}, {y})")
        except Exception as e:
            print(f"update_coordinates2系统{system_index}发生错误：{e}")

    def refresh_buttons(self, system_index):
        try:
            for button in self.buttons[system_index]:
                try:
                    button.deleteLater()
                except Exception as e:
                    print(f"删除上表面按钮时出错: {str(e)}")
            self.buttons[system_index] = []

            try:
                self.refresh_scale(1, system_index)
            except Exception as e:
                print(f"刷新刻度时出错: {str(e)}")
        except Exception as e:
            print(f"刷新系统{system_index+1}上表面按钮过程中出现未知错误: {str(e)}")


    def refresh_buttons2(self, system_index):
        try:
            for button in self.buttons2[system_index]:
                try:
                    button.deleteLater()
                except Exception as e:
                    print(f"删除下表面按钮时出错: {str(e)}")
            self.buttons2[system_index] = []

            # 刷新刻度
            try:
                self.refresh_scale(2, system_index)
            except Exception as e:
                print(f"刷新刻度时出错: {str(e)}")
        except Exception as e:
            print(f"刷新系统{system_index+1}下表面按钮过程中出现未知错误: {str(e)}")

    def refresh_scale(self, tag, system_index):
        try:
            """绘制刻度线和刻度标签"""
            if tag == 1:
                up_labels = self._up_show_labels()
                label = up_labels[system_index]
                label_width = label.width()
                label_height = label.height()
                fukuan_mm = self.fukuan_mm[system_index]
                axis_left_label = self.up_axis_left[system_index]
                axis_bottom_label = self.up_axis_bottom[system_index]
            else:
                down_labels = self._down_show_labels()
                label = down_labels[system_index]
                label_width = label.width()
                label_height = label.height()
                fukuan_mm = self.fukuan_mm2[system_index]
                axis_left_label = self.down_axis_left[system_index]
                axis_bottom_label = self.down_axis_bottom[system_index]

            # 与缺陷红点同一套像素映射（create_button），保证刻度与 JSON 坐标一致
            button_w_px = 10
            button_h_px = 10
            left_pad = 0
            bottom_pad = 0
            x_max_px = max(1, label_width - button_w_px - left_pad)
            y_max_px = max(1, label_height - button_h_px - bottom_pad)

            window_start_mm = float(self.wave_window_start_mm[system_index])
            window_end_mm = float(self.wave_window_end_mm[system_index])
            window_len_mm = window_end_mm - window_start_mm
            if window_len_mm <= 1e-6:
                window_len_mm = max(1720.0, 1.0)

            # 左轴（宽度mm）
            left_w = axis_left_label.width()
            left_h = axis_left_label.height()
            left_pm = QPixmap(left_w, left_h)
            left_pm.fill(QtCore.Qt.transparent)
            painter_l = QPainter(left_pm)
            pen = QPen(QtCore.Qt.black)
            painter_l.setPen(pen)
            painter_l.setFont(QFont("Arial", 8))
            y_full = float(fukuan_mm) if fukuan_mm and fukuan_mm > 0 else 600.0
            # 纵轴主轴线（外侧）
            painter_l.drawLine(left_w - 1, 0, left_w - 1, y_max_px)
            for i in range(7):
                y_pos = int(round(i * y_max_px / 6.0))
                y_pos = max(0, min(y_max_px, y_pos))
                width_mm = (i / 6.0) * y_full
                painter_l.drawLine(left_w - 7, y_pos, left_w - 1, y_pos)
                text_rect_y = max(0, min(left_h - 16, y_pos - 8))
                painter_l.drawText(0, text_rect_y, left_w - 9, 16, Qt.AlignRight | Qt.AlignVCenter, f'{width_mm:.0f}')
            painter_l.end()
            axis_left_label.setPixmap(left_pm)

            # 下轴（长度m）
            bottom_w = axis_bottom_label.width()
            bottom_h = axis_bottom_label.height()
            bottom_pm = QPixmap(bottom_w, bottom_h)
            bottom_pm.fill(QtCore.Qt.transparent)
            painter_b = QPainter(bottom_pm)
            painter_b.setPen(pen)
            painter_b.setFont(QFont("Arial", 8))
            # 横轴主轴线（外侧）
            painter_b.drawLine(0, 0, x_max_px, 0)
            for i in range(11):
                x_rel = int(round(i * x_max_px / 10.0))
                x_rel = max(0, min(x_max_px, x_rel))
                length_mm = window_start_mm + (x_rel / float(x_max_px)) * window_len_mm
                length_m = length_mm / 1000.0
                painter_b.drawLine(x_rel, 0, x_rel, 6)
                text_x = max(0, min(bottom_w - 42, x_rel - 21))
                painter_b.drawText(text_x, 6, 42, 12, Qt.AlignHCenter | Qt.AlignTop, f'{length_m:.2f}')
            painter_b.end()
            axis_bottom_label.setPixmap(bottom_pm)
        except Exception as e:
            print(f"刷新刻度过程中出现错误: {str(e)}")

    def create_button_two(self):
        for i in range(self._visible_count()):
            self.create_button(i)  # 处理系统i的上表面
            self.create_button2(i)  # 处理系统i的下表面

    def create_button(self, system_index):
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)

            detection_system_index = system_index + 1
            fukuan_key = f"fukuan_{detection_system_index}"
            baseline_width = config0.get(fukuan_key, 0)

            if baseline_width <= 0:
                return
            display_labels = self._up_show_labels()
            display_label = display_labels[system_index]
            label_width_px = display_label.width()
            label_height_px = display_label.height()
            button_w_px = 10
            button_h_px = 10
            left_pad = 0
            bottom_pad = 0
            x_max_px = max(1, label_width_px - button_w_px - left_pad)
            y_max_px = max(1, label_height_px - button_h_px - bottom_pad)
            x_origin = left_pad

            # 清空旧按钮，按当前滑动窗口重绘
            self.refresh_buttons(system_index)

            window_start = self.wave_window_start_mm[system_index]
            window_end = self.wave_window_end_mm[system_index]
            window_len = max(1.0, window_end - window_start)

            points = self.defect_points_up[system_index]
            # 仅保留窗口附近点，控制内存与渲染量
            keep_from = window_start - window_len * 1.5
            self.defect_points_up[system_index] = [p for p in points if p["y"] >= keep_from]
            points = self.defect_points_up[system_index]

            visible = [p for p in points if window_start <= p["y"] <= window_end]
            if len(visible) > 160:
                visible = visible[-160:]

            latest_visible = None
            for p in visible:
                # 长度方向与幅宽波形使用同一窗口范围，保证同步滑动
                button_x_rel = int(((p["y"] - window_start) / window_len) * x_max_px)
                button_x_rel = max(0, min(x_max_px, button_x_rel))
                button_x = x_origin + button_x_rel

                point_fukuan = p["fukuan"] if p["fukuan"] > 0 else baseline_width
                button_y = int((max(0.0, p["x"]) / float(point_fukuan)) * y_max_px) if point_fukuan else 0
                button_y = max(0, min(y_max_px, button_y))

                button = QtWidgets.QPushButton(display_label)
                button.setGeometry(QRect(button_x, button_y, button_w_px, button_h_px))
                button.setStyleSheet("background-color: red;")
                button.clicked.connect(lambda checked, fp=p["path"], fn=p["file"], si=system_index:
                                       self.display_image2_up(fp, fn, si))
                button.raise_()
                button.show()
                self.buttons[system_index].append(button)
                latest_visible = p

            if latest_visible:
                self.display_image_up(latest_visible["path"], latest_visible["file"], system_index)
        except Exception as e:
            print(f"create按钮111过程中出现未知错误: {str(e)}")

    def create_button2(self, system_index):
        """处理下表面按钮创建（对应label_down_show）"""
        try:
            # 检查系统是否激活
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)

            detection_system_index = system_index + 1
            fukuan_key = f"fukuan_{detection_system_index}"
            baseline_width = config0.get(fukuan_key, 0)

            if baseline_width <= 0:
                return
            display_labels = self._down_show_labels()
            display_label = display_labels[system_index]
            label_width_px = display_label.width()
            label_height_px = display_label.height()
            button_w_px = 10
            button_h_px = 10
            left_pad = 0
            bottom_pad = 0
            x_max_px = max(1, label_width_px - button_w_px - left_pad)
            y_max_px = max(1, label_height_px - button_h_px - bottom_pad)
            x_origin = left_pad

            # 清空旧按钮，按当前滑动窗口重绘
            self.refresh_buttons2(system_index)

            window_start = self.wave_window_start_mm[system_index]
            window_end = self.wave_window_end_mm[system_index]
            window_len = max(1.0, window_end - window_start)

            points = self.defect_points_down[system_index]
            keep_from = window_start - window_len * 1.5
            self.defect_points_down[system_index] = [p for p in points if p["y"] >= keep_from]
            points = self.defect_points_down[system_index]

            visible = [p for p in points if window_start <= p["y"] <= window_end]
            if len(visible) > 160:
                visible = visible[-160:]

            latest_visible = None
            for p in visible:
                button_x_rel = int(((p["y"] - window_start) / window_len) * x_max_px)
                button_x_rel = max(0, min(x_max_px, button_x_rel))
                button_x = x_origin + button_x_rel

                point_fukuan = p["fukuan"] if p["fukuan"] > 0 else baseline_width
                button_y = int((max(0.0, p["x"]) / float(point_fukuan)) * y_max_px) if point_fukuan else 0
                button_y = max(0, min(y_max_px, button_y))

                button = QtWidgets.QPushButton(display_label)
                button.setGeometry(QRect(button_x, button_y, button_w_px, button_h_px))
                button.setStyleSheet("background-color: red;")
                button.clicked.connect(lambda checked, fp=p["path"], fn=p["file"], si=system_index:
                                       self.display_image2_down(fp, fn, si))
                button.raise_()
                button.show()
                self.buttons2[system_index].append(button)
                latest_visible = p

            if latest_visible:
                self.display_image_down(latest_visible["path"], latest_visible["file"], system_index)

        except Exception as e:
            print(f"create_button2系统{system_index + 1}过程中出现未知错误: {str(e)}")

    def find_file_with_coordinates(self, directory, x, y):
        """
        在指定目录中寻找包含 {x}_{y} 的文件。

        :param directory: 目录路径
        :param x: x 坐标
        :param y: y 坐标
        :return: 匹配的文件名，如果未找到返回 None
        """
        # 遍历目录中的文件
        try:
            if not os.path.exists(directory):
                print("目录不存在")
                return None

            files = os.listdir(directory)
            for file_name in os.listdir(directory):
                # 检查文件名是否包含 {x}_{y}
                if f"{x}_{y}" in file_name:
                    return file_name  # 返回匹配的文件名
            return None  # 未找到则返回 None
        except Exception as e:
            print(f"寻找细节图文件名过程中出现未知错误: {str(e)}")


    def display_image_up(self, folder_path, file_name,  system_index):
        try:
            realtime_labels = self._up_realtime_labels()
            label = realtime_labels[system_index]
            # 显示选中的图片
            pixmap = QPixmap(os.path.join(folder_path, file_name))

            # 获取label的尺寸
            label_width = label.width()
            label_height = label.height()

            # 将图片拉伸至label的尺寸，忽略长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.IgnoreAspectRatio)
            label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"实时显示系统{system_index + 1}上表面图像错误: {str(e)}")

    def display_image_down(self, folder_path, file_name,  system_index):
        try:
            realtime_labels = self._down_realtime_labels()
            label = realtime_labels[system_index]
            # 显示选中的图片
            pixmap = QPixmap(os.path.join(folder_path, file_name))

            # 获取label的尺寸
            label_width = label.width()
            label_height = label.height()

            # 将图片拉伸至label的尺寸，忽略长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.IgnoreAspectRatio)
            label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"实时显示系统{system_index + 1}上表面图像错误: {str(e)}")

    def display_image2_up(self, folder_path, file_name, system_index):
        try:
            if not os.path.exists(folder_path):
                print(f"文件夹不存在: {folder_path}")
                return

            # 3. 检查文件是否存在
            full_path = os.path.join(folder_path, file_name)
            print(f"完整文件路径: {full_path}")

            if not os.path.exists(full_path):
                print(f"文件不存在: {full_path}")
                # 列出目录内容帮助调试
                files = os.listdir(folder_path)
                print(f"目录中的文件: {files}")
                return

            # 4. 检查文件可读性
            if not os.access(full_path, os.R_OK):
                print(f"文件不可读: {full_path}")
                return
            full_path = os.path.join(folder_path, file_name)
            print(f"点击加载路径: {full_path}")
            pixmap = QPixmap(full_path)
            if pixmap.isNull():
                print(f"❌ QPixmap加载失败")
                return
            click_labels = self._up_click_labels()
            label = click_labels[system_index]

            # 获取label的尺寸
            label_width = label.width()
            label_height = label.height()

            # 将图片拉伸至label的尺寸，忽略长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.IgnoreAspectRatio)
            label.setPixmap(scaled_pixmap)
            print("图片已设置到 Label")
            # 7. 强制刷新
            label.repaint()
            QApplication.processEvents()  # 处理挂起的事件

        except Exception as e:
            print(f"点击显示系统{system_index + 1}上表面图像错误: {str(e)}")

    def display_image2_down(self, folder_path, file_name, system_index):
        try:
            if not os.path.exists(folder_path):
                print(f"文件夹不存在: {folder_path}")
                return

            # 3. 检查文件是否存在
            full_path = os.path.join(folder_path, file_name)
            print(f"完整文件路径: {full_path}")

            if not os.path.exists(full_path):
                print(f"文件不存在: {full_path}")
                # 列出目录内容帮助调试
                files = os.listdir(folder_path)
                print(f"目录中的文件: {files}")
                return

            # 4. 检查文件可读性
            if not os.access(full_path, os.R_OK):
                print(f"文件不可读: {full_path}")
                return
            full_path = os.path.join(folder_path, file_name)
            print(f"点击加载路径: {full_path}")
            pixmap = QPixmap(full_path)
            if pixmap.isNull():
                print(f"❌ QPixmap加载失败")
                return
            click_labels = self._down_click_labels()
            label = click_labels[system_index]

            # 获取label的尺寸
            label_width = label.width()
            label_height = label.height()

            # 将图片拉伸至label的尺寸，忽略长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.IgnoreAspectRatio)
            label.setPixmap(scaled_pixmap)
            label.repaint()
            QApplication.processEvents()  # 处理挂起的事件
        except Exception as e:
            print(f"点击显示系统{system_index + 1}下表面图像错误: {str(e)}")

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    _apply_app_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())