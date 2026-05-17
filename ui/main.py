try:
    # 优先使用 PyQt5（原工程依赖）
    from PyQt5 import QtWidgets, QtCore
    from PyQt5.QtWidgets import QInputDialog, QMessageBox, QLineEdit, QComboBox
    from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QFontMetrics, QDesktopServices
    from PyQt5.QtCore import (
        QRect,
        Qt,
        QTimer,
        QThread,
        pyqtSignal,
        pyqtSlot,
        QProcess,
        QEvent,
        QUrl,
        QMetaObject,
        QElapsedTimer,
        QEventLoop,
    )
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QLabel,
        QPushButton,
        QHBoxLayout,
        QVBoxLayout,
        QSizePolicy,
    )
except ModuleNotFoundError:
    # 兼容：未安装 PyQt5 时自动回退 PySide6
    from PySide6 import QtWidgets, QtCore
    from PySide6.QtWidgets import QInputDialog, QMessageBox, QLineEdit, QComboBox
    from PySide6.QtGui import QPixmap, QPainter, QPen, QFont, QFontMetrics, QDesktopServices
    from PySide6.QtCore import (
        QRect,
        Qt,
        QTimer,
        QThread,
        Signal as pyqtSignal,
        Slot as pyqtSlot,
        QProcess,
        QEvent,
        QUrl,
        QMetaObject,
        QElapsedTimer,
        QEventLoop,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QLabel,
        QPushButton,
        QHBoxLayout,
        QVBoxLayout,
        QSizePolicy,
    )
from mainui import Ui_MainWindow  # 导入pyuic生成的类
from para import ParaWindow
from report_change import ReportWindow
from report_center import ReportCenterWindow
from cls_config import (
    ClsConfigWindow,
    product_combo_entries,
    product_cls_key_from_combo_text,
)
from cls_train_window import ClsTrainWindow
from cls_wizard import ClsWizardWindow
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import time
import traceback
import subprocess
import json
import json
import serial
import yaml
import numpy as np
import math
from datetime import timedelta, datetime

_PROJECT_ROOT = os.path.join(_REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import strip_result_paths as _strip_paths

_AUTH_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "auth.yaml")
_RUNTIME_STATE_PATH = os.path.join(_REPO_ROOT, "config", "runtime_state.json")
_LINE_HEARTBEAT_PATH = os.path.join(_REPO_ROOT, "config", "line_heartbeat.json")


def _ui_strip_dir_basename_for_roll(result_roll_path: str, detection_system_index: int) -> str:
    """把 UI 的条带序号(1..N)解析为 detect result 内实际条带目录名（与写端 strip_dir_list 对齐）。"""
    try:
        sid = int(detection_system_index)
    except Exception:
        sid = 1
    if sid < 1:
        sid = 1
    try:
        return _strip_paths.resolve_strip_dir_basename(str(result_roll_path or ""), sid)
    except Exception:
        return f"strip_{sid}"


def _clamp_strip_count_ui(n: int) -> int:
    return min(4, max(1, int(n)))


def _truth_strip_index_1based(ui_slot_1based: int, strip_count: int) -> int:
    """
    UI：从左到右输入依次为 1..n（与界面控件从左到右一致）。
    物理/算法：仍保持「图像从左到右依次为 1..n」不变（不写检测端）。

    需求：UI 左起第 k 个输入对应图像中从右数第 k 条带 =>
    UI 槽位 ui(1..n) 映射到物理序号 truth = n - ui + 1。
    """
    n = _clamp_strip_count_ui(strip_count)
    try:
        ui = int(ui_slot_1based)
    except Exception:
        ui = 1
    ui = max(1, min(n, ui))
    return n - ui + 1


def _open_image_path_with_system_viewer(path: str) -> bool:
    """用系统默认关联程序打开本地图像文件（Windows 优先 os.startfile）。"""
    p = os.path.normpath(str(path or ""))
    if not p or not os.path.isfile(p):
        return False
    try:
        if os.name == "nt":
            os.startfile(p)
            return True
    except Exception:
        pass
    try:
        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(p)))
    except Exception:
        return False


def _write_runtime_state(paused: bool) -> None:
    """
    运行态控制：给 detect_anomalies_online.py 一个“暂停/继续”开关。
    paused=True  -> 暂停（不杀进程、不关 socket，接收端按暂停策略处理）
    paused=False -> 继续
    """
    try:
        os.makedirs(os.path.dirname(_RUNTIME_STATE_PATH), exist_ok=True)
        tmp = _RUNTIME_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"paused": bool(paused)}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _RUNTIME_STATE_PATH)
    except Exception:
        # 控制文件写失败不应导致 UI 崩溃
        pass


def _read_auth_password(role: str) -> str:
    """从 auth.yaml 读取指定角色的密码，读取失败则回退到 '000'。"""
    try:
        with open(_AUTH_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("passwords", {}).get(role, "000"))
    except Exception:
        return "000"


def _read_line_heartbeat_ts() -> float:
    try:
        with open(_LINE_HEARTBEAT_PATH, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        return float(d.get("ts", 0.0) or 0.0)
    except Exception:
        return 0.0


def _defect_length_mm_to_px(y_mm, start_mm, end_mm, x_max_px, ui):
    """
    缺陷分布 / 幅宽曲线共用：物理长度 y(mm) 映射到横轴像素。
    ui 含 nonlinear、tail_phys_ratio、tail_pixel_ratio（见 _read_ui_defect_display_config）。
    """
    try:
        L = float(end_mm) - float(start_mm)
    except (TypeError, ValueError):
        return 0.0
    if L <= 1e-9:
        return 0.0
    try:
        y = float(y_mm)
    except (TypeError, ValueError):
        return 0.0
    sm, em = float(start_mm), float(end_mm)
    y = max(sm, min(em, y))
    xm = max(1.0, float(x_max_px))
    use_nl = bool(ui.get("nonlinear"))
    tpr = float(ui.get("tail_pixel_ratio", 0.2) or 0)
    tph = float(ui.get("tail_phys_ratio", 0.35) or 0)
    if not use_nl or tpr <= 1e-6 or tpr >= 1.0 - 1e-6 or tph <= 1e-9:
        return (y - sm) / L * xm
    tail_phys_ratio = min(0.99, max(1e-6, tph))
    tail_phys = L * tail_phys_ratio
    split_mm = em - tail_phys
    if split_mm <= sm + 1e-6:
        return (y - sm) / L * xm
    px_left = xm * (1.0 - tpr)
    px_right = xm * tpr
    if y <= split_mm:
        denom = max(split_mm - sm, 1e-9)
        return (y - sm) / denom * px_left
    denom = max(em - split_mm, 1e-9)
    return px_left + (y - split_mm) / denom * px_right


# ---------- 幅宽允许带判定（对称容差：偏窄 / 偏宽均视为「超出范围」）----------
# 下阈值 = 设定 − max(绝对容差mm, 设定×相对比例)；上阈值 = 设定 + 同带宽。
# 测量噪声、标定误差和正常工艺波动不会一碰就报警。
FUKUAN_NARROW_ABS_MM = 12.0
FUKUAN_NARROW_REL = 0.025
# 报警：最近 N 帧全部超出允许范围（过窄或过宽）
FUKUAN_ALARM_WINDOW = 32
# 预警：当前序列末尾连续超出范围帧数达到以下值，且未满足上述报警
FUKUAN_WARN_TAIL_STREAK = 6


def fukuan_narrow_threshold_mm(baseline_mm: float) -> float:
    """低于此实测值视为「显著偏窄」（已扣掉允许带）。"""
    if baseline_mm is None or baseline_mm <= 0:
        return float("inf")
    band = max(FUKUAN_NARROW_ABS_MM, float(baseline_mm) * FUKUAN_NARROW_REL)
    return float(baseline_mm) - band


def fukuan_wide_threshold_mm(baseline_mm: float) -> float:
    """高于此实测值视为「显著偏宽」（与偏窄使用同一带宽）。"""
    if baseline_mm is None or baseline_mm <= 0:
        return float("-inf")
    band = max(FUKUAN_NARROW_ABS_MM, float(baseline_mm) * FUKUAN_NARROW_REL)
    return float(baseline_mm) + band


def fukuan_out_of_range_mm(measured_mm: float, baseline_mm: float) -> bool:
    """实测超出允许带（偏窄或偏宽）。"""
    if baseline_mm is None or baseline_mm <= 0:
        return False
    lo = fukuan_narrow_threshold_mm(float(baseline_mm))
    hi = fukuan_wide_threshold_mm(float(baseline_mm))
    return float(measured_mm) < lo or float(measured_mm) > hi


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

# 幅宽曲线内存与插值窗口：避免 fukuan.json 极长时 UI 内存与 CPU 线性爆炸
TOTAL_DATA_MAX_SAMPLES = 10000


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
    # 源码：与 main.py 同目录；PyInstaller onefile：theme.qss 由 spec 的 datas 置于 _MEIPASS 根目录
    if getattr(sys, "frozen", False):
        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    else:
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
    # data, has_abnormal, detection_system_index, last_measured_mm, tail_out_of_range_streak
    # meta: dict from fukuan_meta.json last entry (may be None)
    update_signal = pyqtSignal(list, bool, int, float, int, object)

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
        返回 (是否报警, 最新实测mm, 末尾连续超出允许范围帧数)。
        允许范围：[fukuan_narrow_threshold_mm, fukuan_wide_threshold_mm]（对称容差）。
        报警：序列长度 ≥ FUKUAN_ALARM_WINDOW，且最近 FUKUAN_ALARM_WINDOW 帧全部超出允许范围。
        """
        if not values or baseline_width <= 0:
            return False, float("nan"), 0
        bl = float(baseline_width)
        nums = [float(v) for v in values]
        last_mm = nums[-1]

        def _oor(v: float) -> bool:
            return fukuan_out_of_range_mm(v, bl)

        tail = 0
        for v in reversed(nums):
            if _oor(v):
                tail += 1
            else:
                break

        w = int(FUKUAN_ALARM_WINDOW)
        n = len(nums)
        has_abnormal = n >= w and all(_oor(v) for v in nums[-w:])
        return has_abnormal, last_mm, tail

    def run(self):
        # 禁止 QTimer(self)：QThread 对象属于主线程 affinity，在工作线程里给其挂子对象会触发 Qt 跨线程告警，
        # 且在暂停/关闭时 stop timer 可能报错并导致进程崩溃。
        self.timer = QTimer()
        self.timer.setTimerType(Qt.CoarseTimer)
        self.timer.timeout.connect(self.fukuan)
        self.timer.start(2000)
        self.exec_()

    @pyqtSlot()
    def _stop_in_thread(self):
        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
                self.timer = None
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass

    def stop(self, blocking_ms=8000):
        if not self.isRunning():
            return
        self._is_running = False
        # 禁止 BlockingQueuedConnection：主线程在 fukuan() 未返回时无法处理 stop 槽，会死锁卡死 UI。
        try:
            QMetaObject.invokeMethod(self, "_stop_in_thread", Qt.QueuedConnection)
        except Exception:
            try:
                self.quit()
            except Exception:
                pass
        self.wait(int(blocking_ms))

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
            # 夜班跨日：凌晨 0~6 点允许回看昨天目录
            folder_path.append(yesterday_folder_path)
            return folder_path

        return folder_path

    @staticmethod
    def _cam_folder_name_from_id(camid):
        """
        目录命名统一：
        - camid=2 -> 上表面
        - camid=3 -> 下表面
        其余保持原数字目录（兼容旧结构）
        """
        try:
            c = int(camid)
        except Exception:
            return str(camid)
        if c == 2:
            return "上表面"
        if c == 3:
            return "下表面"
        return str(c)

    def read_json_data(self, file_path):
        # Windows 下写端与读端并发时，可能短暂读到“空/半截 JSON”
        # 这里做短重试，避免 UI 因竞争导致长期显示缺失
        last_err = None
        for _ in range(3):
            if not getattr(self, "_is_running", True):
                return []
            try:
                if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
                    return []
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        print(f"读取JSON文件失败: {last_err}")
        return []

    def fukuan(self):
        try:
            if not getattr(self, "_is_running", True):
                return
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)

            n = _clamp_strip_count_ui(int(config0.get("strip_count", 3) or 3))
            truth = _truth_strip_index_1based(self.detection_system_index, n)
            fukuan_key = f"fukuan_{truth}"
            baseline_width = config0.get(fukuan_key, 0)

            if baseline_width <= 0:
                self.update_signal.emit([], False, self.detection_system_index, float("nan"), 0, None)
                return

            id = config0['conduct_id']
            found_folders = self.find_folders_with_id0(os.path.join(_REPO_ROOT, 'detect result'), id)

            if len(found_folders) == 0:
                return

            root_path = found_folders[0]
            cam_folder = self._cam_folder_name_from_id(self.camid)
            strip_base = _ui_strip_dir_basename_for_roll(root_path, truth)
            self.fukuan_path = os.path.join(root_path, str(cam_folder), strip_base) + os.sep
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
                    last_meta = None
                    try:
                        meta_path = os.path.join(self.fukuan_path, "fukuan_meta.json")
                        meta_data = self.read_json_data(meta_path)
                        if isinstance(meta_data, list) and meta_data:
                            last_meta = meta_data[-1]
                    except Exception:
                        last_meta = None
                    self.update_signal.emit(
                        new_data,
                        has_abnormal,
                        self.detection_system_index,
                        last_mm,
                        tail_streak,
                        last_meta,
                    )

        except Exception as e:
            print(f"检测系统{self.detection_system_index}读取数据失败: {e}")
            self.update_signal.emit([], False, self.detection_system_index, float("nan"), 0, None)

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
        # 只打印一次的诊断信息（避免刷屏）
        self._diag_once = set()

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
        # JSONL 追尾读取状态
        self._jsonl_offset = 0
        self._jsonl_partial = ""
        self._jsonl_path = None
        self._jsonl_last_size = 0

    def run(self):
        poll_ms = 500
        try:
            with open(
                os.path.join(_REPO_ROOT, "config", "config.yaml"),
                "r",
                encoding="utf-8",
            ) as f:
                _cfg = yaml.safe_load(f) or {}
            poll_ms = int(_cfg.get("ui_defect_coord_poll_ms", 500) or 500)
        except Exception:
            pass
        poll_ms = max(100, min(poll_ms, 5000))
        self.timer = QTimer()
        self.timer.setTimerType(Qt.CoarseTimer)
        self.timer.timeout.connect(self.read_coordinates)
        self.timer.start(poll_ms)
        self.exec_()

    @pyqtSlot()
    def _stop_in_thread(self):
        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
                self.timer = None
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass

    def stop(self, blocking_ms=8000):
        if not self.isRunning():
            return
        self._is_running = False
        # 与 WaveformThread 相同：避免 BlockingQueuedConnection 在 read_coordinates 耗时期间死锁 UI。
        try:
            QMetaObject.invokeMethod(self, "_stop_in_thread", Qt.QueuedConnection)
        except Exception:
            try:
                self.quit()
            except Exception:
                pass
        self.wait(int(blocking_ms))

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
            # 夜班跨日：凌晨 0~6 点允许回看昨天目录
            folder_path.append(yesterday_folder_path)
            return folder_path

        return folder_path

    @staticmethod
    def _cam_folder_name_from_id(camid):
        """同 WaveformThread：camid=2/3 映射为 上表面/下表面。"""
        try:
            c = int(camid)
        except Exception:
            return str(camid)
        if c == 2:
            return "上表面"
        if c == 3:
            return "下表面"
        return str(c)

    def read_coordinates(self):
        try:
            if not getattr(self, "_is_running", True):
                return
            # UI 读取来源开关（默认 jsonl）
            ui_coord_source = "jsonl"
            max_lines = 200
            try:
                with open(os.path.join(_REPO_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as _cf:
                    _cfg = yaml.safe_load(_cf) or {}
                ui_coord_source = str(_cfg.get("ui_coord_source", "jsonl") or "jsonl").lower()
                max_lines = int(_cfg.get("ui_coord_max_lines_per_tick", 200) or 200)
            except Exception:
                pass
            max_lines = max(10, min(max_lines, 2000))

            # 1. 读取配置
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
                config0 = yaml.safe_load(file)
                n = _clamp_strip_count_ui(int(config0.get("strip_count", 3) or 3))
                truth = _truth_strip_index_1based(self.detection_system_index, n)
                fukuan_key = f"fukuan_{truth}"
                baseline_width = config0.get(fukuan_key, 0)
                if baseline_width <= 0:
                    return
                self.fukuan_fallback = float(baseline_width)

            id = config0['conduct_id']

            # 2. 查找文件夹
            base_detect_dir = os.path.join(_REPO_ROOT, "detect result")
            found_folders = self.find_folders_with_id(base_detect_dir, id)
            if len(found_folders) == 0:
                k = ("no_root", str(id))
                if k not in self._diag_once:
                    self._diag_once.add(k)
                    now = datetime.now()
                    today_date = now.strftime('%Y%m%d')
                    yesterday_date = (now - timedelta(days=1)).strftime('%Y%m%d')
                    exp_today = os.path.join(base_detect_dir, today_date, str(id))
                    exp_yday = os.path.join(base_detect_dir, yesterday_date, str(id))
                    print(
                        f"[UI][缺陷读取] 未找到结果目录：conduct_id={id} system={self.detection_system_index} camid={self.camid}\n"
                        f"  - 期望(今天): {exp_today}\n"
                        f"  - 期望(昨天): {exp_yday} (凌晨0~6点允许回看)"
                    )
                return

            root_path = found_folders[0]
            cam_folder = self._cam_folder_name_from_id(self.camid)
            strip_base = _ui_strip_dir_basename_for_roll(root_path, truth)
            self.folder_path = os.path.join(root_path, str(cam_folder), strip_base) + os.sep
            new_folder_name = "defect_images"
            self.base_folder = os.path.join(self.folder_path, new_folder_name)

            # 目标坐标文件路径
            coord_file_path = os.path.join(self.folder_path, "image_anomaly_center.json")
            self.fukuan_file_path = os.path.join(self.folder_path, "fukuan.json")
            jsonl_path = os.path.join(self.folder_path, "defect_events_center.jsonl")
            # jsonl 模式下，不能因为 legacy JSON 的“缺失/空文件”提前 return
            if ui_coord_source != "jsonl":
                if not os.path.exists(coord_file_path):
                    k = ("no_coord", coord_file_path)
                    if k not in self._diag_once:
                        self._diag_once.add(k)
                        print(
                            f"[UI][缺陷读取] 坐标文件不存在，无法显示缺陷点：\n"
                            f"  - coord_file_path: {coord_file_path}\n"
                            f"  - jsonl_path: {jsonl_path}\n"
                            f"  - base_folder(defect_images): {self.base_folder}"
                        )
                    return

                # 防止空文件报错：legacy_json 模式需要等待写端填充
                if os.path.getsize(coord_file_path) == 0:
                    return

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

            # ======================
            # JSONL 追尾读取（推荐）
            # ======================
            if ui_coord_source == "jsonl" and os.path.exists(jsonl_path):
                # 文件切换/轮转/截断：offset 归零
                if self._jsonl_path != jsonl_path:
                    self._jsonl_path = jsonl_path
                    self._jsonl_offset = 0
                    self._jsonl_partial = ""
                    self._jsonl_last_size = 0
                try:
                    cur_size = os.path.getsize(jsonl_path)
                    if cur_size < self._jsonl_offset:
                        # 被轮转或截断
                        self._jsonl_offset = 0
                        self._jsonl_partial = ""
                    if cur_size == self._jsonl_offset:
                        return
                    with open(jsonl_path, "r", encoding="utf-8") as fr:
                        fr.seek(self._jsonl_offset)
                        chunk = fr.read()
                        self._jsonl_offset = fr.tell()
                except Exception:
                    return

                buf = (self._jsonl_partial or "") + (chunk or "")
                lines = buf.split("\n")
                # 最后一行可能是半截，留到下次
                self._jsonl_partial = lines[-1]
                ready = lines[:-1]
                if not ready:
                    return

                processed = 0
                for ln in ready:
                    if not getattr(self, "_is_running", True):
                        return
                    if processed >= max_lines:
                        break
                    s = (ln or "").strip()
                    if not s:
                        continue
                    try:
                        ev = json.loads(s)
                    except Exception:
                        continue
                    pts = ev.get("points", [])
                    if not isinstance(pts, list) or not pts:
                        continue
                    # 每行可能包含多个点；按点 emit（保持现有 UI 缓存/按钮逻辑不变）
                    for p in pts:
                        if not getattr(self, "_is_running", True):
                            return
                        if processed >= max_lines:
                            break
                        try:
                            x, y = p
                            x = float(x); y = float(y)
                        except Exception:
                            continue
                        current_multiple = int(y // 1720)
                        # 根据长度段索引选择对应幅宽（Stable fukuan.json）
                        fukuan_for_point = self.fukuan_fallback
                        idx0 = current_multiple - 1
                        if 0 <= idx0 < len(self.fukuan_data):
                            fukuan_for_point = self.fukuan_data[idx0]
                        elif 0 <= current_multiple < len(self.fukuan_data):
                            fukuan_for_point = self.fukuan_data[current_multiple]
                        self.position += 1
                        self.image_loaded.emit(
                            x,
                            y,
                            current_multiple,
                            self.position,
                            float(fukuan_for_point) if fukuan_for_point else float(self.fukuan_fallback),
                            self.base_folder,
                            self.detection_system_index,
                        )
                        processed += 1
                return
            elif ui_coord_source == "jsonl":
                # 选择了 jsonl，但文件不存在：通常说明检测端未写 jsonl，或写到别的目录（camid/目录命名不一致）
                k = ("no_jsonl", jsonl_path)
                if k not in self._diag_once:
                    self._diag_once.add(k)
                    print(
                        f"[UI][缺陷读取] ui_coord_source=jsonl 但未发现 jsonl 文件：\n"
                        f"  - jsonl_path: {jsonl_path}\n"
                        f"  - 将继续等待 jsonl 出现（不会回退到 legacy JSON）"
                    )
                # jsonl 模式下不要回退 legacy_json（避免一直读空的 image_anomaly_center.json）
                return

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
                ok = False
                last_err = None
                # 并发写入时可能读到半截 JSON：短重试，失败不清空旧数据
                for _ in range(3):
                    if not getattr(self, "_is_running", True):
                        return
                    try:
                        if os.path.getsize(coord_file_path) == 0:
                            last_err = ValueError("empty file")
                            time.sleep(0.05)
                            continue
                        with open(coord_file_path, "r", encoding="utf-8") as file:
                            data = json.load(file)
                        self.all_coordinates = [coord for sublist in data for coord in sublist]
                        ok = True
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.05)

                if not ok:
                    print(f"JSON读取失败: {last_err}")
                    # 保留上一轮 all_coordinates，避免 UI 因一次读失败而“完全不显示”
                    return

                # ---【输出读取状态】---
                print(f"✅ 系统{self.detection_system_index} [相机{self.camid}] 数据更新! 总坐标数: {len(self.all_coordinates)}")
                # --------------------

                if is_new_file:
                    self.position = 0
                    print(f"   🔄 新文件，计数器重置")

                self.last_file_path = coord_file_path
                self.last_mtime = current_mtime

            # 发送数据逻辑（积压多时单次多推，减轻「窗口已滑过才入队」）
            remaining = len(self.all_coordinates) - self.position
            if remaining > 0:
                if remaining > 500:
                    read_count = min(50, remaining)
                elif remaining > 200:
                    read_count = min(25, remaining)
                elif remaining > 80:
                    read_count = min(15, remaining)
                else:
                    read_count = min(10, remaining)
                for i in range(read_count):
                    if not getattr(self, "_is_running", True):
                        return
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
        # 各条带最近一次在界面显示的缺陷图完整路径（用于双击用系统看图软件打开）
        self._preview_image_path_up = [None] * self.MAX_STRIPS
        self._preview_image_path_down = [None] * self.MAX_STRIPS
        self._production_nav_filter_installed = False
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
        # 滑动窗口“防空白”锚点：记录每个系统上表面/下表面最新出现的缺陷长度 y(mm)
        # 初始值用 -1 表示尚未收到任何缺陷点
        self.latest_defect_y = [-1.0 for _ in range(self.MAX_STRIPS)]
        # 缺陷点缓存：与波形窗口同时间轴滑动显示
        self.defect_points_up = [[] for _ in range(self.MAX_STRIPS)]
        self.defect_points_down = [[] for _ in range(self.MAX_STRIPS)]
        # 当前波形窗口的绝对长度范围（mm），用于缺陷图同步滑动
        self.wave_window_start_mm = [0.0 for _ in range(self.MAX_STRIPS)]
        self.wave_window_end_mm = [1720.0 for _ in range(self.MAX_STRIPS)]
        self.csharp_process = None
        self.python_process = None
        # 本次打开主界面的“确认”标记：必须本次点过确认，才允许开始/暂停
        self._confirmed_this_session = False
        # 与右上角检测状态一致：True=运行中（未暂停），换卷前须为 False
        self._ui_detection_running = False
        self.loader_thread =  [None for _ in range(self.MAX_STRIPS)]
        self.loader_thread2 =  [None for _ in range(self.MAX_STRIPS)]
        self.total_data = [[] for _ in range(self.MAX_STRIPS)]
        self.POINTS_TO_DISPLAY = 100
        self.abnormal_status = [False] * self.MAX_STRIPS
        self.fukuan_last_measured = [float("nan")] * self.MAX_STRIPS
        self.fukuan_tail_narrow = [0] * self.MAX_STRIPS
        # 幅宽双轨/保护信息（Stable 显示 + Raw 追溯）
        self.fukuan_last_raw = [float("nan")] * self.MAX_STRIPS
        self.fukuan_last_valid = [True] * self.MAX_STRIPS
        self.fukuan_last_reason = [""] * self.MAX_STRIPS
        self.fukuan_last_mode = [""] * self.MAX_STRIPS
        self.display_end_indices = [0 for _ in range(self.MAX_STRIPS)]
        # 浮点游标：平滑追赶最新数据长度，避免整档跳跃
        self.display_smooth_end = [0.0 for _ in range(self.MAX_STRIPS)]

        self.waveform_threads = [None for _ in range(self.MAX_STRIPS)]  # 幅宽监测线程
        self.figures = [None for _ in range(self.MAX_STRIPS)]  # 波形图
        self.canvases = [None for _ in range(self.MAX_STRIPS)]  # 画布
        self.axes = [None for _ in range(self.MAX_STRIPS)]  # 坐标轴
        # 幅宽曲线“内置绘制画布”（替代 Matplotlib）。由 update_all_plots() 使用 QPainter 绘制折线与基准线
        self.fukuan_plot_canvases = [None for _ in range(self.MAX_STRIPS)]
        # 幅宽波形的外置坐标轴（用 QLabel + QPixmap 画刻度，避免 Matplotlib 内置文字挤占画布）
        self.fukuan_axis_left_labels = [None for _ in range(self.MAX_STRIPS)]
        self.fukuan_axis_bottom_labels = [None for _ in range(self.MAX_STRIPS)]
        # 幅宽外置轴标题（对标缺陷分布窗口的“宽度(mm)”与“长度(m)”）
        self.fukuan_width_titles = [None for _ in range(self.MAX_STRIPS)]
        self.fukuan_len_titles = [None for _ in range(self.MAX_STRIPS)]

        self.pushButton_start.clicked.connect(self.button_start_click)#界面上的按钮pushButton_start和自己定义的button_start_click函数连接起来
        self.pushButton_stop.clicked.connect(self.button_stop_click)
        self.pushButton_para.clicked.connect(self.pushButton_para_click)
        # 主界面不再提供「报告打印」按钮入口：避免与「报告中心->生成报告」功能重复
        try:
            self.pushButton_old_report.hide()
            self.pushButton_old_report.setEnabled(False)
        except Exception:
            pass
        # 右上角：产线状态 + 检测状态（统一布局，表达清晰）
        try:
            self._layout_status_panel()
        except Exception:
            self.line_state = None
        self.pushButton_report.clicked.connect(self.pushButton_report_click)
        self.pushButton.clicked.connect(self.baojing_close)
        self.button_exchange.clicked.connect(self.exchangeNEWONE)
        self.para_config01.clicked.connect(self.save_config01)
        self.para_window = None#声明窗口实例变量，初始化为 None 表示窗口尚未创建
        self.report_window = None
        self.report_center_window = None
        # 初始：未确认配置时不允许开始/暂停
        try:
            self._refresh_start_stop_enabled()
        except Exception:
            pass
        # 初始：仅在“暂停态（未运行）”允许换卷
        try:
            self._refresh_exchange_enabled()
        except Exception:
            pass

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_all_plots)
        # 约 12.5 FPS：降低 YAML/插值/重绘负载，幅宽曲线仍足够流畅
        self.render_timer.start(80)

        # update_all_plots 用：config.yaml / config0.yaml 按 mtime 缓存，避免每帧解析
        self._plot_cfg_yaml_path = os.path.join(_REPO_ROOT, "config", "config.yaml")
        self._plot_cfg0_yaml_path = os.path.join(_REPO_ROOT, "config", "config0.yaml")
        self._plot_cfg_cache = None
        self._plot_cfg_mtime = None
        self._plot_cfg0_cache = None
        self._plot_cfg0_mtime = None

        # defect_images 目录枚举缓存：(dir_path, mtime) -> set(文件名)
        self._defect_dir_listing_cache = {}
        # 产线状态刷新（低频）
        self._line_state_timer = QTimer(self)
        self._line_state_timer.timeout.connect(self._refresh_line_state)
        self._line_state_timer.start(500)
        self._create_strip4_controls()
        self._init_strip_count_ui()
        self._init_scrollable_strip_layout()
        self._init_external_axis_canvases()
        self.apply_strip_layout(self.system_count)
        self._hydrate_production_inputs_from_config0()
        self._setup_fukuan_status_panel()
        self._install_production_focus_navigation()
        self._install_detection_image_double_click_open()

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
            "幅宽状态说明（对称容差，偏窄/偏宽均计为超出范围）：\n"
            f"· 允许范围：实测须在 [设定−带宽, 设定+带宽] mm 内，"
            f"带宽=max({FUKUAN_NARROW_ABS_MM:.0f}mm, 设定×{FUKUAN_NARROW_REL:.1%})；\n"
            f"· 报警：最近 {FUKUAN_ALARM_WINDOW} 帧全部超出允许范围；\n"
            f"· 预警：末尾连续≥{FUKUAN_WARN_TAIL_STREAK}帧超出，且未满 {FUKUAN_ALARM_WINDOW} 帧全超；\n"
            "· 注意：当前帧超出允许范围，但末尾连续未达预警；\n"
            "· 正常：当前在允许范围内，且未触发报警。"
        )
        for lbl in self._small_fukuan_labels():
            lbl.setToolTip(tip)

    def _format_fukuan_status_richtext(
        self,
        baseline,
        last_mm,
        tail_streak,
        has_alarm,
        active,
        has_waveform_points,
        protected=False,
        raw_mm=None,
        mode=None,
        reason=None,
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
        lo = fukuan_narrow_threshold_mm(baseline)
        hi = fukuan_wide_threshold_mm(baseline)
        delta = last_mm - baseline
        delta_txt = f"{delta:+.1f}"
        if baseline:
            pct_txt = f" ({delta / baseline * 100.0:+.1f}%)"
        else:
            pct_txt = ""
        sub_lines = []
        if protected:
            try:
                raw_txt = f"{float(raw_mm):.1f}" if raw_mm is not None and float(raw_mm) == float(raw_mm) else "NaN"
            except Exception:
                raw_txt = "NaN"
            m = str(mode or "")
            r = str(reason or "")
            hint = f"本帧测量异常已保护（Raw={raw_txt}mm"
            if m:
                hint += f"，mode={m}"
            if r:
                hint += f"，reason={r}"
            hint += "）"
            sub_lines.append(hint)
        oor = fukuan_out_of_range_mm(last_mm, baseline)
        if has_alarm:
            tier = "报警"
            tier_color = "#c0392b"
            sub_lines.append(
                f"最近 {FUKUAN_ALARM_WINDOW} 帧均超出允许范围 [{lo:.0f}, {hi:.0f}] mm"
            )
        elif tail_streak >= FUKUAN_WARN_TAIL_STREAK and oor:
            tier = "预警"
            tier_color = "#d35400"
            sub_lines.append(
                f"末尾已连续 {tail_streak} 帧超出允许范围 [{lo:.0f}, {hi:.0f}] mm"
            )
        elif oor:
            tier = "注意"
            tier_color = "#b7950b"
            sub_lines.append(f"当前超出允许范围 [{lo:.0f}, {hi:.0f}] mm")
        else:
            tier = "正常"
            tier_color = "#1e8449"
        extra = ""
        if sub_lines:
            extra = "<br/>" + "<br/>".join(
                [f"<span style='font-size:9px;color:#566573'>{s}</span>" for s in sub_lines[:2]]
            )
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
            self.system_count = min(4, max(1, raw_count))
        except Exception:
            self.system_count = 3
        if hasattr(self, "strip_count_combo"):
            self.strip_count_combo.blockSignals(True)
            self.strip_count_combo.setCurrentText(str(self.system_count))
            self.strip_count_combo.blockSignals(False)

    def _hydrate_production_inputs_from_config0(self):
        """从 config0 按物理条带读入，再映射到 UI 左→右槽位（与 save_config01 写入对称）。"""
        cfg_path = os.path.join(_REPO_ROOT, "config", "config0.yaml")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
        except Exception:
            return
        n = _clamp_strip_count_ui(int(c.get("strip_count", getattr(self, "system_count", 3)) or 3))
        prev = int(getattr(self, "system_count", n))
        self.system_count = n
        if hasattr(self, "strip_count_combo"):
            self.strip_count_combo.blockSignals(True)
            self.strip_count_combo.setCurrentText(str(n))
            self.strip_count_combo.blockSignals(False)
        if prev != n:
            try:
                self.apply_strip_layout(n)
            except Exception:
                pass
        if hasattr(self, "conduct_id"):
            self.conduct_id.setText(str(c.get("conduct_id", "") or "").strip())
        for ui_slot in range(1, 5):
            fe = getattr(self, f"fukuan_{ui_slot}", None)
            if fe is None:
                continue
            if ui_slot <= n:
                truth = _truth_strip_index_1based(ui_slot, n)
                raw = c.get(f"fukuan_{truth}", 0)
                try:
                    x = float(raw or 0)
                except (TypeError, ValueError):
                    x = 0.0
                if x <= 0:
                    fe.setText("")
                else:
                    fe.setText(str(int(x)) if x == int(x) else str(x))
            else:
                fe.setText("")
        if hasattr(self, "strip_card_edits"):
            for idx, ed in enumerate(self.strip_card_edits):
                ui_slot = idx + 1
                if ui_slot <= n:
                    truth = _truth_strip_index_1based(ui_slot, n)
                    ed.setText(str(c.get(f"strip_card_{truth}", "") or "").strip())
                else:
                    ed.setText("")

    def _production_focus_chain_widgets(self):
        """质保书号 → 幅宽1→卡号1→幅宽2→卡号2…（仅当前条数下可见框，符合从左到右、先幅宽后卡号的填写习惯）。"""
        chain = []
        if hasattr(self, "conduct_id"):
            chain.append(self.conduct_id)
        n = self._visible_count()
        for i in range(n):
            fw = getattr(self, f"fukuan_{i + 1}", None)
            if fw is not None:
                chain.append(fw)
            if hasattr(self, "strip_card_edits") and i < len(self.strip_card_edits):
                chain.append(self.strip_card_edits[i])
        return chain

    def _setup_production_tab_order(self):
        """仅串联质保书号与幅宽、卡号之间的 Tab 顺序（Shift+Tab 反向）。"""
        chain = self._production_focus_chain_widgets()
        if len(chain) < 2:
            return
        for i in range(len(chain) - 1):
            try:
                self.setTabOrder(chain[i], chain[i + 1])
            except Exception:
                pass

    def _refresh_production_field_focus_policy(self):
        """隐藏条对应的幅宽/卡号不参与焦点，避免 Tab 误入未启用列。"""
        n = self._visible_count()
        if hasattr(self, "conduct_id"):
            self.conduct_id.setFocusPolicy(Qt.StrongFocus)
        for i in range(1, 5):
            w = getattr(self, f"fukuan_{i}", None)
            if w is None:
                continue
            w.setFocusPolicy(Qt.StrongFocus if i <= n else Qt.NoFocus)
        if hasattr(self, "strip_card_edits"):
            for j, ed in enumerate(self.strip_card_edits, start=1):
                ed.setFocusPolicy(Qt.StrongFocus if j <= n else Qt.NoFocus)

    def _install_production_focus_navigation(self):
        widgets = []
        if hasattr(self, "conduct_id"):
            widgets.append(self.conduct_id)
        for i in range(1, 5):
            w = getattr(self, f"fukuan_{i}", None)
            if w is not None:
                widgets.append(w)
        if hasattr(self, "strip_card_edits"):
            widgets.extend(self.strip_card_edits)
        if not self._production_nav_filter_installed:
            for w in widgets:
                try:
                    w.installEventFilter(self)
                except Exception:
                    pass
            self._production_nav_filter_installed = True
        self._refresh_production_field_focus_policy()
        self._setup_production_tab_order()

    def _production_focus_prev_next(self, current, delta):
        chain = self._production_focus_chain_widgets()
        if not chain:
            return False
        try:
            i = chain.index(current)
        except ValueError:
            return False
        j = (i + delta) % len(chain)
        chain[j].setFocus()
        try:
            if isinstance(chain[j], QLineEdit):
                chain[j].selectAll()
        except Exception:
            pass
        return True

    @staticmethod
    def _lineedit_at_left_boundary(le):
        if le.selectedText():
            return le.selectionStart() == 0
        return le.cursorPosition() <= 0

    @staticmethod
    def _lineedit_at_right_boundary(le):
        t = le.text()
        ln = len(t)
        if le.selectedText():
            return le.selectionStart() + len(le.selectedText()) >= ln
        return le.cursorPosition() >= ln

    def _defect_image_label_strip_index(self, obj):
        """若为缺陷/实时图相关 QLabel，返回 ('up'|'down', strip_idx)；否则 None。"""
        for i, lb in enumerate(self._up_show_labels()):
            if obj is lb:
                return "up", i
        for i, lb in enumerate(self._down_show_labels()):
            if obj is lb:
                return "down", i
        for i, lb in enumerate(self._up_realtime_labels()):
            if obj is lb:
                return "up", i
        for i, lb in enumerate(self._down_realtime_labels()):
            if obj is lb:
                return "down", i
        for i, lb in enumerate(self._up_click_labels()):
            if obj is lb:
                return "up", i
        for i, lb in enumerate(self._down_click_labels()):
            if obj is lb:
                return "down", i
        return None

    def _resolve_double_click_defect_image_path(self, surface, strip_idx):
        prev = (
            self._preview_image_path_up if surface == "up" else self._preview_image_path_down
        )
        p = prev[strip_idx] if 0 <= strip_idx < len(prev) else None
        if p and os.path.isfile(p):
            return p
        pts = self.defect_points_up if surface == "up" else self.defect_points_down
        if 0 <= strip_idx < len(pts):
            for pt in reversed(pts[strip_idx]):
                base = (pt.get("path") or "").strip()
                fn = (pt.get("file") or "").strip()
                if base and fn:
                    fp = os.path.normpath(os.path.join(base, fn))
                    if os.path.isfile(fp):
                        return fp
        return None

    def _install_detection_image_double_click_open(self):
        targets = []
        try:
            targets.extend(self._up_show_labels())
            targets.extend(self._down_show_labels())
            targets.extend(self._up_realtime_labels())
            targets.extend(self._down_realtime_labels())
            targets.extend(self._up_click_labels())
            targets.extend(self._down_click_labels())
        except Exception:
            return
        for lb in targets:
            if lb is None:
                continue
            try:
                lb.installEventFilter(self)
            except Exception:
                pass

    def eventFilter(self, obj, event):
        try:
            et = event.type()
        except Exception:
            return False
        if et == QEvent.MouseButtonDblClick:
            hit = self._defect_image_label_strip_index(obj)
            if hit is not None:
                surface, strip_idx = hit
                img_path = self._resolve_double_click_defect_image_path(surface, strip_idx)
                if img_path and _open_image_path_with_system_viewer(img_path):
                    return True
                return False
        if et == QEvent.KeyPress and isinstance(obj, QLineEdit):
            try:
                if obj not in self._production_focus_chain_widgets():
                    return False
            except Exception:
                return False
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                d = 1 if key == Qt.Key_Down else -1
                if self._production_focus_prev_next(obj, d):
                    return True
                return False
            if key == Qt.Key_Right:
                if self._lineedit_at_right_boundary(obj):
                    if self._production_focus_prev_next(obj, 1):
                        return True
                return False
            if key == Qt.Key_Left:
                if self._lineedit_at_left_boundary(obj):
                    if self._production_focus_prev_next(obj, -1):
                        return True
                return False
        return False

    def _layout_production_config_strip(self):
        """
        顶部生产配置区栅格布局：消除 4 条幅宽与产品型号重叠；
        标签列与输入列纵向分组，标签在首行、输入在次行，同列水平居中对齐。
        """
        _R = QRect
        # 配置区宽度与高度（四行：标签行 / 输入行 / 带钢卡号行 / 功能按钮行）
        fw, fh = 971, 168
        self.frame.setGeometry(QtCore.QRect(self.frame.x(), self.frame.y(), fw, fh))

        row1_y, row1_h = 6, 28
        row2_y, row2_h = 40, 38
        # 与幅宽输入框同高，确保视觉一致
        row3_y, row3_h = 82, 38
        row4_y, row4_h = 124, 30

        conduct_x, conduct_w = 10, 190
        gap_after_conduct = 10
        # 四条幅宽等宽排列；宽度需能完整显示占位符「幅宽N(mm)」
        fukuan_w, fukuan_gap = 105, 5
        fukuan_x0 = conduct_x + conduct_w + gap_after_conduct
        fukuan_block_w = 4 * fukuan_w + 3 * fukuan_gap
        fukuan_end = fukuan_x0 + fukuan_block_w

        gap_before_combo = 10
        combo_x = fukuan_end + gap_before_combo
        combo_w = 210
        exch_x = combo_x + combo_w + 10
        exch_w = 82

        # 质保书号：标签与输入框同一列中心对齐
        self.label_ID.setGeometry(_R(conduct_x, row1_y, conduct_w, row1_h))
        self.label_ID.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.conduct_id.setGeometry(_R(conduct_x, row2_y, conduct_w, row2_h))

        # 幅宽：主标签与右侧说明「从东向西…」分列，避免与首列幅宽输入重叠
        self.label_ID_6.setGeometry(_R(fukuan_x0, row1_y, 76, row1_h))
        self.label_ID_6.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        if hasattr(self, "label_fukuan_order_hint"):
            hint_x = fukuan_x0 + 78
            hint_w = max(120, min(280, combo_x - hint_x - 6))
            self.label_fukuan_order_hint.setGeometry(_R(hint_x, row1_y, hint_w, row1_h))
            self.label_fukuan_order_hint.setAlignment(Qt.AlignLeft | Qt.AlignBottom)

        fx = fukuan_x0
        for i, w in enumerate(
            (self.fukuan_1, self.fukuan_2, self.fukuan_3, self.fukuan_4), start=1
        ):
            w.setGeometry(_R(fx, row2_y, fukuan_w, row2_h))
            fx += fukuan_w + fukuan_gap

        # 带钢卡号：放在对应幅宽正下方，一一对应
        if hasattr(self, "label_strip_card"):
            self.label_strip_card.setGeometry(_R(fukuan_x0, row3_y, 80, row3_h))
            self.label_strip_card.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        fx = fukuan_x0
        for w in getattr(self, "strip_card_edits", []):
            w.setGeometry(_R(fx, row3_y, fukuan_w, row3_h))
            fx += fukuan_w + fukuan_gap

        # 带钢条数：标签与下拉同一行（首行），不占用第二行
        self.strip_count_label.setGeometry(_R(470, row1_y, 82, row1_h))
        self.strip_count_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.strip_count_combo.setGeometry(_R(552, row1_y, 56, row1_h))

        # 产品型号：标签与下拉左对齐
        lbl_pw_w = 76
        self.label_ID_7.setGeometry(_R(combo_x, row1_y, lbl_pw_w, row1_h))
        self.label_ID_7.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.product_cls_combo.setGeometry(_R(combo_x, row2_y, combo_w, row2_h))

        # 第四行（row4）：三个分类入口按钮，独占一行，与产品型号下拉左对齐
        # 与 row2/row3 错开，彻底消除与输入区重叠
        b3_x = combo_x        # 与「产品型号」下拉框左边缘对齐
        btn_gap3 = 8

        # 1. 缺陷分类向导（小白主入口，最显眼）
        if hasattr(self, "btn_cls_wizard"):
            wiz_w = 124
            self.btn_cls_wizard.setGeometry(_R(b3_x, row4_y, wiz_w, row4_h))
            b3_x += wiz_w + btn_gap3

        # 2. 类别配置（专业入口）
        if hasattr(self, "btn_cls_config"):
            cfg_w = 88
            self.btn_cls_config.setGeometry(_R(b3_x, row4_y, cfg_w, row4_h))
            b3_x += cfg_w + btn_gap3

        # 3. 分类训练（专业入口）
        if hasattr(self, "btn_cls_train"):
            tr_w = 88
            self.btn_cls_train.setGeometry(_R(b3_x, row4_y, tr_w, row4_h))

        # 参数设置（确认）保持在右上角 row1
        self.para_config01.setGeometry(_R(878, row1_y, 75, 30))
        # 切换按钮保持在 row2 右侧，不受 row3 影响
        self.button_exchange.setGeometry(_R(exch_x, row2_y, exch_w, row2_h))

        # 下方条带滚动区随配置区高度下移，避免被遮挡
        if hasattr(self, "strip_scroll_area"):
            top = self.frame.y() + self.frame.height() + 6
            self.strip_scroll_area.setGeometry(QRect(0, top, 1761, 900))

    def _init_strip_count_ui(self):
        # 与生产卡号/幅宽配置放在同一配置区
        input_font = QFont("Arial", 14)
        self.fukuan_1.setFont(input_font)
        self.fukuan_2.setFont(input_font)
        self.fukuan_3.setFont(input_font)
        self.fukuan_1.setPlaceholderText("幅宽1(mm)")
        self.fukuan_2.setPlaceholderText("幅宽2(mm)")
        self.fukuan_3.setPlaceholderText("幅宽3(mm)")
        try:
            self.label_ID_6.setText("幅宽及卡号")
            self.label_ID_6.setToolTip("请在下方填写各条带钢的幅宽与对应带钢卡号（与条数一一对应）。")
            self.label_fukuan_order_hint = QLabel("（从东向西顺序依次输入）", self.frame)
            self.label_fukuan_order_hint.setStyleSheet(
                "QLabel { color: #5a5a5a; font: 12px 'Arial'; background: transparent; border: none; }"
            )
            self.label_fukuan_order_hint.setToolTip(
                "幅宽1、幅宽2…与带钢卡号1、2…按产线从东向西的顺序一一对应填写。"
            )
        except Exception:
            pass
        # 左上角：固定文字「质保书号」为 QLabel，输入在下一行，避免被误认为整格都是输入框
        self.label_ID.setText("质保书号")
        self.label_ID.setStyleSheet(
            "QLabel { background: transparent; color: #1a1a1a; border: none; }"
        )
        self.label_ID.setToolTip("固定说明文字；请在下方输入框填写质保书号。")
        self.conduct_id.setPlaceholderText("请输入质保书号")

        # 隐藏 mainui 生成的 QLineEdit（保留对象避免潜在引用报错）
        self.product_cls.hide()

        self.label_ID_7.setText("产品型号")
        self.label_ID_7.setToolTip(
            "【产品型号】\n"
            "下拉为「显示名称 [编号]」；保存到 config0 的仍为数字编号（对应 data{N}）。\n"
            "显示名称在「类别配置」中维护（rptcfg 的 product_cls_names）。\n"
            "允收矩阵在类别配置中编辑，经 make_standard 生成 table.json。"
        )

        # 用 QComboBox 替换原 QLineEdit，选项来自 rptcfg 中已有的 data{N} 键
        self.product_cls_combo = QComboBox(self.frame)
        self.product_cls_combo.setFont(input_font)
        self.product_cls_combo.setEditable(True)
        self.product_cls_combo.setToolTip(
            "从列表选择可看到显示名称；也可直接输入编号。\n"
            "名称在「类别配置」中设置；底层仍用 data{编号}。"
        )
        self._refresh_product_cls_combo()
        self._sync_product_cls_combo_from_config0()

        self.btn_cls_config = QPushButton("类别配置", self.frame)
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
            "打开类别配置窗口（需密码，见 config/auth.yaml 的 cls_config）。\n"
            "在此维护产品型号（data{N}）、缺陷类别名称和允收矩阵，\n"
            "保存后可生成 table.json 供报告判定使用。"
        )
        self.btn_cls_config.clicked.connect(self._open_cls_config)
        self._cls_config_window = None

        self.btn_cls_train = QPushButton("分类训练", self.frame)
        self.btn_cls_train.setFont(font_btn)
        self.btn_cls_train.setStyleSheet(
            "QPushButton { background-color: #E3F2FD; color: #1565C0;"
            " border: 1px solid #90CAF9; border-radius: 4px;"
            " padding-left: 14px; padding-right: 12px; }"
            "QPushButton:hover { background-color: #BBDEFB; }"
        )
        self.btn_cls_train.setToolTip(
            "打开分类训练与模型管理。\n"
            "可配置训练集（按类别文件夹）、后台一键训练并启用分类模型。"
        )
        self.btn_cls_train.clicked.connect(self._open_cls_train)
        self._cls_train_window = None

        self.btn_cls_wizard = QPushButton("缺陷分类向导", self.frame)
        self.btn_cls_wizard.setFont(font_btn)
        self.btn_cls_wizard.setStyleSheet(
            "QPushButton { background-color: #FFF3E0; color: #E65100;"
            " border: 1px solid #FFCC80; border-radius: 4px;"
            " padding-left: 14px; padding-right: 12px; }"
            "QPushButton:hover { background-color: #FFE0B2; }"
        )
        self.btn_cls_wizard.setToolTip("推荐：工人小白模式，一步一步完成缺陷类型配置与训练准备。")
        self.btn_cls_wizard.clicked.connect(self._open_cls_wizard)
        self._cls_wizard_window = None

        self.fukuan_4 = QLineEdit(self.frame)
        self.fukuan_4.setFont(input_font)
        self.fukuan_4.setPlaceholderText("幅宽4(mm)")

        # 带钢卡号：每条带钢对应一个名称输入（位于幅宽正下方）
        self.label_strip_card = QLabel("带钢卡号(对应幅宽)", self.frame)
        self.label_strip_card.setStyleSheet("font: 14px 'Arial'; font-weight: bold;")
        self.label_strip_card.setToolTip("请输入每条带钢对应的名称（带钢卡号），并与上方幅宽一一对应。")
        self.strip_card_edits = []
        for i in range(1, 5):
            ed = QLineEdit(self.frame)
            ed.setFont(QFont("Arial", 14))
            ed.setPlaceholderText(f"带钢卡号{i}")
            ed.setToolTip("该条带钢名称/带钢卡号（将用于主界面显示与报告显示）。")
            self.strip_card_edits.append(ed)

        self.strip_count_label = QLabel("带钢条数", self.frame)
        self.strip_count_label.setStyleSheet("font: 16px 'Arial'; font-weight: bold;")
        self.strip_count_combo = QComboBox(self.frame)
        self.strip_count_combo.addItems(["1", "2", "3", "4"])
        self.strip_count_combo.blockSignals(True)
        self.strip_count_combo.setCurrentText(str(self.system_count))
        self.strip_count_combo.blockSignals(False)
        self.strip_count_combo.currentIndexChanged.connect(self.apply_strip_count_preview)

        # 统一几何布局（解决幅宽4与产品型号重叠）
        self._layout_production_config_strip()
        # 初次加载时按条数隐藏多余的“带钢卡号”输入
        try:
            self.apply_strip_layout(self.system_count)
        except Exception:
            pass

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
        password, ok = QInputDialog.getText(
            self,
            "密码验证",
            "进入类别配置需要权限验证，请输入密码:",
            QLineEdit.Password,
        )
        if not ok:
            return
        if password != _read_auth_password("cls_config"):
            QMessageBox.warning(self, "密码错误", "请输入正确的密码！", QMessageBox.Ok)
            return
        if self._cls_config_window is None or not self._cls_config_window.isVisible():
            self._cls_config_window = ClsConfigWindow(self)
            self._cls_config_window.finished.connect(self._on_cls_config_closed)
        self._cls_config_window.show()
        self._cls_config_window.raise_()
        self._cls_config_window.activateWindow()

    def _on_cls_config_closed(self):
        """类别配置窗口关闭时刷新型号下拉列表。"""
        self._refresh_product_cls_combo()

    def _is_detect_running(self) -> bool:
        try:
            return (
                getattr(self, "python_process", None) is not None
                and self.python_process.state() != QProcess.NotRunning
            )
        except Exception:
            return False

    def _open_cls_train(self):
        """
        打开分类训练/模型管理窗口。
        训练与启用属于参数维护类权限，复用 auth.parameter_settings 的密码。
        """
        password, ok = QInputDialog.getText(
            self,
            "密码验证",
            "分类训练与模型管理（工程/工艺专用）\n请输入密码:",
            QLineEdit.Password,
        )
        if not ok:
            return
        if password != _read_auth_password("parameter_settings"):
            QMessageBox.warning(self, "密码错误", "请输入正确的密码！", QMessageBox.Ok)
            return

        if not hasattr(self, "_cls_train_window") or self._cls_train_window is None:
            self._cls_train_window = ClsTrainWindow(self, is_detect_running_fn=self._is_detect_running)
        self._cls_train_window.show()
        self._cls_train_window.raise_()
        self._cls_train_window.activateWindow()

    def _open_cls_wizard(self):
        """工人小白入口：缺陷分类向导（不需要密码）。"""
        if not hasattr(self, "_cls_wizard_window") or self._cls_wizard_window is None:
            self._cls_wizard_window = ClsWizardWindow(self, is_detect_running_fn=self._is_detect_running)
        self._cls_wizard_window.show()
        self._cls_wizard_window.raise_()
        self._cls_wizard_window.activateWindow()

    def apply_strip_count_preview(self, *_args):
        """切换带钢条数时立即刷新幅宽输入框与下方带钢显示区；不写 config0（检测前须点「确认」保存）。"""
        try:
            value = self.strip_count_combo.currentText()
            n = min(4, max(1, int(value)))
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
            self.system_count = min(4, max(1, int(value)))
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
        _top = int(self.frame.y() + self.frame.height() + 6)
        self.strip_scroll_area.setGeometry(QRect(0, _top, 1761, 900))
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
        visible_count = min(max(1, int(count)), 4)
        if hasattr(self, "fukuan_4"):
            self.fukuan_4.setVisible(visible_count == 4)
        # 幅宽输入框数量与条数同步：1条显示前1个，2条显示前2个，3条显示前3个，4条显示前4个
        self.fukuan_1.setVisible(True)
        self.fukuan_2.setVisible(visible_count >= 2)
        self.fukuan_3.setVisible(visible_count >= 3)
        if hasattr(self, "fukuan_4"):
            self.fukuan_4.setVisible(visible_count >= 4)
        # 带钢卡号输入框与条数同步
        if hasattr(self, "strip_card_edits"):
            for i, ed in enumerate(self.strip_card_edits, start=1):
                ed.setVisible(i <= visible_count)
        if hasattr(self, "label_strip_card"):
            self.label_strip_card.setVisible(True)
        row_h = 306
        row_gap = 16
        # 条带间距统一：不再单独给第一条额外留白
        first_row_extra_gap = 0
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

        # 同步每条带钢“显示标题”为带钢卡号（若已填写）
        try:
            cards = []
            if hasattr(self, "strip_card_edits"):
                cards = [ed.text().strip() for ed in self.strip_card_edits]
            self._apply_strip_card_titles(cards)
        except Exception:
            pass
        if getattr(self, "_production_nav_filter_installed", False):
            try:
                self._refresh_production_field_focus_policy()
                self._setup_production_tab_order()
            except Exception:
                pass

    @staticmethod
    def _format_vertical_card_text(s: str, max_chars: int = 6) -> str:
        s = str(s or "").strip()
        if not s:
            return ""
        raw = s[:max_chars]
        out = "\n".join(list(raw))
        if len(s) > max_chars:
            out += "\n…"
        return out

    def _apply_strip_card_titles(self, strip_cards):
        """
        将条带左侧窄标签改为优先显示“带钢卡号”。
        UI 原始标签很窄，采用竖排显示，避免重叠。
        """
        cards = list(strip_cards or [])
        cards = (cards + ["", "", "", ""])[:4]

        # strip1~3：来自 mainui.py 的 label_title_12/13/14
        for i, attr in enumerate(("label_title_12", "label_title_13", "label_title_14"), start=1):
            if not hasattr(self, attr):
                continue
            w = getattr(self, attr)
            card = str(cards[i - 1] or "").strip()
            txt = self._format_vertical_card_text(card) if card else f"钢\n带\n{i}"
            try:
                w.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                w.setWordWrap(True)
                w.setText(txt)
                w.setToolTip(card if card else f"钢带{i}")
            except Exception:
                pass

        # strip4：本工程动态创建 label_strip4_title
        if hasattr(self, "label_strip4_title"):
            card4 = str(cards[3] or "").strip()
            txt4 = self._format_vertical_card_text(card4) if card4 else "钢\n带\n4"
            try:
                self.label_strip4_title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.label_strip4_title.setWordWrap(True)
                self.label_strip4_title.setText(txt4)
                self.label_strip4_title.setToolTip(card4 if card4 else "钢带4")
            except Exception:
                pass

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

    def _sync_fukuan_external_axis_geometry(self, idx: int) -> None:
        """幅宽外置刻度轴 + 标题几何与 label_fukuan_* 对齐（对标缺陷分布外轴逻辑）。"""
        try:
            hosts = self._fukuan_labels()
            if idx >= len(hosts):
                return
            host = hosts[idx]
            if host is None:
                return
            fw_rect = host.geometry()

            axl = self.fukuan_axis_left_labels[idx] if idx < len(self.fukuan_axis_left_labels) else None
            axb = self.fukuan_axis_bottom_labels[idx] if idx < len(self.fukuan_axis_bottom_labels) else None
            tw = self.fukuan_width_titles[idx] if idx < len(self.fukuan_width_titles) else None
            tl = self.fukuan_len_titles[idx] if idx < len(self.fukuan_len_titles) else None

            if axl is not None:
                axl.setGeometry(QRect(max(0, fw_rect.x() - 39), fw_rect.y(), 40, fw_rect.height()))
                axl.show()
                axl.raise_()
            if axb is not None:
                axb.setGeometry(QRect(fw_rect.x(), fw_rect.y() + fw_rect.height() - 1, fw_rect.width(), 20))
                axb.show()
                axb.raise_()

            # 标题位置与缺陷外轴一致：宽度标题在左轴上方；长度标题在底轴数字下方
            if tw is not None:
                tw.setGeometry(QRect(max(0, fw_rect.x() - 62), max(0, fw_rect.y() - 18), 58, 14))
                tw.show()
                tw.raise_()
            if tl is not None:
                tl.setGeometry(QRect(fw_rect.x() + fw_rect.width() // 2 - 46, fw_rect.y() + fw_rect.height() + 22, 92, 14))
                tl.show()
                tl.raise_()
        except Exception:
            pass

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

        # 幅宽外置轴与标题也同步（对标缺陷分布窗口）
        self._sync_fukuan_external_axis_geometry(idx)

    def _visible_count(self):
        return min(max(1, int(self.system_count)), self.MAX_STRIPS)

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
        if getattr(self, "_ui_detection_running", False):
            QMessageBox.information(
                self,
                "无法换卷",
                "检测处于运行状态时不能换卷。请先点击「暂停」停止界面刷新后，再执行换卷。",
            )
            return
        # 兜底：确保界面刷新定时器停下（理论上暂停按钮已做，但这里防竞态/误触发）
        try:
            if getattr(self, "render_timer", None) is not None:
                self.render_timer.stop()
        except Exception:
            pass
        # 换卷开始：先禁用按钮防止重复点击
        try:
            self.button_exchange.setEnabled(False)
        except Exception:
            pass
        # 换卷属于“跨卷重置”，进入前先清空全部视觉区，避免残留上一卷画面/按钮/刻度
        try:
            self._clear_all_visuals()
        except Exception:
            pass
        self.conduct_id.clear()
        self.fukuan_1.clear()  # 清空第一个检测系统幅宽输入框
        self.fukuan_2.clear()  # 清空第二个检测系统幅宽输入框
        self.fukuan_3.clear()  # 清空第三个检测系统幅宽输入框
        self.fukuan_4.clear()  # 清空第四个检测系统幅宽输入框
        if hasattr(self, "strip_card_edits"):
            for ed in self.strip_card_edits:
                ed.clear()
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
            'strip_card_1': '',
            'strip_card_2': '',
            'strip_card_3': '',
            'strip_card_4': '',
            'strip_card_list': [],
            'confirmed_at': '',
            'product_cls': ''
        }
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'w', encoding='utf-8') as file:
                yaml.dump(data, file, allow_unicode=True)

            # 重置位置计数器
            self.pos = [0 for _ in range(self.MAX_STRIPS)]
            self.pos2 = [0 for _ in range(self.MAX_STRIPS)]
            # 换卷：缺陷分布滑动窗口、幅宽曲线缓存与外置坐标轴需与 __init__ 一致复位；
            # 否则上一卷的窗口刻度/红点仍留在界面上（create_button 在幅宽为 0 时不会清按钮）
            for _i in range(self.MAX_STRIPS):
                self.defect_points_up[_i] = []
                self.defect_points_down[_i] = []
                self.coordinate_queue[_i] = []
                self.coordinate_queue2[_i] = []
                self.wave_window_start_mm[_i] = 0.0
                self.wave_window_end_mm[_i] = 1720.0
                self.latest_defect_y[_i] = -1.0
                self.last_multiple[_i] = -1
                self.last_multiple2[_i] = -1
                self.cm[_i] = 0
                self.cm2[_i] = 0
                self.base_folder[_i] = None
                self.base_folder2[_i] = None
                self.fukuan_mm[_i] = 0.0
                self.fukuan_mm2[_i] = 0.0
                self.abnormal_status[_i] = False
                self.fukuan_last_measured[_i] = float("nan")
                self.fukuan_tail_narrow[_i] = 0
                self.display_end_indices[_i] = 0
                self.display_smooth_end[_i] = 0.0
                self.total_data[_i] = []
            for _i in range(self.MAX_STRIPS):
                self.refresh_buttons(_i)
                self.refresh_buttons2(_i)
            try:
                self.update_all_plots()
            except Exception:
                pass

            # 换卷：强制重启 Python 接收端，确保新一卷从零开始
            try:
                self._restart_python_receiver_for_roll_change()
            except Exception:
                pass

            QMessageBox.information(self, "输入信息",
                                    "已清空配置，请输入：\n"
                                    "• 质保书号\n"
                                    "• 需要的检测系统幅宽（不需要的设为0）\n"
                                    "• 各条带钢卡号（与幅宽一一对应）\n"
                                    "• 产品型号")
            try:
                self._confirmed_this_session = False
                self._refresh_start_stop_enabled()
                self._refresh_exchange_enabled()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.warning(self, "重置失败", f"重置配置文件时出错: {str(e)}")
        finally:
            # 换卷结束：由 _refresh_exchange_enabled 按运行态恢复可交互性
            try:
                self._refresh_exchange_enabled()
            except Exception:
                pass

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
            strip_cards = []
            if hasattr(self, "strip_card_edits"):
                strip_cards = [ed.text().strip() for ed in self.strip_card_edits]
            product_cls = (
                self._current_product_cls_key_from_combo()
                if hasattr(self, "product_cls_combo")
                else self.product_cls.text().strip()
            )

            print(
                f"输入数据 - 质保书号:'{conduct_id}', 幅宽1:'{fukuan_1_text}', 幅宽2:'{fukuan_2_text}', 幅宽3:'{fukuan_3_text}', 类别:'{product_cls}', 带钢卡号:{strip_cards}")

            # 验证必填字段
            if not conduct_id:
                QMessageBox.warning(self, "输入错误", "质保书号不能为空！")
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

            # 校验：启用的条带若幅宽>0，则必须填写对应带钢卡号
            strip_cards = (strip_cards + ["", "", "", ""])[:4]
            fws = [fukuan_1, fukuan_2, fukuan_3, fukuan_4]
            for i in range(self.system_count):
                if float(fws[i]) > 0 and not str(strip_cards[i]).strip():
                    QMessageBox.warning(self, "输入错误", f"第{i+1}条带钢已填写幅宽，但带钢卡号为空。请补充带钢卡号{i+1}。")
                    return

            # UI 左->右 与 物理(图像左->右) 的输入映射：仅在此处重排写入 config0，检测端算法不变
            n = int(self.system_count)
            phys_fw = [0.0, 0.0, 0.0, 0.0]
            phys_card = ["", "", "", ""]
            for ui_slot in range(1, n + 1):
                p = _truth_strip_index_1based(ui_slot, n)
                phys_fw[p - 1] = float(fws[ui_slot - 1])
                phys_card[p - 1] = str(strip_cards[ui_slot - 1] or "").strip()

            # 创建字典
            data = {
                'conduct_id': conduct_id,
                'strip_count': self.system_count,
                'fukuan_1': phys_fw[0],
                'fukuan_2': phys_fw[1],
                'fukuan_3': phys_fw[2],
                'fukuan_4': phys_fw[3],
                'fukuan_list': phys_fw[:self.system_count],
                'strip_card_1': phys_card[0],
                'strip_card_2': phys_card[1],
                'strip_card_3': phys_card[2],
                'strip_card_4': phys_card[3],
                'strip_card_list': phys_card[:self.system_count],
                'confirmed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'product_cls': product_cls
            }
            with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'w', encoding='utf-8') as file:
                yaml.dump(data, file, allow_unicode=True)

            print("配置保存成功！")
            # 本次会话已确认
            self._confirmed_this_session = True

            # 确认后：把条带标题/可见显示优先改成“带钢卡号”
            try:
                self._apply_strip_card_titles(strip_cards)
            except Exception:
                pass
            try:
                self._refresh_start_stop_enabled()
            except Exception:
                pass

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

    def _config_ready_for_start(self) -> bool:
        """未点击“确认”写入 config0 或关键字段不完整时，不允许开始/暂停。"""
        # 必须是“本次打开主界面”后点过确认，避免沿用上次残留配置误启动
        if not bool(getattr(self, "_confirmed_this_session", False)):
            return False
        try:
            with open(os.path.join(_REPO_ROOT, "config", "config0.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return False
        if not str(cfg.get("confirmed_at", "") or "").strip():
            return False
        if not str(cfg.get("conduct_id", "") or "").strip():
            return False
        if not str(cfg.get("product_cls", "") or "").strip():
            return False
        try:
            n = int(cfg.get("strip_count", 3) or 3)
            n = min(4, max(1, n))
        except Exception:
            n = 3
        fws = []
        for i in range(1, 5):
            try:
                fws.append(float(cfg.get(f"fukuan_{i}", 0) or 0))
            except Exception:
                fws.append(0.0)
        if not any(fws[i] > 0 for i in range(n)):
            return False
        cards = cfg.get("strip_card_list")
        if not isinstance(cards, (list, tuple)):
            cards = [
                str(cfg.get("strip_card_1", "") or "").strip(),
                str(cfg.get("strip_card_2", "") or "").strip(),
                str(cfg.get("strip_card_3", "") or "").strip(),
                str(cfg.get("strip_card_4", "") or "").strip(),
            ]
        cards = list(cards) + ["", "", "", ""]
        for i in range(n):
            if fws[i] > 0 and not str(cards[i] or "").strip():
                return False
        return True

    def _refresh_start_stop_enabled(self):
        ok = self._config_ready_for_start()
        try:
            self.pushButton_start.setEnabled(bool(ok))
        except Exception:
            pass
        try:
            self.pushButton_stop.setEnabled(bool(ok))
        except Exception:
            pass
        # start/stop 状态变化时，同步刷新换卷按钮可用性与颜色
        try:
            self._refresh_exchange_enabled()
        except Exception:
            pass

    def _refresh_exchange_enabled(self):
        """
        换卷按钮仅在“暂停态（未运行）”可交互。
        注意：换卷与“确认配置”无关；它只受运行态与后端进程切换期影响。
        """
        try:
            # 运行中不允许换卷（核心约束）
            enabled = not bool(getattr(self, "_ui_detection_running", False))

            # 进程处于启动/退出切换期时也禁用，避免竞态（尽量保守）
            proc = getattr(self, "python_process", None)
            if proc is not None:
                try:
                    st = proc.state()
                    if st == QProcess.Starting:
                        enabled = False
                except Exception:
                    pass

            enabled = bool(enabled)
            self.button_exchange.setEnabled(enabled)
            # 视觉状态：禁用时灰色，可用时恢复彩色
            try:
                if enabled:
                    # 可交互：高亮、字体加粗
                    self.button_exchange.setStyleSheet(
                        "QPushButton{background-color:#2F80ED;color:white;border-radius:6px;font-weight:600;}"
                        "QPushButton:hover{background-color:#256BD1;}"
                        "QPushButton:pressed{background-color:#1F5AB2;}"
                    )
                else:
                    # 不可用：灰色
                    self.button_exchange.setStyleSheet(
                        "QPushButton{background-color:#BDBDBD;color:#666666;border-radius:6px;font-weight:600;}"
                    )
            except Exception:
                pass
        except Exception:
            pass

    def _clear_all_visuals(self):
        """换卷时一次性清空所有视觉区，避免残留上一卷的任何图像/红点/刻度/曲线。"""
        # 1) 清空缺陷红点按钮（上/下表面）
        try:
            for i in range(self.MAX_STRIPS):
                try:
                    self.refresh_buttons(i)
                except Exception:
                    # refresh_buttons 内部已兜底，这里再兜底一次
                    self.buttons[i] = []
                try:
                    self.refresh_buttons2(i)
                except Exception:
                    self.buttons2[i] = []
        except Exception:
            pass

    def _restart_python_receiver_for_roll_change(self):
        """
        换卷专用：强制重启 Python 接收端，确保新一卷从“全新会话”开始。
        注意：这与“暂停/继续”不同；普通暂停不应杀进程。
        """
        # 换卷后应保持 paused=True，直到用户重新确认并点击开始
        try:
            _write_runtime_state(paused=True)
        except Exception:
            pass

        proc = getattr(self, "python_process", None)
        if proc is None:
            return
        try:
            if proc.state() != QProcess.NotRunning:
                try:
                    self._silent_disconnect_process_signals(proc)
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.waitForFinished(1500)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.python_process = None
        except Exception:
            pass
        try:
            self._refresh_exchange_enabled()
        except Exception:
            pass

        # 2) 清空缺陷分布主图（label_up_show / label_down_show）
        try:
            for lb in (self._up_show_labels() + self._down_show_labels()):
                if lb is None:
                    continue
                try:
                    lb.clear()
                    lb.setPixmap(QPixmap())
                except Exception:
                    pass
        except Exception:
            pass

        # 3) 清空实时小图与细节图（*_1 与 *_2）
        try:
            for lb in (
                self._up_realtime_labels()
                + self._down_realtime_labels()
                + self._up_click_labels()
                + self._down_click_labels()
            ):
                if lb is None:
                    continue
                try:
                    lb.clear()
                    lb.setPixmap(QPixmap())
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._preview_image_path_up = [None] * self.MAX_STRIPS
            self._preview_image_path_down = [None] * self.MAX_STRIPS
        except Exception:
            pass

        # 4) 清空缺陷分布外置刻度轴
        try:
            for ax in (list(getattr(self, "up_axis_left", []))
                       + list(getattr(self, "up_axis_bottom", []))
                       + list(getattr(self, "down_axis_left", []))
                       + list(getattr(self, "down_axis_bottom", []))):
                if ax is None:
                    continue
                try:
                    ax.clear()
                    ax.setPixmap(QPixmap())
                except Exception:
                    pass
        except Exception:
            pass

        # 5) 清空幅宽曲线与其外置坐标轴
        try:
            for i in range(self.MAX_STRIPS):
                host = None
                try:
                    host = self._fukuan_labels()[i]
                except Exception:
                    host = None
                if host is not None:
                    try:
                        host.clear()
                        host.setPixmap(QPixmap())
                    except Exception:
                        pass
                axl = None
                axb = None
                try:
                    axl = self.fukuan_axis_left_labels[i]
                    axb = self.fukuan_axis_bottom_labels[i]
                except Exception:
                    axl, axb = None, None
                for ax in (axl, axb):
                    if ax is None:
                        continue
                    try:
                        ax.clear()
                        ax.setPixmap(QPixmap())
                    except Exception:
                        pass
        except Exception:
            pass

    def _ensure_csharp_running(self) -> bool:
        """启动 / 召回 C# 多相机采集程序。优先环境变量，其次仓库内可执行文件。"""
        root = _PROJECT_ROOT
        csharp_exe = os.environ.get("MULTICAM_DEMO_EXE", "").strip() or None
        csharp_workdir = os.environ.get("MULTICAM_DEMO_CWD", "").strip() or None
        if csharp_exe and os.path.isfile(csharp_exe):
            if not csharp_workdir or not os.path.isdir(csharp_workdir):
                csharp_workdir = os.path.dirname(csharp_exe)
        else:
            ext = os.path.join(root, "external", "MultiCamDemo", "MultiCamDemo.exe")
            if os.path.isfile(ext):
                csharp_exe = ext
                csharp_workdir = os.path.join(root, "external", "MultiCamDemo")
            else:
                csharp_exe = None
                csharp_workdir = None
                for sub in ("Release", "Debug"):
                    d = os.path.join(root, "DalsaGrabDemoTcp", "MultiCamDemo", "bin", sub)
                    exe = os.path.join(d, "MultiCamDemo.exe")
                    if os.path.isfile(exe):
                        csharp_exe = exe
                        csharp_workdir = d
                        break
        if not csharp_exe or not os.path.isfile(csharp_exe):
            QMessageBox.critical(
                self,
                "启动失败",
                "未找到 MultiCamDemo.exe。\n\n请任选其一：\n"
                "1) 设置环境变量 MULTICAM_DEMO_EXE（及可选 MULTICAM_DEMO_CWD）\n"
                "2) 将程序放到 external/MultiCamDemo/MultiCamDemo.exe\n"
                "3) 在仓库内编译 DalsaGrabDemoTcp，生成 bin/Release 或 bin/Debug\n",
            )
            return False
        proc = getattr(self, "csharp_process", None)
        try:
            running = (
                proc is not None
                and proc.state() != QProcess.NotRunning
                and int(proc.processId() or 0) != 0
            )
        except Exception:
            running = False
        if running:
            return True
        try:
            if proc is not None:
                proc.kill()
        except Exception:
            pass
        self.csharp_process = QProcess(self)
        self.csharp_process.setProcessChannelMode(QProcess.MergedChannels)
        self.csharp_process.setWorkingDirectory(csharp_workdir)
        try:
            self.csharp_process.finished.connect(lambda *_: setattr(self, "csharp_process", None))
            self.csharp_process.errorOccurred.connect(lambda *_: setattr(self, "csharp_process", None))
        except Exception:
            pass
        self.csharp_process.start(csharp_exe)
        if not self.csharp_process.waitForStarted(5000):
            error_state = self.csharp_process.error()
            error_string = self.csharp_process.errorString()
            QMessageBox.critical(
                self,
                "启动失败",
                f"C# 程序启动失败！\n\n错误代码: {error_state}\n错误详情: {error_string}\n\n路径: {csharp_exe}",
            )
            self.csharp_process = None
            return False
        return True

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
        # 不使用 button_stop_click + 冗长 wait：主线程会长时间阻塞，表现为关窗卡死、python.exe 无响应。
        # 退出顺序：先断开子进程信号并 kill 检测端（释放 888x 端口），再快速结束监视线程。
        try:
            try:
                self.render_timer.stop()
            except Exception:
                pass
            try:
                _write_runtime_state(paused=True)
            except Exception:
                pass
            # 先结束检测端/C#：释放监听端口（避免仅靠杀主进程留下子 python 占位 888x）
            self._kill_child_processes_on_exit()
            self._force_stop_monitor_threads_for_exit()
        except Exception as e:
            print(f"[UI][exit] 关闭清理异常: {e}", flush=True)
        finally:
            event.accept()
        super().closeEvent(event)

    def _silent_disconnect_process_signals(self, proc) -> None:
        if proc is None:
            return
        for name in ("readyReadStandardOutput", "readyReadStandardError", "finished", "errorOccurred"):
            try:
                getattr(proc, name).disconnect()
            except Exception:
                pass

    def _join_stop_monitor_qthread(
        self,
        t,
        *,
        label="",
        allow_hard_kill=True,
        graceful_wait_ms=None,
    ) -> None:
        """
        主线程调用：向工作线程投递 _stop_in_thread，再 wait。
        allow_hard_kill=False：仅用于「暂停」，禁止 QThread.terminate()（易与 Matplotlib/Qt 状态机冲突导致整进程退出）。
        """
        if t is None:
            return
        if graceful_wait_ms is None:
            graceful_wait_ms = 450 if allow_hard_kill else 2800
        try:
            nm = t.objectName() or type(t).__name__
        except Exception:
            nm = "QThread"
        try:
            running0 = t.isRunning()
        except Exception:
            running0 = False
        tag = f"{label or nm}"
        print(f"[UI][stop] → 请求停止线程 {tag} allow_hard_kill={allow_hard_kill} grace_ms={graceful_wait_ms}", flush=True)
        try:
            if hasattr(t, "_is_running"):
                t._is_running = False
        except Exception:
            pass
        if not running0:
            print(f"[UI][stop]   线程 {tag} 本就未运行", flush=True)
            return
        try:
            QMetaObject.invokeMethod(t, "_stop_in_thread", Qt.QueuedConnection)
        except Exception as ex:
            print(f"[UI][stop]   invokeMethod 失败 {tag}: {ex}", flush=True)
        if not t.wait(int(graceful_wait_ms)):
            if allow_hard_kill:
                print(f"[UI][stop]   {tag} 优雅退出超时，使用 terminate()", flush=True)
                try:
                    t.terminate()
                except Exception as ex:
                    print(f"[UI][stop]   terminate() 异常 {tag}: {ex}", flush=True)
                t.wait(380)
            else:
                print(
                    f"[UI][stop][warn] {tag} 在 {graceful_wait_ms}ms 内未结束，未使用 terminate（防崩溃）；"
                    f"仍可继续用界面；若异常请再试一次暂停或关闭软件。",
                    flush=True,
                )
        try:
            done = not t.isRunning()
        except Exception:
            done = True
        print(f"[UI][stop] ← 线程 {tag} stopped_ok={done}", flush=True)

    def _force_stop_monitor_threads_for_exit(self) -> None:
        """关窗：停掉全部幅宽/缺陷线程（含未显示条数）。允许 terminate。"""
        for i in range(self.MAX_STRIPS):
            self._join_stop_monitor_qthread(
                self.waveform_threads[i] if i < len(self.waveform_threads) else None,
                label=f"waveform[{i}]",
                allow_hard_kill=True,
                graceful_wait_ms=450,
            )
            self._join_stop_monitor_qthread(
                self.loader_thread[i] if i < len(self.loader_thread) else None,
                label=f"loader_up[{i}]",
                allow_hard_kill=True,
                graceful_wait_ms=450,
            )
            self._join_stop_monitor_qthread(
                self.loader_thread2[i] if i < len(self.loader_thread2) else None,
                label=f"loader_dn[{i}]",
                allow_hard_kill=True,
                graceful_wait_ms=450,
            )
            if i < len(self.waveform_threads):
                self.waveform_threads[i] = None
            if i < len(self.loader_thread):
                self.loader_thread[i] = None
            if i < len(self.loader_thread2):
                self.loader_thread2[i] = None

    def _kill_child_processes_on_exit(self) -> None:
        """务必结束 QProcess 子进程，否则检测端 python 会继续占用监听端口。"""
        # 1) Python 检测端（持有 TCP listen）
        p = getattr(self, "python_process", None)
        if p is not None:
            try:
                self._silent_disconnect_process_signals(p)
                if p.state() != QProcess.NotRunning:
                    print("[UI][exit] 正在结束 Python 接收端子进程…", flush=True)
                    p.kill()
                    if not p.waitForFinished(2500):
                        print("[UI][exit][warn] Python 子进程未在 2.5s 内退出，可能需任务管理器结束残留 python.exe", flush=True)
            except Exception as e:
                print(f"[UI][exit] 结束 Python 子进程出错: {e}", flush=True)
            finally:
                self.python_process = None

        # 2) C# 发送端
        cs = getattr(self, "csharp_process", None)
        if cs is not None:
            try:
                self._silent_disconnect_process_signals(cs)
                if cs.state() != QProcess.NotRunning:
                    cs.terminate()
                    if not cs.waitForFinished(1200):
                        cs.kill()
                        cs.waitForFinished(600)
            except Exception as e:
                print(f"[UI][exit] 结束 C# 子进程出错: {e}", flush=True)
            finally:
                self.csharp_process = None

    def _gather_pause_workers(self):
        """当前可见系统的幅宽线程 + 上下表面缺陷负载线程列表。"""
        workers = []
        n = self._visible_count()
        for i in range(n):
            t = self.waveform_threads[i] if i < len(self.waveform_threads) else None
            workers.append((t, f"waveform sys{i+1}"))
        for i in range(n):
            lt = self.loader_thread[i] if i < len(self.loader_thread) else None
            workers.append((lt, f"loader_up sys{i+1}"))
            lt2 = self.loader_thread2[i] if i < len(self.loader_thread2) else None
            workers.append((lt2, f"loader_dn sys{i+1}"))
        return workers

    def _pause_wait_workers_with_pumping(self, workers, deadline_ms=2800):
        """主线程泵事件，避免「逐线程 wait×3s」把界面卡死；工作线程应在长循环中响应 _is_running。"""
        app = QApplication.instance()
        et = QElapsedTimer()
        et.start()
        while et.elapsed() < int(deadline_ms):
            alive = any(t is not None and t.isRunning() for (t, _lab) in workers)
            if not alive:
                print("[UI][stop] 所有监视线程已协作退出。", flush=True)
                return True
            if app is not None:
                ms_left = max(1, int(deadline_ms) - int(et.elapsed()))
                slice_ms = min(48, ms_left)
                try:
                    app.processEvents(QEventLoop.AllEvents, slice_ms)
                except Exception:
                    try:
                        app.processEvents()
                    except Exception:
                        pass
            else:
                time.sleep(0.02)
        return False

    def button_stop_click(self):
        """暂停：不杀检测子进程；派发停止后短时泵事件，避免主线程卡顿。"""
        print("[UI][stop] ========== 点击「暂停/停止检测界面」==========", flush=True)
        try:
            try:
                self.render_timer.stop()
            except Exception as e:
                print(f"[UI][stop] render_timer.stop: {e}", flush=True)

            workers = self._gather_pause_workers()
            for t, tag in workers:
                if t is None:
                    continue
                try:
                    if hasattr(t, "_is_running"):
                        t._is_running = False
                except Exception:
                    pass
                try:
                    if t.isRunning():
                        QMetaObject.invokeMethod(t, "_stop_in_thread", Qt.QueuedConnection)
                        print(f"[UI][stop] 已派发 _stop → {tag}", flush=True)
                except Exception as ex:
                    print(f"[UI][stop] invokeMethod 失败 {tag}: {ex}", flush=True)

            _write_runtime_state(paused=True)
            self._set_run_state(False)

            cooperative = self._pause_wait_workers_with_pumping(workers, deadline_ms=2800)

            still = [(t, lab) for (t, lab) in workers if t is not None and t.isRunning()]
            if still:
                if not cooperative:
                    print(
                        f"[UI][stop][warn] {len(still)} 个线程未及时退出（可能此前卡在长循环）；将 terminate",
                        flush=True,
                    )
                for t, tag in still:
                    try:
                        print(f"[UI][stop] terminate() → {tag}", flush=True)
                        t.terminate()
                        t.wait(400)
                        print(f"[UI][stop] ← {tag} stopped_ok={not t.isRunning()}", flush=True)
                    except Exception as ex:
                        print(f"[UI][stop] terminate 异常 {tag}: {ex}", flush=True)

            print(
                "已暂停：UI 停止刷新，接收端保活并从历史长度继续计数",
                flush=True,
            )
            print("[UI][stop] ========== 暂停流程结束 ==========", flush=True)
        except Exception:
            print("[UI][stop][FATAL] button_stop_click 未捕获异常:", flush=True)
            traceback.print_exc()

    def terminate_processes(self):
        """
        兼容旧调用：与关窗时一致，强制结束子进程并释放端口。
        """
        self._kill_child_processes_on_exit()



    def _safe_stop_monitor_threads(self):
        """开始检测前停止幅宽/缺陷读取线程与渲染定时器，避免重复点开始导致信号与线程倍增。"""
        try:
            self.render_timer.stop()
        except Exception:
            pass
        self._force_stop_monitor_threads_for_exit()

    def  button_start_click(self):
        # 继续/启动：
        # - 若接收端已在运行，则仅取消暂停（不重启、不清零长度）
        # - 若未运行，则正常启动（并确保 paused=False）
        if not self._config_ready_for_start():
            QMessageBox.information(
                self,
                "配置未确认",
                "请先填写质保书号、幅宽与带钢卡号，并点击「确认」保存配置后再开始检测。",
            )
            self._set_run_state(False)
            self._refresh_start_stop_enabled()
            return
        try:
            if getattr(self, "python_process", None) is not None and self.python_process.state() != QProcess.NotRunning:
                # Python 已在运行：这里是“继续”，但 C# 可能被手动关闭，需要尝试召回
                try:
                    self._ensure_csharp_running()
                except Exception:
                    pass
                _write_runtime_state(paused=False)
            else:
                _write_runtime_state(paused=False)
                self.start_programs()
        except Exception as e:
            print(f"button_start_click 启动分支异常: {e}")
        # 启动成功后再显示运行
        try:
            running = getattr(self, "python_process", None) is not None and self.python_process.state() != QProcess.NotRunning
        except Exception:
            running = False
        self._set_run_state(bool(running))

        try:
            self._safe_stop_monitor_threads()
        except Exception:
            pass

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        camid = config["camrea_id_up_cls"]
        camid2 = config["camrea_id_down_cls"]

        with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as file:
            config0 = yaml.safe_load(file)

        # 为各检测系统分别创建上下表面的缺陷检测线程
        self._load_system_count_from_config()
        threads = []
        for i in range(self._visible_count()):
            detection_system_index = i + 1
            n = _clamp_strip_count_ui(int(config0.get("strip_count", 3) or 3))
            truth = _truth_strip_index_1based(detection_system_index, n)
            fukuan_key = f"fukuan_{truth}"
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
        self.render_timer.start(80)
        self.showfukuan()

    def _layout_status_panel(self) -> None:
        """右上角：产线状态 + 检测状态，两行对齐、胶囊式状态值。"""
        if getattr(self, "_status_panel_laid_out", False):
            return
        self._status_panel_laid_out = True

        gb = self.groupBox_2
        gb.setTitle("系统状态")
        # 必须落在 frame_2 内，且与左侧「报告生成」按钮留出间隙（避免左边框与图标/标题挤在一起）
        try:
            fr = self.frame_2
            fw = max(120, int(fr.width()))
            fh = max(80, int(fr.height()))
            right_m = 10
            gap_after_report = 16
            report_right = 0
            try:
                bg = self.pushButton_report.geometry()
                br = int(bg.x() + bg.width())
                # 「报告生成」文字标签可能比图标按钮更靠右，取二者最大右边界避免与系统状态框挤在一起
                lr = 0
                try:
                    lg = self.label_ID_5.geometry()
                    lr = int(lg.x() + lg.width())
                except Exception:
                    pass
                report_right = max(br, lr)
            except Exception:
                pass
            min_x = (report_right + gap_after_report) if report_right > 0 else 4

            panel_w = min(168, fw - min_x - right_m)
            panel_w = max(118, panel_w)
            x = fw - panel_w - right_m
            if x < min_x:
                x = min_x
                panel_w = max(118, fw - x - right_m)

            panel_h = min(108, max(96, fh - 6))
            gb.setGeometry(QRect(x, 2, panel_w, panel_h))
        except Exception:
            gb.setGeometry(QRect(600, 2, 160, 102))

        gb.setFont(QFont("Microsoft YaHei UI", 10))
        gb.setStyleSheet(
            "QGroupBox#groupBox_2 { font-weight: 600; border: 1px solid #cfd8dc; border-radius: 6px; "
            "margin-top: 8px; background: #fafafa; }"
            "QGroupBox#groupBox_2::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; "
            "color: #37474f; }"
        )

        outer = QVBoxLayout(gb)
        outer.setContentsMargins(8, 18, 8, 8)
        outer.setSpacing(6)

        lbl_font = QFont("Microsoft YaHei UI", 9)
        lbl_style = "color:#607d8b;"

        lbl_line = QLabel("产线状态")
        lbl_line.setFont(lbl_font)
        lbl_line.setStyleSheet(lbl_style)
        lbl_line.setMinimumWidth(52)
        lbl_line.setMaximumWidth(52)
        lbl_line.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 去掉 designer 里 Arial 20pt + 强制调色板，否则胶囊会被撑出父控件
        pill_font = QFont("Microsoft YaHei UI", 10, QFont.DemiBold)
        self.run_state.setFont(pill_font)
        self.run_state.setPalette(QApplication.palette())

        self.line_state = QLabel("静止")
        self.line_state.setFont(pill_font)
        self.line_state.setObjectName("line_state_pill")
        self.line_state.setAlignment(Qt.AlignCenter)
        fm = QFontMetrics(pill_font)
        pill_w = (
            max(
                fm.horizontalAdvance("运行"),
                fm.horizontalAdvance("暂停"),
                fm.horizontalAdvance("静止"),
            )
            + 12
        )
        pill_h = fm.height() + 8
        self.line_state.setFixedSize(pill_w, pill_h)
        self.line_state.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._apply_line_state_style(False)

        h1 = QHBoxLayout()
        h1.setSpacing(6)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(lbl_line, 0, Qt.AlignVCenter)
        h1.addStretch(1)
        h1.addWidget(self.line_state, 0, Qt.AlignVCenter)

        lbl_det = QLabel("检测状态")
        lbl_det.setFont(lbl_font)
        lbl_det.setStyleSheet(lbl_style)
        lbl_det.setMinimumWidth(52)
        lbl_det.setMaximumWidth(52)
        lbl_det.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.run_state.setObjectName("detect_state_pill")
        self.run_state.setAlignment(Qt.AlignCenter)
        self.run_state.setFixedSize(pill_w, pill_h)
        self.run_state.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._set_run_state(False)

        h2 = QHBoxLayout()
        h2.setSpacing(6)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(lbl_det, 0, Qt.AlignVCenter)
        h2.addStretch(1)
        h2.addWidget(self.run_state, 0, Qt.AlignVCenter)

        outer.addLayout(h1)
        outer.addLayout(h2)

    def _apply_line_state_style(self, running: bool) -> None:
        if not getattr(self, "line_state", None):
            return
        if running:
            self.line_state.setStyleSheet(
                "QLabel#line_state_pill { color:#0d47a1; background:#e3f2fd; "
                "border:1px solid #90caf9; border-radius:8px; padding:2px 6px; }"
            )
        else:
            self.line_state.setStyleSheet(
                "QLabel#line_state_pill { color:#546e7a; background:#eceff1; "
                "border:1px solid #cfd8dc; border-radius:8px; padding:2px 6px; }"
            )

    def _apply_detect_state_style(self, running: bool) -> None:
        try:
            if running:
                self.run_state.setStyleSheet(
                    "QLabel#detect_state_pill { color:#1b5e20; background:#e8f5e9; "
                    "border:1px solid #81c784; border-radius:8px; padding:2px 6px; }"
                )
            else:
                self.run_state.setStyleSheet(
                    "QLabel#detect_state_pill { color:#b71c1c; background:#ffebee; "
                    "border:1px solid #ef9a9a; border-radius:8px; padding:2px 6px; }"
                )
        except Exception:
            pass

    def _set_run_state(self, running: bool) -> None:
        """统一设置右上角检测状态，避免分支遗漏导致状态错乱。"""
        self._ui_detection_running = bool(running)
        try:
            self.run_state.setText("运行" if running else "暂停")
            self._apply_detect_state_style(running)
        except Exception:
            pass
        try:
            self._refresh_exchange_enabled()
        except Exception:
            pass

    def _refresh_line_state(self) -> None:
        """产线状态：若最近一段时间收到图片 => 运行，否则静止。"""
        try:
            if not getattr(self, "line_state", None):
                return
            ts = _read_line_heartbeat_ts()
            now = time.time()
            running = (ts > 0) and ((now - ts) <= 2.0)
            self.line_state.setText("运行" if running else "静止")
            self._apply_line_state_style(running)
        except Exception:
            pass


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
                n = _clamp_strip_count_ui(int(config0.get("strip_count", 3) or 3))
                truth = _truth_strip_index_1based(detection_system_index, n)
                fukuan_key = f"fukuan_{truth}"
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
                if not self.fukuan_plot_canvases[i]:
                    # 幅宽窗口“完全对标缺陷分布窗口”的结构：
                    # - 直接用 label_fukuan_i 自身作为绘图画布（QPixmap + QPainter）
                    # - 外置坐标轴不嵌入布局，而是在同一 strip frame 内按几何贴边摆放（与缺陷外轴一致）
                    self.fukuan_plot_canvases[i] = fukuan_labels[i]

                    # 外置坐标轴的父对象与缺陷分布一致：放在 strip frame（label 的 parent）里
                    parent_frame = fukuan_labels[i].parent()
                    if parent_frame is None:
                        parent_frame = fukuan_labels[i]

                    if self.fukuan_axis_left_labels[i] is None:
                        self.fukuan_axis_left_labels[i] = QLabel(parent_frame)
                        self.fukuan_axis_left_labels[i].setStyleSheet("background-color: transparent;")
                        self.fukuan_axis_left_labels[i].show()

                    if self.fukuan_axis_bottom_labels[i] is None:
                        self.fukuan_axis_bottom_labels[i] = QLabel(parent_frame)
                        self.fukuan_axis_bottom_labels[i].setStyleSheet("background-color: transparent;")
                        self.fukuan_axis_bottom_labels[i].show()

                    # 幅宽外置轴标题（对标缺陷分布的标题样式与位置）
                    if self.fukuan_width_titles[i] is None:
                        tw = QLabel("宽度(mm)", parent_frame)
                        tw.setStyleSheet("font: 9pt 'Arial'; color: #1e2a39; background-color: transparent;")
                        tw.setAlignment(Qt.AlignCenter)
                        self.fukuan_width_titles[i] = tw
                    if self.fukuan_len_titles[i] is None:
                        tl = QLabel("长度 (m)", parent_frame)
                        tl.setStyleSheet("font: 8pt 'Arial'; color: #1e2a39; background-color: transparent;")
                        tl.setAlignment(Qt.AlignCenter)
                        self.fukuan_len_titles[i] = tl

                    # 参照缺陷分布外轴的几何逻辑（_sync_external_axis_geometry）：
                    # 左轴右边线 == 幅宽窗口左边线；下轴上边线 == 幅宽窗口下边线
                    fw_rect = fukuan_labels[i].geometry()
                    self.fukuan_axis_left_labels[i].setGeometry(QRect(max(0, fw_rect.x() - 39), fw_rect.y(), 40, fw_rect.height()))
                    self.fukuan_axis_bottom_labels[i].setGeometry(QRect(fw_rect.x(), fw_rect.y() + fw_rect.height() - 1, fw_rect.width(), 20))
                    self.fukuan_axis_left_labels[i].raise_()
                    self.fukuan_axis_bottom_labels[i].raise_()

                    # 标题初次摆放；后续在 apply_strip_layout() 内由 _sync_fukuan_external_axis_geometry() 统一刷新
                    self._sync_fukuan_external_axis_geometry(i)

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

    def plot_waveform(self, data, has_abnormal, detection_system_index, last_mm, tail_streak, meta):
        try:
            system_index = detection_system_index - 1
            self.total_data[system_index] = data
            if len(data) > TOTAL_DATA_MAX_SAMPLES:
                self.total_data[system_index] = data[-TOTAL_DATA_MAX_SAMPLES:]
            self.abnormal_status[system_index] = has_abnormal
            self.fukuan_last_measured[system_index] = last_mm
            self.fukuan_tail_narrow[system_index] = tail_streak
            # meta: 由检测端输出（Raw/Stable/来源/原因）
            if isinstance(meta, dict):
                try:
                    self.fukuan_last_raw[system_index] = float(meta.get("raw")) if meta.get("raw") is not None else float("nan")
                except Exception:
                    self.fukuan_last_raw[system_index] = float("nan")
                try:
                    self.fukuan_last_valid[system_index] = bool(meta.get("valid", True))
                except Exception:
                    self.fukuan_last_valid[system_index] = True
                self.fukuan_last_reason[system_index] = str(meta.get("reason", "") or "")
                self.fukuan_last_mode[system_index] = str(meta.get("mode", "") or "")
        except Exception as e:
            print(f"plot_waveform系统{detection_system_index}发生错误：{e}")

    @staticmethod
    def _read_ui_defect_display_config(cfg=None):
        """从 config.yaml 读取缺陷窗口/轴映射参数；cfg 为已 load 的字典时可传入避免重复读盘。"""
        defaults = {
            "target_window_m": 30.0,
            "max_window_mm": 300_000.0,
            "backtrack_max_mm": 150_000.0,
            "lag_margin_mm": 20_000.0,
            "nonlinear": False,
            "tail_phys_ratio": 0.35,
            "tail_pixel_ratio": 0.2,
        }
        if cfg is None:
            try:
                with open(
                    os.path.join(_REPO_ROOT, "config", "config.yaml"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
        try:
            return {
                "target_window_m": float(
                    cfg.get("ui_defect_window_target_m", defaults["target_window_m"])
                ),
                "max_window_mm": float(
                    cfg.get("ui_defect_window_max_mm", defaults["max_window_mm"])
                ),
                "backtrack_max_mm": float(
                    cfg.get("ui_defect_backtrack_max_mm", defaults["backtrack_max_mm"])
                ),
                "lag_margin_mm": float(
                    cfg.get("ui_defect_lag_margin_mm", defaults["lag_margin_mm"])
                ),
                "nonlinear": bool(cfg.get("ui_defect_axis_nonlinear", False)),
                "tail_phys_ratio": float(
                    cfg.get("ui_defect_axis_tail_phys_ratio", defaults["tail_phys_ratio"])
                ),
                "tail_pixel_ratio": float(
                    cfg.get("ui_defect_axis_tail_pixel_ratio", defaults["tail_pixel_ratio"])
                ),
            }
        except Exception:
            return dict(defaults)

    def update_all_plots(self):
        """
        统一渲染入口：由 QTimer 驱动，处理所有可见系统（2~4条）的渲染逻辑。
        合并了逻辑计算和 Matplotlib 绘图，代码量大，但函数数量最少。
        """
        full_cfg = {}
        try:
            p = getattr(self, "_plot_cfg_yaml_path", os.path.join(_REPO_ROOT, "config", "config.yaml"))
            mtime = os.path.getmtime(p)
            if getattr(self, "_plot_cfg_mtime", None) != mtime or getattr(self, "_plot_cfg_cache", None) is None:
                with open(p, "r", encoding="utf-8") as f:
                    self._plot_cfg_cache = yaml.safe_load(f) or {}
                self._plot_cfg_mtime = mtime
            full_cfg = self._plot_cfg_cache or {}
        except Exception:
            full_cfg = {}
        self._ui_defect_cfg = self._read_ui_defect_display_config(full_cfg)
        ui = self._ui_defect_cfg
        TARGET_X_WINDOW_M = ui["target_window_m"]

        config0 = {}
        try:
            p0 = getattr(self, "_plot_cfg0_yaml_path", os.path.join(_REPO_ROOT, "config", "config0.yaml"))
            m0 = os.path.getmtime(p0)
            if getattr(self, "_plot_cfg0_mtime", None) != m0 or getattr(self, "_plot_cfg0_cache", None) is None:
                with open(p0, "r", encoding="utf-8") as f:
                    self._plot_cfg0_cache = yaml.safe_load(f) or {}
                self._plot_cfg0_mtime = m0
            config0 = self._plot_cfg0_cache or {}
        except Exception:
            config0 = {}

        # 浮点游标平滑参数：每帧至少前进 min_step 个采样点，追赶 gap 时按 alpha 比例 + 上限 max_step
        SMOOTH_MIN_STEP = 0.025
        SMOOTH_ALPHA = 0.14
        SMOOTH_MAX_STEP = 10.0

        # 循环处理可见系统（2~4条）
        for system_index in range(self._visible_count()):
            try:
                detection_system_index = system_index + 1
                small_fukuan_labels = self._small_fukuan_labels()

                # --- 1. 配置读取和初始检查（config0 已在帧首按 mtime 缓存）---
                n = _clamp_strip_count_ui(int(config0.get("strip_count", 3) or 3))
                truth = _truth_strip_index_1based(detection_system_index, n)
                baseline_width = config0.get(f"fukuan_{truth}", 0)

                raw_data = self.total_data[system_index]
                total_len = len(raw_data)

                canvas = self.fukuan_plot_canvases[system_index]
                if canvas is None:
                    continue  # 确保幅宽内置画布存在

                # --- 2. 状态标签和非活跃/无数据处理 ---
                if baseline_width <= 0 or total_len < 2:
                    self.display_smooth_end[system_index] = 0.0
                    # 清空幅宽曲线画布
                    try:
                        canvas_w = max(1, canvas.width())
                        canvas_h = max(1, canvas.height())
                        pm_clear = QPixmap(canvas_w, canvas_h)
                        pm_clear.fill(QtCore.Qt.white)
                        painter_clear = QPainter(pm_clear)
                        painter_clear.setFont(QFont("Arial", 10))
                        painter_clear.setPen(QtCore.Qt.gray)
                        if baseline_width <= 0:
                            painter_clear.drawText(
                                8, int(canvas_h * 0.45), canvas_w - 16,
                                30, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                                "无钢带（幅宽设置为0）"
                            )
                        painter_clear.end()
                        canvas.setPixmap(pm_clear)
                    except Exception:
                        pass
                    # 标签更新
                    if baseline_width <= 0:
                        small_fukuan_labels[system_index].setText(
                            self._format_fukuan_status_richtext(
                                0, float("nan"), 0, False, False, False
                            )
                        )
                        # 即使无钢带，也刷新外置坐标轴（保持 UI 结构一致）
                        total_len_ok = False
                    else:
                        small_fukuan_labels[system_index].setText(
                            self._format_fukuan_status_richtext(
                                baseline_width,
                                self.fukuan_last_measured[system_index],
                                self.fukuan_tail_narrow[system_index],
                                self.abnormal_status[system_index],
                                True,
                                total_len >= 2,
                                protected=(not bool(self.fukuan_last_valid[system_index])),
                                raw_mm=self.fukuan_last_raw[system_index],
                                mode=self.fukuan_last_mode[system_index],
                                reason=self.fukuan_last_reason[system_index],
                            )
                        )
                        total_len_ok = False

                    # 幅宽外置坐标轴：在无/少数据时仍绘制（避免“坐标轴消失”）
                    left_lbl = self.fukuan_axis_left_labels[system_index] if system_index < len(self.fukuan_axis_left_labels) else None
                    bottom_lbl = self.fukuan_axis_bottom_labels[system_index] if system_index < len(self.fukuan_axis_bottom_labels) else None
                    if left_lbl is not None and bottom_lbl is not None and left_lbl.width() > 10 and bottom_lbl.width() > 10:
                        try:
                            button_w_px = 10
                            button_h_px = 10
                            left_pad = 0
                            bottom_pad = 0

                            # X 外置刻度（长度 m）
                            bottom_w = bottom_lbl.width()
                            bottom_h = bottom_lbl.height()
                            bottom_pm = QPixmap(bottom_w, bottom_h)
                            bottom_pm.fill(QtCore.Qt.transparent)
                            painter_b = QPainter(bottom_pm)
                            pen = QPen(QtCore.Qt.black)
                            painter_b.setPen(pen)
                            painter_b.setFont(QFont("Arial", 8))
                            x_max_px = max(1, bottom_w - button_w_px - left_pad)
                            window_start_mm = float(self.wave_window_start_mm[system_index])
                            window_end_mm = float(self.wave_window_end_mm[system_index])
                            window_len_mm = max(1e-9, window_end_mm - window_start_mm)
                            painter_b.drawLine(0, 0, x_max_px, 0)
                            for i_tick in range(11):
                                length_mm = window_start_mm + (i_tick / 10.0) * window_len_mm
                                x_rel = int(
                                    round(
                                        _defect_length_mm_to_px(
                                            length_mm,
                                            window_start_mm,
                                            window_end_mm,
                                            x_max_px,
                                            ui,
                                        )
                                    )
                                )
                                x_rel = max(0, min(x_max_px, x_rel))
                                length_m = length_mm / 1000.0
                                painter_b.drawLine(x_rel, 0, x_rel, 6)
                                text_x = max(0, min(bottom_w - 42, x_rel - 21))
                                painter_b.drawText(text_x, 6, 42, 12, Qt.AlignHCenter | Qt.AlignTop, f"{length_m:.2f}")
                            painter_b.end()
                            bottom_lbl.setPixmap(bottom_pm)

                            # Y 外置刻度（幅宽 mm）
                            y_full = float(self.fukuan_mm[system_index]) if (self.fukuan_mm[system_index] and self.fukuan_mm[system_index] > 0) else float(baseline_width)
                            if y_full <= 0:
                                y_full = 600.0
                            left_w = left_lbl.width()
                            left_h = left_lbl.height()
                            left_pm = QPixmap(left_w, left_h)
                            left_pm.fill(QtCore.Qt.transparent)
                            painter_l = QPainter(left_pm)
                            painter_l.setPen(pen)
                            painter_l.setFont(QFont("Arial", 8))
                            y_max_px = max(1, left_h - button_h_px - bottom_pad)
                            painter_l.drawLine(left_w - 1, 0, left_w - 1, y_max_px)
                            for i_tick in range(7):
                                y_pos = int(round(i_tick * y_max_px / 6.0))
                                y_pos = max(0, min(y_max_px, y_pos))
                                width_mm = (i_tick / 6.0) * y_full
                                painter_l.drawLine(left_w - 7, y_pos, left_w - 1, y_pos)
                                text_rect_y = max(0, min(left_h - 16, y_pos - 8))
                                painter_l.drawText(0, text_rect_y, left_w - 9, 16, Qt.AlignRight | Qt.AlignVCenter, f"{width_mm:.0f}")
                            painter_l.end()
                            left_lbl.setPixmap(left_pm)
                        except Exception:
                            pass

                    continue

                # --- 3. 流动呈现逻辑（浮点游标 + 插值，窗口连续左移）---
                ratio = float(full_cfg.get("cam1_standard_ratio_x") or 0.0)
                if abs(ratio) < 1e-12:
                    ratio = 1.0

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
                # 右边界：持续追最新长度；左边界：在缺陷跟不上时“锚定上一轮仍有点的窗口起点”
                desired_start_f = max(0.0, end_f - float(N_window - 1))
                desired_start_mm = float(desired_start_f * mm_per_idx)
                end_mm = float(end_f * mm_per_idx)

                latest_y_mm = float(self.latest_defect_y[system_index])
                # 初始值 -1 表示尚未收到缺陷点；此时不做锚定
                if latest_y_mm >= 0.0 and latest_y_mm < desired_start_mm:
                    prev_start_mm = float(self.wave_window_start_mm[system_index])
                    start_mm = prev_start_mm if prev_start_mm >= 0.0 else desired_start_mm
                else:
                    start_mm = desired_start_mm

                # 限制最大窗口长度，避免锚定导致窗口无限拉长
                max_window_mm = float(ui["max_window_mm"])
                if end_mm - start_mm > max_window_mm:
                    start_mm = end_mm - max_window_mm

                # 检测滞后：缓存内仍有较小 y 的点时，向左扩展窗口以包住晚到的坐标（上下表面合并）
                lag_margin = float(ui["lag_margin_mm"])
                ymins = []
                for _p in self.defect_points_up[system_index]:
                    try:
                        ymins.append(float(_p["y"]))
                    except (KeyError, TypeError, ValueError):
                        pass
                for _p in self.defect_points_down[system_index]:
                    try:
                        ymins.append(float(_p["y"]))
                    except (KeyError, TypeError, ValueError):
                        pass
                if ymins:
                    y_min_cache = min(ymins)
                    if y_min_cache < float(start_mm) + 1e-6:
                        start_mm = min(float(start_mm), y_min_cache - lag_margin)
                    start_mm = max(0.0, float(start_mm))
                    if end_mm - start_mm > max_window_mm:
                        start_mm = end_mm - max_window_mm

                # 与缺陷图共用同一长度窗口（mm，与波形索引一致）
                start_mm = max(0.0, float(start_mm))
                start_f = start_mm / mm_per_idx if mm_per_idx != 0 else 0.0
                self.wave_window_start_mm[system_index] = start_mm
                self.wave_window_end_mm[system_index] = end_mm
                current_multiple = int(self.wave_window_start_mm[system_index] // 1720)
                self.cm[system_index] = current_multiple
                self.cm2[system_index] = current_multiple

                # --- 4. 数据插值与 X 轴（连续滑动）：仅对窗口内索引切片，避免 np.arange(total_len) 全量 ---
                lo = int(max(0, np.floor(start_f)))
                hi = int(min(total_len - 1, np.ceil(end_f)))
                if hi < lo:
                    lo, hi = 0, max(0, total_len - 1)
                idx_axis = np.arange(lo, hi + 1, dtype=float)
                raw_arr = np.asarray(raw_data[lo : hi + 1], dtype=float)
                n_samples = max(64, min(512, int(max(2, (end_f - start_f) * 6)) + 1))
                xi = np.linspace(start_f, end_f, n_samples)
                xi = np.clip(xi, float(lo), float(hi))
                y_plot = np.interp(xi, idx_axis, raw_arr)
                x_plot = (xi - start_f) * x_step
                abs_x_offset = start_f * x_step

                # --- 5. QPainter 绘图和样式更新 ---

                # 更新状态面板文本
                small_fukuan_labels[system_index].setText(
                    self._format_fukuan_status_richtext(
                        baseline_width,
                        self.fukuan_last_measured[system_index],
                        self.fukuan_tail_narrow[system_index],
                        self.abnormal_status[system_index],
                        True,
                        total_len >= 2,
                        protected=(not bool(self.fukuan_last_valid[system_index])),
                        raw_mm=self.fukuan_last_raw[system_index],
                        mode=self.fukuan_last_mode[system_index],
                        reason=self.fukuan_last_reason[system_index],
                    )
                )

                # 内置绘图的像素映射：与外置刻度、缺陷红点同一套 mm->px（支持非线性尾段压缩）
                button_w_px = 10
                button_h_px = 10
                left_pad = 0
                bottom_pad = 0

                canvas_w = max(1, canvas.width())
                canvas_h = max(1, canvas.height())
                x_max_px = max(1, canvas_w - button_w_px - left_pad)
                y_max_px = max(1, canvas_h - button_h_px - bottom_pad)

                # tick/y 映射上限：保持与 refresh_scale()/外置刻度同一口径
                y_full = float(self.fukuan_mm[system_index]) if (self.fukuan_mm[system_index] and self.fukuan_mm[system_index] > 0) else float(baseline_width)
                if y_full <= 0:
                    y_full = 1.0

                y_plot_clamped = np.clip(y_plot, 0.0, y_full) if len(y_plot) > 0 else np.asarray([], dtype=float)

                pm = QPixmap(canvas_w, canvas_h)
                pm.fill(QtCore.Qt.white)
                painter_plot = QPainter(pm)
                painter_plot.setRenderHint(QPainter.Antialiasing, True)

                # 基准线（绿色虚线）
                baseline_y_pix = int(round((baseline_width / y_full) * y_max_px)) if y_full > 0 else 0
                baseline_y_pix = max(0, min(y_max_px, baseline_y_pix))
                pen_base = QPen(QtCore.Qt.green)
                pen_base.setWidth(2)
                pen_base.setStyle(QtCore.Qt.DashLine)
                painter_plot.setPen(pen_base)
                painter_plot.drawLine(0, baseline_y_pix, x_max_px, baseline_y_pix)

                # 幅宽曲线（蓝色折线）
                pen_curve = QPen(QtCore.Qt.blue)
                pen_curve.setWidth(1)
                painter_plot.setPen(pen_curve)

                if len(x_plot) > 1 and len(y_plot_clamped) == len(x_plot):
                    # x_plot 为相对窗口起点的长度（米）；绝对 mm = start_mm + 米*1000
                    px_prev = None
                    py_prev = None
                    for xv_m, yi in zip(x_plot, y_plot_clamped):
                        y_abs_mm = float(start_mm) + float(xv_m) * 1000.0
                        x_pix = int(
                            round(
                                _defect_length_mm_to_px(
                                    y_abs_mm, float(start_mm), float(end_mm), x_max_px, ui
                                )
                            )
                        )
                        y_val = float(yi)
                        y_pix = int(round((y_val / y_full) * y_max_px))
                        x_pix = max(0, min(x_max_px, x_pix))
                        y_pix = max(0, min(y_max_px, y_pix))
                        if px_prev is not None:
                            painter_plot.drawLine(px_prev, py_prev, x_pix, y_pix)
                        px_prev = x_pix
                        py_prev = y_pix

                painter_plot.end()
                canvas.setPixmap(pm)

                # 外置坐标轴（与幅宽波形横纵范围同源）
                left_lbl = self.fukuan_axis_left_labels[system_index] if system_index < len(self.fukuan_axis_left_labels) else None
                bottom_lbl = self.fukuan_axis_bottom_labels[system_index] if system_index < len(self.fukuan_axis_bottom_labels) else None
                if left_lbl is not None and bottom_lbl is not None and left_lbl.width() > 10 and bottom_lbl.width() > 10:
                    try:
                        # ---- X（长度 m）外置刻度 ----
                        button_w_px = 10  # 与 refresh_scale() 对齐
                        button_h_px = 10
                        left_pad = 0
                        bottom_pad = 0
                        bottom_w = bottom_lbl.width()
                        bottom_h = bottom_lbl.height()
                        bottom_pm = QPixmap(bottom_w, bottom_h)
                        bottom_pm.fill(QtCore.Qt.transparent)
                        painter_b = QPainter(bottom_pm)
                        pen = QPen(QtCore.Qt.black)
                        painter_b.setPen(pen)
                        painter_b.setFont(QFont("Arial", 8))
                        x_max_px = max(1, bottom_w - button_w_px - left_pad)
                        window_start_mm = float(self.wave_window_start_mm[system_index])
                        window_end_mm = float(self.wave_window_end_mm[system_index])
                        window_len_mm = max(1e-9, window_end_mm - window_start_mm)
                        painter_b.drawLine(0, 0, x_max_px, 0)
                        for i in range(11):
                            length_mm = window_start_mm + (i / 10.0) * window_len_mm
                            x_rel = int(
                                round(
                                    _defect_length_mm_to_px(
                                        length_mm,
                                        window_start_mm,
                                        window_end_mm,
                                        x_max_px,
                                        ui,
                                    )
                                )
                            )
                            x_rel = max(0, min(x_max_px, x_rel))
                            length_m = length_mm / 1000.0
                            painter_b.drawLine(x_rel, 0, x_rel, 6)
                            text_x = max(0, min(bottom_w - 42, x_rel - 21))
                            painter_b.drawText(text_x, 6, 42, 12, Qt.AlignHCenter | Qt.AlignTop, f'{length_m:.2f}')
                        painter_b.end()
                        bottom_lbl.setPixmap(bottom_pm)

                        # ---- Y（幅宽 mm）外置刻度 ----
                        y_full = float(self.fukuan_mm[system_index]) if (self.fukuan_mm[system_index] and self.fukuan_mm[system_index] > 0) else float(baseline_width)
                        if y_full <= 0:
                            y_full = 1.0
                        left_w = left_lbl.width()
                        left_h = left_lbl.height()
                        left_pm = QPixmap(left_w, left_h)
                        left_pm.fill(QtCore.Qt.transparent)
                        painter_l = QPainter(left_pm)
                        painter_l.setPen(pen)
                        painter_l.setFont(QFont("Arial", 8))
                        y_max_px = max(1, left_h - button_h_px - bottom_pad)
                        painter_l.drawLine(left_w - 1, 0, left_w - 1, y_max_px)
                        for i in range(7):
                            y_pos = int(round(i * y_max_px / 6.0))
                            y_pos = max(0, min(y_max_px, y_pos))
                            width_mm = (i / 6.0) * y_full
                            painter_l.drawLine(left_w - 7, y_pos, left_w - 1, y_pos)
                            text_rect_y = max(0, min(left_h - 16, y_pos - 8))
                            painter_l.drawText(0, text_rect_y, left_w - 9, 16, Qt.AlignRight | Qt.AlignVCenter, f'{width_mm:.0f}')
                        painter_l.end()
                        left_lbl.setPixmap(left_pm)
                    except Exception:
                        # UI 绘制失败不影响检测逻辑
                        pass

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
            # 启动/继续前：取消暂停（接收端读取该运行态控制）
            _write_runtime_state(paused=False)

            with open(os.path.join(_PROJECT_ROOT, "config", "config0.yaml"), "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            missing_configs = []

            # 检查基本必需配置（质保书号与产品型号）
            if not config.get('conduct_id') or config['conduct_id'] == '':
                missing_configs.append("质保书号")
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

                if getattr(self, "python_process", None) is not None and self.python_process.state() != QProcess.NotRunning:
                    print("Python 接收端已在运行，跳过重启（将直接继续）")
                else:
                    self.python_process = QProcess(self)
                    self.python_process.setProcessChannelMode(QProcess.MergedChannels)
                    self.python_process.readyReadStandardOutput.connect(self.handle_output)
                    try:
                        self.python_process.finished.connect(self._on_python_finished)
                    except Exception:
                        pass
                    try:
                        self.python_process.errorOccurred.connect(self._on_python_error)
                    except Exception:
                        pass
                    self.python_process.setWorkingDirectory(_PROJECT_ROOT)
                    args = ["-u", "-m", "app.online.detect_anomalies_online"]
                    print(f"[UI] 启动检测端: {python_exe} {' '.join(args)}")
                    print(f"[UI] 检测端工作目录: {self.python_process.workingDirectory()}")
                    self.python_process.start(python_exe, args)
                    if not self.python_process.waitForStarted(3000):
                        QMessageBox.critical(
                            self,
                            "启动失败",
                            "Python 检测程序启动失败，请检查解释器路径/脚本路径/环境依赖。",
                        )
                        return

                try:
                    with open(os.path.join(_PROJECT_ROOT, "config", "config0.yaml"), "r", encoding="utf-8") as _cf:
                        _c0 = yaml.safe_load(_cf) or {}
                    _cid = str(_c0.get("conduct_id", "") or "")
                    _date = datetime.now().strftime("%Y%m%d")
                    _exp = os.path.join(_PROJECT_ROOT, "detect result", _date, _cid)
                    if not os.path.isdir(_exp):
                        print(f"[UI][warn] 检测端已启动，但结果根目录尚不存在: {_exp}")
                        print("  - 若持续不存在：通常是检测端启动即报错退出 / conduct_id 不一致 / 权限问题")
                except Exception:
                    pass

                # =========================
                # 2) 启动/召回 C# 多相机程序（MultiCamDemo.exe，本仓库 DalsaGrabDemoTcp）
                # =========================
                if not self._ensure_csharp_running():
                    return

                self._set_run_state(True)
                active_str = "、".join(active_systems)
                print(f"系统启动成功，激活的系统: {active_str}")
                for i in range(self.system_count):
                    fw = fukuan_values[i]
                    print(f"系统{i+1}幅宽: {fw}mm {'(激活)' if fw > 0 else '(未激活)'}")

                # ---- 关键诊断：UI camid 与检测端输出目录命名必须一致 ----
                try:
                    with open(os.path.join(_PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as _f:
                        _cfg = yaml.safe_load(_f) or {}
                    up_id = int(_cfg.get("camrea_id_up_cls", -1))
                    down_id = int(_cfg.get("camrea_id_down_cls", -1))
                    # 检测端 detect_anomalies_online.py 使用 0-based CAM2/CAM3 输出为 上表面/下表面，
                    # UI 这里的约定是 1-based：2->上表面，3->下表面
                    if (up_id, down_id) != (2, 3):
                        print(
                            f"[UI][warn] 当前 camrea_id_up_cls/down_cls={up_id}/{down_id}。\n"
                            f"  - UI 约定: 2->上表面, 3->下表面\n"
                            f"  - 若配置不一致，UI 会去错误目录读缺陷文件，表现为“文件夹有但没缺陷信息”。"
                        )
                except Exception:
                    pass
        except Exception as e:
            print(f"start_programs发生错误：{e}")

    def _on_python_finished(self, exitCode, exitStatus):
        try:
            st = int(exitStatus)
        except Exception:
            st = exitStatus
        print(f"[UI][proc] 检测端退出: exitCode={exitCode}, exitStatus={st}")

    def _on_python_error(self, err):
        # 只要触发过 errorOccurred，基本可以判定“检测端未正常运行”
        try:
            code = int(err)
        except Exception:
            code = err
        print(f"[UI][proc] 检测端 QProcess 错误: {code}")

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
            file_name = self.find_file_with_coordinates(path, int(x), int(y)) or ""
            self.defect_points_up[list_index].append({
                "x": float(x),
                "y": float(y),
                "fukuan": float(fukuan_for_point) if fukuan_for_point else 0.0,
                "path": path,
                "file": file_name,
            })
            try:
                k = ("first_pt_up", int(system_index))
                if not hasattr(self, "_ui_diag_once"):
                    self._ui_diag_once = set()
                if k not in self._ui_diag_once:
                    self._ui_diag_once.add(k)
                    print(f"[UI][recv] 上表面 system={system_index} 收到首个缺陷点: x={x} y={y} img='{file_name}' dir='{path}'")
            except Exception:
                pass
            self.cm[list_index] = c_m
            self.pos[list_index] = pos
            self.base_folder[list_index] = path
            self.fukuan_mm[list_index] = fukuan_for_point
            # 更新“最近一次出现”的长度坐标，用于滑动窗口防空白锚定
            y_val = float(y)
            if y_val > self.latest_defect_y[list_index]:
                self.latest_defect_y[list_index] = y_val
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
            file_name = self.find_file_with_coordinates(path, int(x), int(y)) or ""
            self.defect_points_down[list_index].append({
                "x": float(x),
                "y": float(y),
                "fukuan": float(fukuan_for_point) if fukuan_for_point else 0.0,
                "path": path,
                "file": file_name,
            })
            try:
                k = ("first_pt_down", int(system_index))
                if not hasattr(self, "_ui_diag_once"):
                    self._ui_diag_once = set()
                if k not in self._ui_diag_once:
                    self._ui_diag_once.add(k)
                    print(f"[UI][recv] 下表面 system={system_index} 收到首个缺陷点: x={x} y={y} img='{file_name}' dir='{path}'")
            except Exception:
                pass
            self.cm2[list_index] = c_m
            self.pos2[list_index] = pos
            self.base_folder2[list_index] = path
            self.fukuan_mm2[list_index] = fukuan_for_point
            # 更新“最近一次出现”的长度坐标，用于滑动窗口防空白锚定
            y_val = float(y)
            if y_val > self.latest_defect_y[list_index]:
                self.latest_defect_y[list_index] = y_val
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

            ui = getattr(self, "_ui_defect_cfg", None) or MainWindow._read_ui_defect_display_config()

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
            # 横轴主轴线（外侧）；刻度位置与缺陷红点同一套 mm->px
            painter_b.drawLine(0, 0, x_max_px, 0)
            for i in range(11):
                length_mm = window_start_mm + (i / 10.0) * window_len_mm
                x_rel = int(
                    round(
                        _defect_length_mm_to_px(
                            length_mm,
                            window_start_mm,
                            window_end_mm,
                            x_max_px,
                            ui,
                        )
                    )
                )
                x_rel = max(0, min(x_max_px, x_rel))
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

            ui = getattr(self, "_ui_defect_cfg", None) or MainWindow._read_ui_defect_display_config()
            max_backtrack_mm = float(ui["backtrack_max_mm"])

            points = self.defect_points_up[system_index]
            # 仅保留窗口附近点，控制内存与渲染量
            # 防止锚定后窗口长度变得很大，回溯距离过远导致按钮数量暴增
            backtrack_mm = min(window_len * 1.5, max_backtrack_mm)
            keep_from = window_start - backtrack_mm
            self.defect_points_up[system_index] = [p for p in points if p["y"] >= keep_from]
            points = self.defect_points_up[system_index]

            visible = [p for p in points if window_start <= p["y"] <= window_end]
            if len(visible) > 160:
                visible = visible[-160:]

            latest_visible = None
            for p in visible:
                button_x_rel = int(
                    round(
                        _defect_length_mm_to_px(
                            float(p["y"]),
                            float(window_start),
                            float(window_end),
                            x_max_px,
                            ui,
                        )
                    )
                )
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

            ui = getattr(self, "_ui_defect_cfg", None) or MainWindow._read_ui_defect_display_config()
            max_backtrack_mm = float(ui["backtrack_max_mm"])

            points = self.defect_points_down[system_index]
            # 防止锚定后窗口长度变得很大，回溯距离过远导致按钮数量暴增
            backtrack_mm = min(window_len * 1.5, max_backtrack_mm)
            keep_from = window_start - backtrack_mm
            self.defect_points_down[system_index] = [p for p in points if p["y"] >= keep_from]
            points = self.defect_points_down[system_index]

            visible = [p for p in points if window_start <= p["y"] <= window_end]
            if len(visible) > 160:
                visible = visible[-160:]

            latest_visible = None
            for p in visible:
                button_x_rel = int(
                    round(
                        _defect_length_mm_to_px(
                            float(p["y"]),
                            float(window_start),
                            float(window_end),
                            x_max_px,
                            ui,
                        )
                    )
                )
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
        try:
            if not directory or not os.path.isdir(directory):
                return None
            try:
                mtime = os.path.getmtime(directory)
            except Exception:
                mtime = 0.0
            cache = getattr(self, "_defect_dir_listing_cache", None)
            if cache is None:
                self._defect_dir_listing_cache = {}
                cache = self._defect_dir_listing_cache
            ent = cache.get(directory)
            if ent is None or ent[0] != mtime:
                try:
                    name_set = set(os.listdir(directory))
                except Exception:
                    return None
                cache[directory] = (mtime, name_set)
            else:
                name_set = ent[1]
            needle = f"{int(x)}_{int(y)}"
            for file_name in name_set:
                if needle in file_name:
                    return file_name
            return None
        except Exception as e:
            print(f"寻找细节图文件名过程中出现未知错误: {str(e)}")
            return None


    def display_image_up(self, folder_path, file_name,  system_index):
        try:
            if not folder_path or not file_name:
                return
            full_path = os.path.normpath(os.path.join(folder_path, file_name))
            if 0 <= int(system_index) < len(self._preview_image_path_up):
                self._preview_image_path_up[int(system_index)] = full_path
            realtime_labels = self._up_realtime_labels()
            label = realtime_labels[system_index]
            # 显示选中的图片
            pixmap = QPixmap(full_path)

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
            if not folder_path or not file_name:
                return
            full_path = os.path.normpath(os.path.join(folder_path, file_name))
            if 0 <= int(system_index) < len(self._preview_image_path_down):
                self._preview_image_path_down[int(system_index)] = full_path
            realtime_labels = self._down_realtime_labels()
            label = realtime_labels[system_index]
            # 显示选中的图片
            pixmap = QPixmap(full_path)

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
            if not folder_path or not file_name:
                return
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
            full_path = os.path.normpath(os.path.join(folder_path, file_name))
            print(f"点击加载路径: {full_path}")
            pixmap = QPixmap(full_path)
            if pixmap.isNull():
                print(f"❌ QPixmap加载失败")
                return
            if 0 <= int(system_index) < len(self._preview_image_path_up):
                self._preview_image_path_up[int(system_index)] = full_path
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

        except Exception as e:
            print(f"点击显示系统{system_index + 1}上表面图像错误: {str(e)}")

    def display_image2_down(self, folder_path, file_name, system_index):
        try:
            if not folder_path or not file_name:
                return
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
            full_path = os.path.normpath(os.path.join(folder_path, file_name))
            print(f"点击加载路径: {full_path}")
            pixmap = QPixmap(full_path)
            if pixmap.isNull():
                print(f"❌ QPixmap加载失败")
                return
            if 0 <= int(system_index) < len(self._preview_image_path_down):
                self._preview_image_path_down[int(system_index)] = full_path
            click_labels = self._down_click_labels()
            label = click_labels[system_index]

            # 获取label的尺寸
            label_width = label.width()
            label_height = label.height()

            # 将图片拉伸至label的尺寸，忽略长宽比
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.IgnoreAspectRatio)
            label.setPixmap(scaled_pixmap)
            label.repaint()
        except Exception as e:
            print(f"点击显示系统{system_index + 1}下表面图像错误: {str(e)}")

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    _apply_app_theme(app)
    window = MainWindow()
    window.show()
    # PyQt5: exec_(); PySide6: exec()
    _exec = getattr(app, "exec_", None) or getattr(app, "exec")
    sys.exit(_exec())