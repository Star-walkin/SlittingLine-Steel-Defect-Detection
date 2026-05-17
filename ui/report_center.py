from __future__ import annotations

import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from PyQt5.QtWidgets import QInputDialog, QLineEdit
from report_change import ReportWindow
from cls_config import ClsConfigWindow
from cls_model_registry import compat_and_remap as _cls_compat_and_remap, rptcfg_class_names as _rptcfg_class_names


_PROJECT_ROOT = os.path.join(_REPO_ROOT)
_AUTH_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "auth.yaml")
_DETECT_ROOT = os.path.join(_PROJECT_ROOT, "detect result")

import sys

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import strip_result_paths as _strip_paths


def _read_auth_password(role: str) -> str:
    try:
        with open(_AUTH_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("passwords", {}).get(role, "000"))
    except Exception:
        return "000"
_RPTCFG_PATH = os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml")
_CONFIG0_PATH = os.path.join(_PROJECT_ROOT, "config", "config0.yaml")
_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)
_GEN_REPORT_MODULE = "app.report.gen_report_cls"


def _is_date_folder(name: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", name))


def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except Exception:
        return []


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _pick_cam_dir_for_strip_discovery(roll: str, legacy_first_cam: Optional[str]) -> Optional[str]:
    """
    在卷根下选择用于 discover_strip_dir_basenames_under_cam 的一级目录名。
    新数据为 上表面/下表面；旧数据可能为 config 中的数字相机目录。
    """
    roll = str(roll or "").strip()
    if not roll or not os.path.isdir(roll):
        return None
    for name in ("上表面", "下表面"):
        cam_dir = os.path.join(roll, name)
        if not os.path.isdir(cam_dir):
            continue
        bases = _strip_paths.discover_strip_dir_basenames_under_cam(cam_dir)
        if bases:
            return name
    leg = str(legacy_first_cam or "").strip()
    if leg:
        cam_dir = os.path.join(roll, leg)
        if os.path.isdir(cam_dir):
            bases = _strip_paths.discover_strip_dir_basenames_under_cam(cam_dir)
            if bases:
                return leg
    return None


def _open_path(path: str) -> None:
    # Windows: use os.startfile
    os.startfile(path)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ResultSelection:
    date: str
    conduct_id: str
    strip_id_1based: int
    strip_dir_basename: Optional[str] = None

    @property
    def result_all_path(self) -> str:
        return os.path.join(_DETECT_ROOT, self.date, self.conduct_id)

    @property
    def strip_report_basename(self) -> str:
        # 新数据：与检测结果目录同名；旧数据：回退 strip_N
        if self.strip_dir_basename:
            return str(self.strip_dir_basename)
        return f"strip_{self.strip_id_1based}"

    @property
    def report_strip_dir(self) -> str:
        return os.path.join(self.result_all_path, "report", self.strip_report_basename)


class ReportCenterWindow(QtWidgets.QMainWindow):
    """
    报告中心：选择现有检测结果（日期/质保书号/带钢）并生成/查看/修改报告。
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("报告中心")
        self.resize(980, 640)

        self._process: Optional[QProcess] = None
        self._report_window: Optional[ReportWindow] = None
        self._progress_timer = QtCore.QTimer(self)
        self._progress_timer.setInterval(40)  # ~25 FPS
        self._progress_timer.timeout.connect(self._tick_progress)
        self._progress_current = 0.0
        self._progress_target = 0.0

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- selection row
        row = QHBoxLayout()
        root.addLayout(row)

        row.addWidget(QLabel("日期"))
        self.comboDate = QComboBox()
        self.comboDate.setMinimumWidth(140)
        row.addWidget(self.comboDate)

        row.addWidget(QLabel("质保书号"))
        self.comboConductId = QComboBox()
        self.comboConductId.setMinimumWidth(220)
        row.addWidget(self.comboConductId)

        row.addWidget(QLabel("带钢卡号"))
        self.comboStripId = QComboBox()
        self.comboStripId.setMinimumWidth(160)
        row.addWidget(self.comboStripId)

        self.btnRefresh = QPushButton("刷新")
        row.addWidget(self.btnRefresh)
        self.btnTypeConfig = QPushButton("类型配置")
        self.btnTypeConfig.setToolTip(
            "打开“类别配置”窗口（需密码，见 config/auth.yaml 的 cls_config）。"
            "维护型号、允收矩阵、并为每个型号选择分类模型。"
        )
        row.addWidget(self.btnTypeConfig)
        row.addStretch(1)

        # ---- actions row
        row2 = QHBoxLayout()
        root.addLayout(row2)

        self.btnGenerate = QPushButton("生成报告")
        self.btnView = QPushButton("查看报告")
        self.btnModify = QPushButton("修改报告")
        self.btnOpenFolder = QPushButton("打开目录")

        row2.addWidget(self.btnGenerate)
        row2.addWidget(self.btnView)
        row2.addWidget(self.btnModify)
        row2.addWidget(self.btnOpenFolder)
        # 分类模型状态：生成报告前必须合格
        self.lblClsStatus = QLabel("")
        self.lblClsStatus.setStyleSheet("color:#555;")
        row2.addWidget(self.lblClsStatus)
        row2.addStretch(1)

        # ---- progress + log
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        root.addWidget(self.progressBar)

        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)
        font = QtGui.QFont("Consolas", 9)
        self.txtLog.setFont(font)
        root.addWidget(self.txtLog, 1)

        # ---- wiring
        self.btnRefresh.clicked.connect(self.refresh_lists)
        self.btnTypeConfig.clicked.connect(self.open_type_config)
        self.comboDate.currentIndexChanged.connect(self._on_date_changed)
        self.comboConductId.currentIndexChanged.connect(self._on_conduct_changed)

        self.btnGenerate.clicked.connect(self.generate_report)
        self.btnView.clicked.connect(self.view_report)
        self.btnOpenFolder.clicked.connect(self.open_folder)
        self.btnModify.clicked.connect(self.open_modify)

        self.refresh_lists()
        self._refresh_cls_status()

    # -------------------- progress smoothing --------------------
    def _set_progress_target(self, v: float) -> None:
        v = float(max(0.0, min(100.0, v)))
        if v > self._progress_target:
            self._progress_target = v
        if not self._progress_timer.isActive():
            self._progress_timer.start()

    def _tick_progress(self) -> None:
        cur = float(self._progress_current)
        tgt = float(self._progress_target)
        if cur >= tgt - 1e-6:
            self._progress_current = tgt
            self.progressBar.setValue(int(round(tgt)))
            if tgt >= 100.0:
                self._progress_timer.stop()
            return

        # 缓动：越接近目标越慢，但每帧至少+0.3
        gap = tgt - cur
        step = max(0.3, min(3.5, gap * 0.18))
        cur2 = min(tgt, cur + step)
        self._progress_current = cur2
        self.progressBar.setValue(int(round(cur2)))

    # -------------------- scanning --------------------
    def refresh_lists(self) -> None:
        self.comboDate.blockSignals(True)
        self.comboConductId.blockSignals(True)
        self.comboStripId.blockSignals(True)
        try:
            self.comboDate.clear()
            self.comboConductId.clear()
            self.comboStripId.clear()

            dates = [d for d in _safe_listdir(_DETECT_ROOT) if _is_date_folder(d)]
            dates.sort(reverse=True)
            self.comboDate.addItems(dates)

            self._on_date_changed()
        finally:
            self.comboDate.blockSignals(False)
            self.comboConductId.blockSignals(False)
            self.comboStripId.blockSignals(False)
        self._refresh_cls_status()

    def _on_date_changed(self) -> None:
        date = self.comboDate.currentText().strip()
        self.comboConductId.clear()
        self.comboStripId.clear()
        if not date:
            return
        date_dir = os.path.join(_DETECT_ROOT, date)
        ids = [d for d in _safe_listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
        ids.sort()
        self.comboConductId.addItems(ids)
        self._on_conduct_changed()

    def _infer_strip_entries(self, sel: ResultSelection) -> List[Tuple[int, str]]:
        """
        返回 (strip_id_1based, strip_dir_basename) 列表。
        - strip_id_1based 仍保持 1..N 的编号（用于生成脚本参数与 rptcfg.strip_id）
        - strip_dir_basename 为 detect result 内实际目录名（可能为卡号目录）
        """
        first_cam = None
        try:
            with open(os.path.join(_PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cam_up_list = cfg.get("camrea_id_up", [])
            first_cam = str(cam_up_list[0]) if cam_up_list else None
        except Exception:
            first_cam = None

        roll = sel.result_all_path
        cfg0_roll = _strip_paths.read_result_roll_config0(roll)
        n = int(cfg0_roll.get("strip_count", 0) or 0) if isinstance(cfg0_roll, dict) else 0
        if n <= 0:
            try:
                with open(_CONFIG0_PATH, "r", encoding="utf-8") as f:
                    cfg0_live = yaml.safe_load(f) or {}
                n = int(cfg0_live.get("strip_count", 3) or 3)
            except Exception:
                n = 3
        n = max(1, min(8, int(n)))

        names = _strip_paths.strip_dir_list_from_roll(roll, int(n))
        if isinstance(names, list) and len(names) == int(n) and all(str(x).strip() for x in names):
            return [(i + 1, str(names[i])) for i in range(int(n))]

        picked = _pick_cam_dir_for_strip_discovery(roll, first_cam)
        if picked:
            cam_dir = os.path.join(roll, picked)
            bases = _strip_paths.discover_strip_dir_basenames_under_cam(cam_dir)
            if bases:
                return [(i + 1, str(bases[i])) for i in range(len(bases))]

        # 最后兜底：只有编号（目录可能尚未生成）
        return [(i, f"strip_{i}") for i in range(1, int(n) + 1)]

    def _on_conduct_changed(self) -> None:
        date = self.comboDate.currentText().strip()
        cid = self.comboConductId.currentText().strip()
        self.comboStripId.clear()
        if not date or not cid:
            return
        # initial selection: strip 1; later refreshed by inference
        tmp_sel = ResultSelection(date=date, conduct_id=cid, strip_id_1based=1)
        strip_entries = self._infer_strip_entries(tmp_sel)
        # 优先读取该卷根目录 config0_snapshot.yaml；其次读 report/<条带目录>/config0_snapshot.yaml；最后回退编号
        strip_card_map = {}
        try:
            root_snap = os.path.join(tmp_sel.result_all_path, "config0_snapshot.yaml")
            cfg0_root = None
            if os.path.exists(root_snap):
                with open(root_snap, "r", encoding="utf-8") as f:
                    cfg0_root = yaml.safe_load(f) or {}

            for sid, folder in strip_entries:
                val = ""
                if isinstance(cfg0_root, dict):
                    cards = cfg0_root.get("strip_card_list") or []
                    if isinstance(cards, (list, tuple)) and len(cards) >= sid:
                        val = str(cards[sid - 1] or "").strip()
                    if not val:
                        val = str(cfg0_root.get(f"strip_card_{sid}", "") or "").strip()

                if not val:
                    snap = os.path.join(tmp_sel.result_all_path, "report", str(folder), "config0_snapshot.yaml")
                    if os.path.exists(snap):
                        with open(snap, "r", encoding="utf-8") as f:
                            cfg0 = yaml.safe_load(f) or {}
                        cards = cfg0.get("strip_card_list") or []
                        if isinstance(cards, (list, tuple)) and len(cards) >= sid:
                            val = str(cards[sid - 1] or "").strip()
                        else:
                            val = str(cfg0.get(f"strip_card_{sid}", "") or "").strip()

                if val:
                    strip_card_map[int(sid)] = val
        except Exception:
            strip_card_map = {}

        for sid, folder in strip_entries:
            card = strip_card_map.get(int(sid))
            # 甲方口径：优先直接显示工人输入的“带钢卡号”；内部 data 仍保存 strip_id 数字
            label = str(card) if card else str(folder)
            self.comboStripId.addItem(label, int(sid))
            try:
                self.comboStripId.setItemData(self.comboStripId.count() - 1, str(folder), QtCore.Qt.UserRole + 1)
            except Exception:
                pass
            try:
                tip = f"带钢序号：{sid}；目录：{folder}" + (f"；卡号：{card}" if card else "")
                self.comboStripId.setItemData(self.comboStripId.count() - 1, tip, QtCore.Qt.ToolTipRole)
            except Exception:
                pass

    # -------------------- selection helpers --------------------
    def _get_selection(self) -> Optional[ResultSelection]:
        date = self.comboDate.currentText().strip()
        cid = self.comboConductId.currentText().strip()
        if self.comboStripId.currentIndex() < 0:
            strip_s = ""
            strip_id = None
        else:
            strip_id = self.comboStripId.currentData()
            strip_s = self.comboStripId.currentText().strip()
        if not date or not cid or not strip_s or strip_id is None:
            QMessageBox.information(self, "选择不完整", "请先选择日期、质保书号与带钢卡号。")
            return None
        try:
            strip_id = int(strip_id)
            if strip_id < 1:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "带钢号错误", f"带钢号必须为>=1整数，当前：{strip_s}")
            return None

        folder = None
        try:
            folder = self.comboStripId.currentData(QtCore.Qt.UserRole + 1)
        except Exception:
            folder = None
        folder_s = str(folder).strip() if folder is not None else ""
        if not folder_s:
            folder_s = _strip_paths.resolve_strip_dir_basename(os.path.join(_DETECT_ROOT, date, cid), int(strip_id))

        sel = ResultSelection(date=date, conduct_id=cid, strip_id_1based=strip_id, strip_dir_basename=folder_s)
        if not os.path.isdir(sel.result_all_path):
            QMessageBox.warning(self, "目录不存在", f"检测结果目录不存在：\n{sel.result_all_path}")
            return None
        return sel

    # -------------------- actions --------------------
    def generate_report(self) -> None:
        sel = self._get_selection()
        if not sel:
            return

        # 生成报告（update-info=false）会触发分类推理：必须保证分类模型“已配置+有classes.json+与当前标准兼容”
        if not self._check_cls_model_ready(force_open=True):
            return

        # 生成前轻量自检：提示最常见的“类别名/颜色表不一致”风险（不阻断生成）
        try:
            with open(_RPTCFG_PATH, "r", encoding="utf-8") as f:
                rptcfg = yaml.safe_load(f) or {}
            cls_labels = rptcfg.get("class_labels") or {}
            colors = rptcfg.get("colors") or {}
            missing = []
            for _k, _name in cls_labels.items():
                if _name and _name not in colors:
                    missing.append(str(_name))
            if missing:
                self._append_log(
                    "[WARN] colors 缺少以下类别颜色，生成时将自动回退默认色："
                    + "、".join(missing[:10])
                    + ("\n" if len(missing) <= 10 else f"...（共{len(missing)}项）\n")
                )
        except Exception:
            pass

        self._ensure_process_stopped()
        self.txtLog.clear()
        self._progress_current = 0.0
        self._progress_target = 0.0
        self.progressBar.setValue(0)
        self._set_progress_target(3)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_proc_output)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)

        args = [
            "-u",
            "-m",
            _GEN_REPORT_MODULE,
            "--date",
            sel.date,
            "--conduct-id",
            sel.conduct_id,
            "--strip-id",
            str(sel.strip_id_1based),
            "--update-info",
            "false",
        ]
        proc.start(_PYTHON_EXE, args)
        self._process = proc
        self._set_progress_target(6)

    def open_type_config(self) -> None:
        password, ok = QInputDialog.getText(
            self,
            "密码验证",
            "进入类别配置需要权限验证，请输入密码:",
            QLineEdit.Password,
        )
        if not ok:
            return
        if password != _read_auth_password("cls_config"):
            QMessageBox.warning(self, "密码错误", "请输入正确的密码！")
            return
        if not hasattr(self, "_type_cfg_win") or self._type_cfg_win is None:
            self._type_cfg_win = ClsConfigWindow(self)
        self._type_cfg_win.show()
        self._type_cfg_win.raise_()
        self._type_cfg_win.activateWindow()

    def _refresh_cls_status(self) -> None:
        ok, msg = self._check_cls_model_ready(force_open=False, silent=True, return_message=True)
        if ok:
            self.lblClsStatus.setStyleSheet("color:#2e7d32; font-weight:bold;")
            self.lblClsStatus.setText(f"分类模型：已就绪（{msg}）")
        else:
            self.lblClsStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblClsStatus.setText(f"分类模型：未就绪（{msg}）")

    def _check_cls_model_ready(
        self,
        *,
        force_open: bool,
        silent: bool = False,
        return_message: bool = False,
    ):
        # 读取 cls_model_path
        try:
            with open(os.path.join(_PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        model_path = str((cfg or {}).get("cls_model_path", "") or "").strip()
        if not model_path:
            if not silent:
                QMessageBox.warning(self, "分类模型未配置", "当前未配置分类模型（cls_model_path 为空）。")
            if force_open:
                self._force_open_model_picker()
            return (False, "未配置") if return_message else False
        if not os.path.exists(model_path):
            if not silent:
                QMessageBox.warning(self, "分类模型不存在", f"分类模型文件不存在：\n{model_path}")
            if force_open:
                self._force_open_model_picker()
            return (False, "文件不存在") if return_message else False

        # classes.json 必须存在（否则无法确认类别对应关系）
        cj = os.path.join(os.path.dirname(model_path), "classes.json")
        model_classes = []
        if os.path.exists(cj):
            try:
                import json

                with open(cj, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                if isinstance(obj, dict) and "id_to_name" in obj and isinstance(obj["id_to_name"], dict):
                    obj = obj["id_to_name"]
                if isinstance(obj, dict):
                    for kk in sorted(obj.keys(), key=lambda x: int(str(x))):
                        model_classes.append(str(obj.get(kk, "")).strip())
            except Exception:
                model_classes = []

        if not model_classes:
            if not silent:
                QMessageBox.warning(
                    self,
                    "分类模型不完整",
                    "当前模型缺少类别清单（classes.json）。为避免分类错位，禁止生成报告。\n\n"
                    "请进入「报告打印与标准维护」选择可用模型，或通过向导重新训练并启用。",
                )
            if force_open:
                self._force_open_model_picker()
            return (False, "缺少classes.json") if return_message else False

        rpt_names = _rptcfg_class_names(_RPTCFG_PATH)
        ok, remap, diff = _cls_compat_and_remap(model_classes=model_classes, rptcfg_classes=rpt_names)
        if not ok:
            if not silent:
                QMessageBox.warning(
                    self,
                    "分类模型与标准不一致",
                    "当前分类模型的类别与缺陷类别标准不一致，禁止生成报告。\n\n"
                    f"缺少：{', '.join(diff.get('missing') or []) or '-'}\n"
                    f"多出：{', '.join(diff.get('extra') or []) or '-'}\n\n"
                    "请进入「报告打印与标准维护」选择类别一致的模型，或重新训练。",
                )
            if force_open:
                self._force_open_model_picker()
            return (False, "类别不一致") if return_message else False

        # ok
        same_order = remap == list(range(1, len(remap) + 1))
        return (True, "顺序一致" if same_order else "已自动对齐") if return_message else True

    def _force_open_model_picker(self) -> None:
        # 按你的要求：模型选择只在“类别配置”区域可用
        self.open_type_config()

    def view_report(self) -> None:
        sel = self._get_selection()
        if not sel:
            return

        report_dir = sel.report_strip_dir
        if not os.path.isdir(report_dir):
            QMessageBox.information(self, "未生成报告", f"该钢带尚未生成报告：\n{report_dir}")
            return

        files = _safe_listdir(report_dir)
        pdfs = [os.path.join(report_dir, f) for f in files if f.lower().endswith(".pdf")]
        pngs = [os.path.join(report_dir, f) for f in files if f.lower().endswith(".png")]

        target = _first_existing(
            pdfs
            + [p for p in pngs if "表面检测报告" in os.path.basename(p)]
            + [p for p in pngs if "anomaly_distribution" in os.path.basename(p)]
            + pngs
        )
        if not target:
            QMessageBox.information(self, "暂无产物", f"目录内未找到 PDF/PNG：\n{report_dir}")
            return

        try:
            _open_path(target)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"无法打开：\n{target}\n\n{e}")

    def open_folder(self) -> None:
        sel = self._get_selection()
        if not sel:
            return
        path = sel.report_strip_dir if os.path.isdir(sel.report_strip_dir) else sel.result_all_path
        try:
            _open_path(path)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"无法打开目录：\n{path}\n\n{e}")

    def open_modify(self) -> None:
        sel = self._get_selection()
        if not sel:
            return

        # 修改报告属于高权限操作，与「报告打印」使用同一密码
        password, ok = QInputDialog.getText(
            self, "密码验证",
            "修改报告内容需要权限验证，请输入密码:",
            QLineEdit.Password,
        )
        if not ok:
            return
        try:
            with open(_AUTH_CONFIG_PATH, "r", encoding="utf-8") as f:
                auth_cfg = yaml.safe_load(f) or {}
            expected = str(auth_cfg.get("passwords", {}).get("standard_report", "000"))
        except Exception:
            expected = "000"
        if password != expected:
            QMessageBox.warning(self, "密码错误", "请输入正确的密码！")
            return

        # 写入 rptcfg.yaml 绑定当前选择
        try:
            try:
                with open(_RPTCFG_PATH, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except FileNotFoundError:
                data = {}
            data["time"] = sel.date
            data["id"] = sel.conduct_id
            data["strip_id"] = str(sel.strip_id_1based)
            data["update_info"] = True
            with open(_RPTCFG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True)
        except Exception as e:
            QMessageBox.warning(self, "写入失败", f"写入 rptcfg.yaml 失败：\n{e}")
            return

        if self._report_window is None:
            self._report_window = ReportWindow()
        # 每次从报告中心进入修改界面，都强制同步一次左上角信息与标准视图
        try:
            self._report_window.refresh_from_report_center()
        except Exception:
            pass
        self._report_window.show()
        self._report_window.raise_()
        self._report_window.activateWindow()

    # -------------------- process callbacks --------------------
    def _append_log(self, text: str) -> None:
        self.txtLog.moveCursor(QtGui.QTextCursor.End)
        self.txtLog.insertPlainText(text)
        self.txtLog.moveCursor(QtGui.QTextCursor.End)

    def _update_progress_by_text(self, text: str) -> None:
        # 阶段式：稳健且不依赖精确百分比（仅提升 target，实际显示由定时器缓动）
        v = self._progress_target
        if "上表面缺陷分类已完成" in text:
            v = max(v, 30)
        if "下表面缺陷分类已完成" in text:
            v = max(v, 50)
        if "正在生成 strip_" in text or "正在生成 strip" in text:
            v = max(v, 60)
        if "多带钢汇总报告已生成" in text or "报告生成完成" in text:
            v = max(v, 95)
        self._set_progress_target(v)

    def _on_proc_output(self) -> None:
        if not self._process:
            return
        b = bytes(self._process.readAllStandardOutput())
        if not b:
            return
        try:
            text = b.decode("utf-8")
        except UnicodeDecodeError:
            text = b.decode("gbk", errors="replace")
        self._append_log(text)
        self._update_progress_by_text(text)

    def _on_proc_finished(self, exit_code: int, _exit_status: QtCore.QProcess.ExitStatus) -> None:
        if exit_code == 0:
            self._set_progress_target(100)
        else:
            self._set_progress_target(max(self._progress_target, 90))
            self._append_log(f"\n[ERROR] 生成脚本退出码：{exit_code}\n")
        self._process = None

    def _on_proc_error(self, err: QtCore.QProcess.ProcessError) -> None:
        self._append_log(f"\n[ERROR] QProcess 错误：{err}\n")

    def _ensure_process_stopped(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            try:
                self._process.kill()
                self._process.waitForFinished(2000)
            except Exception:
                pass
        self._process = None
