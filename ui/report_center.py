from __future__ import annotations

import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import re
from dataclasses import dataclass
from typing import List, Optional

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
import sys


_PROJECT_ROOT = os.path.join(_REPO_ROOT)
_AUTH_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "auth.yaml")
_DETECT_ROOT = os.path.join(_PROJECT_ROOT, "detect result")
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


def _open_path(path: str) -> None:
    # Windows: use os.startfile
    os.startfile(path)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ResultSelection:
    date: str
    conduct_id: str
    strip_id_1based: int

    @property
    def result_all_path(self) -> str:
        return os.path.join(_DETECT_ROOT, self.date, self.conduct_id)

    @property
    def report_strip_dir(self) -> str:
        return os.path.join(self.result_all_path, "report", f"strip_{self.strip_id_1based}")


class ReportCenterWindow(QtWidgets.QMainWindow):
    """
    报告中心：选择现有检测结果（日期/卡号/带钢）并生成/查看/修改报告。
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

        row.addWidget(QLabel("卡号"))
        self.comboConductId = QComboBox()
        self.comboConductId.setMinimumWidth(220)
        row.addWidget(self.comboConductId)

        row.addWidget(QLabel("钢带号"))
        self.comboStripId = QComboBox()
        self.comboStripId.setMinimumWidth(100)
        row.addWidget(self.comboStripId)

        self.btnRefresh = QPushButton("刷新")
        row.addWidget(self.btnRefresh)
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
        self.comboDate.currentIndexChanged.connect(self._on_date_changed)
        self.comboConductId.currentIndexChanged.connect(self._on_conduct_changed)

        self.btnGenerate.clicked.connect(self.generate_report)
        self.btnView.clicked.connect(self.view_report)
        self.btnOpenFolder.clicked.connect(self.open_folder)
        self.btnModify.clicked.connect(self.open_modify)

        self.refresh_lists()

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

    def _infer_strip_ids(self, sel: ResultSelection) -> List[int]:
        # Prefer scanning strip_ dirs under first up camera dir
        try:
            with open(os.path.join(_PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cam_up_list = cfg.get("camrea_id_up", [])
            first_cam = str(cam_up_list[0]) if cam_up_list else None
        except Exception:
            first_cam = None

        if first_cam:
            cam_dir = os.path.join(sel.result_all_path, first_cam)
            strips = []
            for d in _safe_listdir(cam_dir):
                if d.startswith("strip_") and os.path.isdir(os.path.join(cam_dir, d)):
                    try:
                        strips.append(int(d.split("_", 1)[1]))
                    except Exception:
                        continue
            strips = sorted(set(strips))
            if strips:
                return strips

        # Fallback to config0.strip_count
        try:
            with open(_CONFIG0_PATH, "r", encoding="utf-8") as f:
                cfg0 = yaml.safe_load(f) or {}
            n = int(cfg0.get("strip_count", 3))
            n = max(1, min(8, n))
            return list(range(1, n + 1))
        except Exception:
            return [1, 2, 3]

    def _on_conduct_changed(self) -> None:
        date = self.comboDate.currentText().strip()
        cid = self.comboConductId.currentText().strip()
        self.comboStripId.clear()
        if not date or not cid:
            return
        # initial selection: strip 1; later refreshed by inference
        tmp_sel = ResultSelection(date=date, conduct_id=cid, strip_id_1based=1)
        strip_ids = self._infer_strip_ids(tmp_sel)
        self.comboStripId.addItems([str(i) for i in strip_ids])

    # -------------------- selection helpers --------------------
    def _get_selection(self) -> Optional[ResultSelection]:
        date = self.comboDate.currentText().strip()
        cid = self.comboConductId.currentText().strip()
        strip_s = self.comboStripId.currentText().strip()
        if not date or not cid or not strip_s:
            QMessageBox.information(self, "选择不完整", "请先选择日期、卡号与钢带号。")
            return None
        try:
            strip_id = int(strip_s)
            if strip_id < 1:
                raise ValueError
        except Exception:
            QMessageBox.warning(self, "钢带号错误", f"钢带号必须为>=1整数，当前：{strip_s}")
            return None

        sel = ResultSelection(date=date, conduct_id=cid, strip_id_1based=strip_id)
        if not os.path.isdir(sel.result_all_path):
            QMessageBox.warning(self, "目录不存在", f"检测结果目录不存在：\n{sel.result_all_path}")
            return None
        return sel

    # -------------------- actions --------------------
    def generate_report(self) -> None:
        sel = self._get_selection()
        if not sel:
            return

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
