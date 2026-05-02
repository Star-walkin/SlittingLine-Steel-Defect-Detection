from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QMessageBox, QInputDialog
from PyQt5.QtWidgets import (
    QTableWidget,
    QHBoxLayout,
    QApplication,
    QWidget,
    QLineEdit,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QRect, QFileSystemWatcher
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QEvent, QTimer
import json
from datetime import datetime
from report import Ui_Report  # 引用生成的 ui_para.py 文件
from cls_config import ClsConfigWindow
import ast
import yaml
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import subprocess
import socket
import time
from PyQt5.QtGui import QPainter, QPen, QFont

from rptcfg_store import (
    read_yaml as _rpt_read,
    read_meta as _rpt_read_meta,
    update_keys as _rpt_update,
    list_backups as _rpt_list_backups,
    rollback_to_backup as _rpt_rollback,
)
from cls_wizard import ClsWizardWindow
from cls_model_registry import (
    compat_and_remap as _cls_compat_and_remap,
    rptcfg_class_names as _rptcfg_class_names,
    scan_model_registry as _scan_model_registry,
    write_runtime_remap as _write_runtime_remap,
)

# ── 角标（表格左上角空白格）：对角线 + “缺陷/面积” ─────────────────────────────


class _DiagonalCornerWidget(QWidget):
    def __init__(self, table: QTableWidget, *, tr_text: str = "面积", bl_text: str = "缺陷"):
        super().__init__(table)
        self._table = table
        # 需求：右上=面积，左下=缺陷（对角线：左上→右下）
        self._tr = str(tr_text)
        self._bl = str(bl_text)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._sync_size()

    def _sync_size(self):
        try:
            w = max(10, int(self._table.verticalHeader().width()))
            h = max(10, int(self._table.horizontalHeader().height()))
            self.setFixedSize(w, h)
        except Exception:
            pass

    def paintEvent(self, _event):
        self._sync_size()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # background (match table viewport)
        p.fillRect(self.rect(), self._table.palette().base())

        # diagonal
        pen = QPen(self._table.palette().mid().color())
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(0, 0, self.width() - 1, self.height() - 1)

        # texts
        font = QFont(self._table.font())
        font.setPointSize(max(8, int(font.pointSize() * 0.9)))
        p.setFont(font)
        p.setPen(self._table.palette().text().color())
        pad = 4
        p.drawText(pad, pad, self.width() - pad, self.height() - pad, Qt.AlignRight | Qt.AlignTop, self._tr)
        p.drawText(pad, pad, self.width() - pad, self.height() - pad, Qt.AlignLeft | Qt.AlignBottom, self._bl)
        p.end()


def _apply_matrix_corner_labels(table: QTableWidget, *, tr_text: str = "面积", bl_text: str = "缺陷") -> None:
    """
    将 QTableWidget 左上角空白格替换成“对角线角标”：
    左上=缺陷，右下=面积（可按需改字）。
    """
    if table is None:
        return
    try:
        # 禁用 Qt 默认的“左上角按钮”，否则会遮住 corner widget（你现在看到的空白通常就是它）
        try:
            table.setCornerButtonEnabled(False)
        except Exception:
            pass
        # 方案A：cornerWidget（某些样式/平台可能不显示）
        try:
            w = _DiagonalCornerWidget(table, tr_text=tr_text, bl_text=bl_text)
            table.setCornerWidget(w)
            w.setAutoFillBackground(True)
            w.show()
            w.raise_()
        except Exception:
            w = None

        # 方案B：覆盖层 overlay（强制显示，绕过 Qt corner 区内部实现）
        try:
            ov = getattr(table, "_diag_corner_overlay", None)
            if ov is None:
                ov = _DiagonalCornerWidget(table, tr_text=tr_text, bl_text=bl_text)
                ov.setObjectName("_diag_corner_overlay")
                ov.setAutoFillBackground(True)
                ov.show()
                ov.raise_()
                setattr(table, "_diag_corner_overlay", ov)
            else:
                ov._tr = str(tr_text)  # type: ignore[attr-defined]
                ov._bl = str(bl_text)  # type: ignore[attr-defined]
        except Exception:
            ov = None

        def _sync_any():
            for ww in (w, ov):
                if ww is None:
                    continue
                try:
                    ww._sync_size()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    ww.move(0, 0)
                except Exception:
                    pass
                try:
                    ww.raise_()
                except Exception:
                    pass
                try:
                    ww.updateGeometry()
                except Exception:
                    pass
                ww.update()

        try:
            hh = table.horizontalHeader()
            vh = table.verticalHeader()
            hh.sectionResized.connect(lambda *_: _sync_any())
            vh.sectionResized.connect(lambda *_: _sync_any())
            hh.geometriesChanged.connect(_sync_any)
            vh.geometriesChanged.connect(_sync_any)
            table.viewport().installEventFilter(table)
        except Exception:
            pass
        QTimer.singleShot(0, _sync_any)
    except Exception:
        pass

# ── 路径常量（与 main.py 保持一致）──────────────────────────────────────────
_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)
_GEN_REPORT_MODULE = "app.report.gen_report_cls"
_MAKE_STD_SCRIPT   = os.path.join(_REPO_ROOT, "ui", "make_standard.py")
_DETECT_RESULT_DIR = os.path.join(_REPO_ROOT, "detect result")
_PROJECT_ROOT = os.path.join(_REPO_ROOT)
_DETAIL_OPT_JSON = os.path.join(_PROJECT_ROOT, "config", "detail_optimization.json")
_AUTH_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "auth.yaml")

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
# ────────────────────────────────────────────────────────────────────────────


def _load_json_safe(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else []


def _make_statistic_for_detail(config: dict, rptcfg: dict):
    """与 gen_report 一致的 Statistic_anomaly，用于只算面积区间表不计图。"""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    import pandas as pd
    from function_bank import Statistic_anomaly

    table_path = os.path.join(_PROJECT_ROOT, "table.json")
    standard_area_tables = pd.read_json(table_path, orient="split")
    cfg0_path = os.path.join(_PROJECT_ROOT, "config", "config0.yaml")
    try:
        with open(cfg0_path, "r", encoding="utf-8") as f:
            cfg0 = yaml.safe_load(f) or {}
    except Exception:
        cfg0 = {}
    fukuan0 = cfg0.get("fukuan_1") or cfg0.get("fukuan_2") or 50
    try:
        fukuan0 = float(fukuan0)
    except Exception:
        fukuan0 = 50.0
    return Statistic_anomaly(
        conduct_id=str(rptcfg.get("id", "")),
        fukuan=fukuan0,
        range=config["anomaly_area_cls_range"],
        result_path=".",
        start_time="",
        remove_threshold=config.get("remove_threshold", 0),
        steel_length_range=config.get("steel_length_range") or [],
        update_info=False,
        standard_area_tables=standard_area_tables,
        colors=rptcfg.get("colors", {}),
        class_labels=rptcfg.get("class_labels", {}),
        cls_all=rptcfg.get("class_list", []),
        area_range=rptcfg.get("area_range", []),
    )


def _normalize_cam_ids(cam_ids):
    """与 gen_detect_report 一致：支持 list、'2'、'2,3' 等写法。"""
    if cam_ids is None:
        return []
    if isinstance(cam_ids, (list, tuple)):
        return [str(x).strip() for x in cam_ids if str(x).strip()]
    s = str(cam_ids).strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _detail_json_matrix_sum(obj) -> int:
    try:
        t = 0
        for key in ("matrix_up", "matrix_down"):
            m = obj.get(key) if isinstance(obj, dict) else None
            if not isinstance(m, list):
                continue
            for row in m:
                if not isinstance(row, (list, tuple)):
                    continue
                for v in row:
                    try:
                        t += int(v)
                    except Exception:
                        pass
        return t
    except Exception:
        return 0


def _surface_area_table_from_files(
    result_root: str,
    strip_folder: str,
    cam_ids,
    stat,
    print_cls,
):
    """合并多相机 anomaly_info_result.json，按与 gen_report_cls 相同的扁平化 + filter_classes 再统计。"""
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from gen_report_cls import filter_classes
    from function_bank import area_val_matches_print_area_filter

    all_data = []
    for cam_id in _normalize_cam_ids(cam_ids):
        p = os.path.join(result_root, str(cam_id), strip_folder, "anomaly_info_result.json")
        raw = _load_json_safe(p, [])
        if not isinstance(raw, list):
            raw = []
        all_data.append(raw)
    # 与 gen_report_cls 一致：先按相机摊平一层，再交给 filter_classes 内再摊一层得到 str 条目
    merged = [item for sublist in all_data for item in sublist]
    if not merged:
        empty = {int(c): [] for c in (stat.cls_all or [])}
        return stat.create_area_table(empty)[0]
    clean_info = filter_classes(merged, class_list=list(stat.cls_all or []))
    select_info_area = {int(c): [] for c in (stat.cls_all or [])}
    for cls_raw in print_cls or stat.cls_all:
        try:
            cls = int(cls_raw)
        except Exception:
            continue
        if cls not in clean_info:
            continue
        for triple in clean_info[cls]:
            if len(triple) < 3:
                continue
            x, y, area = triple[0], triple[1], triple[2]
            try:
                y_val = round(float(y) / 1e6, 3)
                area_val = float(area)
            except Exception:
                continue
            len_ok = True
            sr = stat.steel_length_range
            if isinstance(sr, (list, tuple)) and len(sr) == 2:
                len_ok = float(sr[0]) <= y_val <= float(sr[1])
            area_ok = True
            area_ok = area_val_matches_print_area_filter(area_val, stat.area_range)
            if len_ok and area_ok:
                select_info_area[cls].append(area_val)
    return stat.create_area_table(select_info_area)[0]


def _apply_sparse_updates_to_df(area_table, updates):
    """把 rptcfg 的 updates_* 套到 DataFrame 上（显示用）。"""
    if area_table is None or not updates:
        return area_table
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from function_bank import resolve_area_table_column_key

    out = area_table.copy()
    for up in updates:
        try:
            lab = up.get("class_label")
            raw_iv = str(up.get("area_interval", ""))
            nv = int(up.get("new_count", 0))
        except Exception:
            continue
        if lab is None or not str(raw_iv).strip():
            continue
        try:
            col_key = resolve_area_table_column_key(raw_iv, out)
            if lab in out.index and col_key in out.columns:
                out.loc[lab, col_key] = nv
        except Exception:
            continue
    return out


def _df_to_cell_int(df, row_label: str, ui_header: str) -> int:
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from function_bank import resolve_area_table_column_key

    try:
        col = resolve_area_table_column_key(ui_header, df)
        v = df.loc[row_label, col]
        return int(v)
    except Exception:
        return 0



class CheckableComboBox(QComboBox):
    """多选下拉框：每个条目带复选框，点击切换选中状态；弹出框不随条目点击自动关闭。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        # 仅在“点击条目勾选”时阻止自动收起；点击空白处/其它控件时仍应正常关闭
        self._skip_hide = False
        le = self.lineEdit()
        le.setReadOnly(True)
        le.setPlaceholderText("请选择…")
        le.setStyleSheet("QLineEdit{background:white;padding:2px 8px;color:#333;}")
        self._mdl = QStandardItemModel(self)
        self.setModel(self._mdl)
        self._mdl.dataChanged.connect(self._refresh_text)
        self.view().viewport().installEventFilter(self)

    # ── public API ──────────────────────────────────────────────────────────
    def addItem(self, text, data=None):          # noqa: override
        item = QStandardItem(str(text))
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        if data is not None:
            item.setData(data, Qt.UserRole)
        self._mdl.appendRow(item)
        self._refresh_text()

    def selected_data(self):
        out = []
        for i in range(self._mdl.rowCount()):
            it = self._mdl.item(i)
            if it and it.checkState() == Qt.Checked:
                d = it.data(Qt.UserRole)
                out.append(d if d is not None else it.text())
        return out

    def set_checked_values(self, values):
        sv = {str(v) for v in (values or [])}
        for i in range(self._mdl.rowCount()):
            it = self._mdl.item(i)
            if it:
                d = it.data(Qt.UserRole)
                k = str(d if d is not None else it.text())
                it.setCheckState(Qt.Checked if k in sv else Qt.Unchecked)

    def check_all(self):
        for i in range(self._mdl.rowCount()):
            it = self._mdl.item(i)
            if it:
                it.setCheckState(Qt.Checked)

    # ── internals ───────────────────────────────────────────────────────────
    def _refresh_text(self):
        sel = [self._mdl.item(i).text()
               for i in range(self._mdl.rowCount())
               if self._mdl.item(i) and self._mdl.item(i).checkState() == Qt.Checked]
        self.lineEdit().setText(", ".join(sel) if sel else "")

    def hidePopup(self):
        # 默认允许关闭；但“点击条目勾选”那一下不关闭，便于连续多选
        if getattr(self, "_skip_hide", False):
            return
        super().hidePopup()

    def mousePressEvent(self, e):
        # 点击下拉框本体时手动切换弹出状态
        if self.view().isVisible():
            QComboBox.hidePopup(self)
        else:
            super().mousePressEvent(e)

    def eventFilter(self, obj, event):
        if obj == self.view().viewport() and event.type() == QEvent.MouseButtonRelease:
            idx = self.view().indexAt(event.pos())
            if idx.isValid():
                it = self._mdl.itemFromIndex(idx)
                if it:
                    new = (Qt.Unchecked if it.checkState() == Qt.Checked
                           else Qt.Checked)
                    # 阻止本次条目点击触发 popup 关闭；下一轮事件立刻恢复
                    self._skip_hide = True
                    QTimer.singleShot(0, lambda: setattr(self, "_skip_hide", False))
                    it.setCheckState(new)
            return True
        return super().eventFilter(obj, event)


class ReportWindow(QMainWindow, Ui_Report):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("报告打印与标准维护")
        self.class_counter = 1  # 用于自动分配 class_id
        self.input_fields = []
        self.config_data = {}
        self.updates_up = []
        self.updates_down = []

        self.config_data['update_info'] = True

        self._rptcfg_revision_at_load = 0
        self._has_unsaved_edits = False
        self._updated_by = socket.gethostname()
        self._detail_base_matrix_up = None
        self._detail_base_matrix_down = None
        self._detail_table_programmatic = False
        self._ignore_detail_json_reload = False
        self._detail_edit_cooldown_until = 0.0
        self._detail_dirty = False
        self._detail_json_save_timer = QTimer(self)
        self._detail_json_save_timer.setSingleShot(True)
        self._detail_json_save_timer.timeout.connect(self._flush_detail_optimization_json)
        self.justsaveone('update_info', True)

        self.initUI1()
        self.initUI2()
        self.initUI3()

        self.all_ok.clicked.connect(self.print_report)
        self.sdandard_show.clicked.connect(self.standard_show_click)
        # 旧版“报告定位”功能已由报告中心统一接管，这里不再暴露入口
        # self.pushButton_id.clicked.connect(self.pushButton_id_click)

        self._apply_compact_layout()
        self._sync_selection_labels()
        self._polish_report_spacing()
        self._init_sync_guard()

    def _init_sync_guard(self):
        # 同步状态条（若 UI 未提供则动态加）
        try:
            if not hasattr(self, "lbl_sync_status"):
                self.lbl_sync_status = QLabel(self)
                self.lbl_sync_status.setStyleSheet("color:#555;font-size:10px;")
                self.lbl_sync_status.setText("")
                # 放在窗口底部左侧（简单不侵入）
                self.lbl_sync_status.setGeometry(QRect(12, self.height() - 28, 600, 18))
                self.lbl_sync_status.show()
        except Exception:
            pass

        try:
            cfg = _rpt_read(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
            self._rptcfg_revision_at_load = int(cfg.get("rptcfg_revision", 0) or 0)
            self._refresh_sync_label(cfg)
        except Exception:
            self._rptcfg_revision_at_load = 0

        from PyQt5.QtCore import QTimer
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(2000)
        self._sync_timer.timeout.connect(self._poll_rptcfg_revision)
        self._sync_timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            if hasattr(self, "lbl_sync_status"):
                self.lbl_sync_status.setGeometry(QRect(12, self.height() - 28, 800, 18))
        except Exception:
            pass

    def _refresh_sync_label(self, cfg: dict = None):
        if cfg is None:
            cfg = _rpt_read(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
        try:
            rev = int(cfg.get("rptcfg_revision", 0) or 0)
        except Exception:
            rev = 0
        ts = str(cfg.get("rptcfg_updated_at", "") or "")
        try:
            if hasattr(self, "lbl_sync_status"):
                self.lbl_sync_status.setText(f"标准同步：revision={rev}  更新时间={ts or '-'}")
        except Exception:
            pass

    def _poll_rptcfg_revision(self):
        meta = _rpt_read_meta(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
        if meta.revision <= int(getattr(self, "_rptcfg_revision_at_load", 0) or 0):
            return
        # 外部更新
        if getattr(self, "_has_unsaved_edits", False):
            ok = QMessageBox.question(
                self,
                "标准已更新",
                "缺陷标准已在其他界面更新。为避免覆盖，请先刷新后再继续编辑。\n\n是否立即刷新（会放弃当前未保存修改）？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ok != QMessageBox.Yes:
                return
        # 强制刷新标准表格/类别
        try:
            self.initUI2()
            self.initUI3()
        except Exception:
            pass
        self._rptcfg_revision_at_load = meta.revision
        self._has_unsaved_edits = False
        self._refresh_sync_label()

    def _sync_selection_labels(self):
        """从 rptcfg.yaml 同步当前选择信息到界面，同时预填打印参数输入框。"""
        try:
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        try:
            _time = str(cfg.get("time", ""))
            _id   = str(cfg.get("id", ""))
            _strip = str(cfg.get("strip_id", ""))

            # 优先把 strip_id 显示为“带钢卡号”（优先卷根目录 config0_snapshot，其次报告目录）
            try:
                root = os.path.join(os.path.join(_REPO_ROOT, "detect result"), str(_time), str(_id))
                strip_n = int(_strip)
                card = ""

                root_snap = os.path.join(root, "config0_snapshot.yaml")
                if os.path.exists(root_snap):
                    with open(root_snap, "r", encoding="utf-8") as f:
                        cfg0 = yaml.safe_load(f) or {}
                    cards = cfg0.get("strip_card_list") or []
                    if isinstance(cards, (list, tuple)) and len(cards) >= strip_n:
                        card = str(cards[strip_n - 1] or "").strip()
                    if not card:
                        card = str(cfg0.get(f"strip_card_{strip_n}", "") or "").strip()

                if not card:
                    folder = _strip_paths.resolve_strip_dir_basename(root, int(strip_n))
                    snap = os.path.join(root, "report", folder, "config0_snapshot.yaml")
                    if os.path.exists(snap):
                        with open(snap, "r", encoding="utf-8") as f:
                            cfg0 = yaml.safe_load(f) or {}
                        cards = cfg0.get("strip_card_list") or []
                        if isinstance(cards, (list, tuple)) and len(cards) >= strip_n:
                            card = str(cards[strip_n - 1] or "").strip()
                        if not card:
                            card = str(cfg0.get(f"strip_card_{strip_n}", "") or "").strip()

                if card:
                    _strip = card
            except Exception:
                pass

            # show_* 标签（现已隐藏，但保留赋值兼容旧逻辑）
            if hasattr(self, "show_time"):
                self.show_time.setText(_time)
            if hasattr(self, "show_id"):
                self.show_id.setText(_id)
            if hasattr(self, "show_cls"):
                self.show_cls.setText(str(cfg.get("check_cls", "")))
            if hasattr(self, "show_strip_id"):
                self.show_strip_id.setText(_strip)
            # 将值预填到可见的打印参数输入框
            if hasattr(self, "time") and not self.time.text():
                self.time.setText(_time)
            if hasattr(self, "id") and not self.id.text():
                self.id.setText(_id)
            if hasattr(self, "strip_id") and not self.strip_id.text():
                self.strip_id.setText(_strip)
        except Exception:
            pass

    def refresh_from_report_center(self):
        """
        报告中心选择后进入「修改界面」时调用：强制从 rptcfg.yaml 刷新左上角显示，
        并同步产品型号下拉框与标准表格，避免复用旧窗口导致信息不一致。
        """
        self._sync_selection_labels()
        # 同步产品型号下拉框选择（若存在）
        try:
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        try:
            key = str(cfg.get("product_cls", "")).strip()
            if hasattr(self, "product_cls"):
                self.product_cls.setText(key)
            if hasattr(self, "product_cls_combo"):
                ix = -1
                for i in range(self.product_cls_combo.count()):
                    if str(self.product_cls_combo.itemData(i)) == key:
                        ix = i
                        break
                if ix >= 0 and self.product_cls_combo.currentIndex() != ix:
                    self.product_cls_combo.setCurrentIndex(ix)
        except Exception:
            pass
        # 刷新允收矩阵
        try:
            self.create_table()
        except Exception:
            pass
        try:
            self.initUI3()
        except Exception:
            pass

    def _apply_compact_layout(self):
        """
        将左上角"报告定位"frame 改造成紧凑的"打印参数"条（日期/质保书号/带钢卡号），
        隐藏其中的报告定位导航元素，保留并重新布局三个关键输入字段，方便用户在打印前
        确认参数。其余控件相应下移。
        """
        _R = QRect

        # 1) 改造 frame：隐藏定位导航装饰，只保留打印参数输入对
        if hasattr(self, "frame"):
            # 不隐藏 show_time/show_id/show_strip_id：你希望左上角直接固定显示报告中心确定的内容
            for attr in ("label_2", "pushButton_id", "show_cls",
                         "label_5", "check_cls", "label_13"):
                if hasattr(self, attr):
                    getattr(self, attr).hide()

            # 标签预留足够宽度，输入框与标签之间留间隙，避免「别」「号」等被遮挡
            if hasattr(self, "label_3"):
                self.label_3.setGeometry(_R(12, 10, 52, 26))
            if hasattr(self, "time"):
                self.time.setGeometry(_R(72, 10, 132, 26))
            if hasattr(self, "label_4"):
                self.label_4.setGeometry(_R(12, 42, 88, 26))
                try:
                    self.label_4.setText("质保书号")
                except Exception:
                    pass
            if hasattr(self, "id"):
                self.id.setGeometry(_R(108, 42, 122, 26))
            if hasattr(self, "label_14"):
                self.label_14.setGeometry(_R(240, 42, 56, 26))
                try:
                    self.label_14.setText("带钢卡号")
                except Exception:
                    pass
            if hasattr(self, "strip_id"):
                self.strip_id.setGeometry(_R(304, 42, 84, 26))

            # 按你的要求：左上角日期/质保书号/带钢卡号不再使用输入框（由报告中心确定）
            if hasattr(self, "time"):
                self.time.hide()
            if hasattr(self, "id"):
                self.id.hide()
            if hasattr(self, "strip_id"):
                self.strip_id.hide()
            if hasattr(self, "pushButton_id"):
                self.pushButton_id.hide()

            # 显示只读标签，并放到原输入框的位置
            if hasattr(self, "show_time"):
                self.show_time.setGeometry(_R(72, 10, 132, 26))
                self.show_time.show()
            if hasattr(self, "show_id"):
                self.show_id.setGeometry(_R(108, 42, 122, 26))
                self.show_id.show()
            if hasattr(self, "show_strip_id"):
                self.show_strip_id.setGeometry(_R(304, 42, 84, 26))
                self.show_strip_id.show()

            self.frame.setGeometry(_R(40, 0, 420, 76))
            self.frame.setToolTip(
                "打印报告所需的基本参数。\n"
                "通过「报告生成」入口选卷后会自动填充；本界面不再提供手动输入。"
            )
            self.frame.show()

        # 2) 缺陷类别区域整体下移，为打印参数条腾出空间
        try:
            if hasattr(self, "label"):
                self.label.setGeometry(_R(50, 82, 220, 28))
        except Exception:
            pass

        try:
            if hasattr(self, "scrollArea01"):
                self.scrollArea01.setGeometry(_R(40, 114, 400, 692))
        except Exception:
            pass

        try:
            if hasattr(self, "pushButton_add"):
                self.pushButton_add.setGeometry(_R(60, 820, 151, 41))
            if hasattr(self, "cls_save"):
                self.cls_save.setGeometry(_R(270, 820, 151, 41))
            if hasattr(self, "all_ok"):
                self.all_ok.setGeometry(_R(60, 873, 361, 51))
        except Exception:
            pass

        # 3) 右侧主区域左移，填补原 frame 区域的空白
        try:
            if hasattr(self, "frame_2"):
                self.frame_2.setGeometry(_R(440, 0, 900, 991))
        except Exception:
            pass

    def _polish_report_spacing(self):
        """微调标签与输入框间距，避免中文标签与控件边缘重叠。"""
        _R = QRect
        try:
            # 右侧：产品型号行（「产品类别」与输入框、按钮拉开）
            if hasattr(self, "label_12"):
                # 加宽，避免「类」字贴边/被遮
                self.label_12.setGeometry(_R(20, 44, 132, 30))
            if hasattr(self, "product_cls"):
                self.product_cls.setGeometry(_R(160, 44, 172, 30))
            # 产品型号下拉框（替代 product_cls 输入框）
            if hasattr(self, "product_cls_combo"):
                # 左移 + 加宽，避免与「查看」重叠；并与「查看」同高
                self.product_cls_combo.setGeometry(_R(160, 44, 220, 30))
                self.product_cls_combo.setMinimumHeight(30)
                self.product_cls_combo.setStyleSheet("QComboBox{padding:2px 10px;}")
            if hasattr(self, "sdandard_show"):
                # 放到下拉框右侧，不遮挡
                self.sdandard_show.setGeometry(_R(388, 44, 72, 30))
            if hasattr(self, "_btn_open_cls_cfg"):
                # 加宽，避免「类别配置」文字被遮挡
                self._btn_open_cls_cfg.setGeometry(468, 44, 104, 30)
            if hasattr(self, "btnWizard"):
                # 缺陷分类向导：紧贴「类别配置→」右侧
                self.btnWizard.setParent(self.frame_2)
                self.btnWizard.setGeometry(580, 44, 110, 30)
                self.btnWizard.show()
            if hasattr(self, "btnRollback"):
                # 撤销上次修改：标题行右侧，与「强制同步」同行
                self.btnRollback.setParent(self.frame_2)
                self.btnRollback.setGeometry(462, 8, 162, 28)
                self.btnRollback.show()
            if hasattr(self, "standard_save"):
                # 强制同步：标题行最右
                self.standard_save.setGeometry(_R(632, 8, 120, 28))

            # 分类模型行：放在产品类别行下方
            if hasattr(self, "lblModel"):
                self.lblModel.setGeometry(_R(20, 78, 132, 28))
            if hasattr(self, "comboModel"):
                self.comboModel.setGeometry(_R(160, 78, 220, 28))
            if hasattr(self, "btnModelView"):
                self.btnModelView.setGeometry(_R(388, 78, 80, 28))
            if hasattr(self, "btnModelEnable"):
                self.btnModelEnable.setGeometry(_R(472, 78, 64, 28))
            if hasattr(self, "lblModelStatus"):
                # 紧贴「分类模型：」右侧并大幅加宽，避免长路径被挤到窗口右缘截断
                self.lblModelStatus.setGeometry(_R(160, 78, 720, 28))

            # 允收标准表格下移 + 缩高，腾出“分类模型行”
            if hasattr(self, "tableWidget_standard"):
                self.tableWidget_standard.setGeometry(_R(30, 118, 711, 313))

            # 右侧：优化选项行（打印面积 / 打印类别）
            if hasattr(self, "label_8"):
                self.label_8.setGeometry(_R(20, 476, 100, 28))
            if hasattr(self, "lineEdit_area"):
                self.lineEdit_area.hide()
            if hasattr(self, '_combo_area') and self._combo_area is not None:
                self._combo_area.setGeometry(_R(128, 472, 172, 34))
                self._combo_area.setMinimumHeight(34)
            if hasattr(self, "label_9"):
                self.label_9.setGeometry(_R(318, 476, 100, 28))
            if hasattr(self, "lineEdit_cls"):
                self.lineEdit_cls.hide()
            if hasattr(self, '_combo_cls') and self._combo_cls is not None:
                self._combo_cls.setGeometry(_R(426, 472, 172, 34))
                self._combo_cls.setMinimumHeight(34)
            if hasattr(self, "change_save"):
                self.change_save.setGeometry(_R(688, 512, 151, 41))

            # Tab 内标题大致居中
            if hasattr(self, "label_10"):
                self.label_10.setGeometry(_R(270, 8, 200, 28))
            if hasattr(self, "label_11"):
                self.label_11.setGeometry(_R(270, 8, 200, 28))
        except Exception:
            pass

    def _apply_table_spacing(self):
        """表头/行标题增加内边距与最小宽度，避免文字贴边。"""
        _tbl_style = (
            "QTableWidget { gridline-color: #c8c8c8; }"
            "QHeaderView::section { padding: 6px 10px; min-height: 28px; }"
        )
        for tw in (
            getattr(self, "tableWidget_standard", None),
            getattr(self, "tableWidget_change", None),
            getattr(self, "tableWidget_change_2", None),
        ):
            if tw is None:
                continue
            tw.setStyleSheet(_tbl_style)
            vh = tw.verticalHeader()
            vh.setSectionResizeMode(QHeaderView.Fixed)
            vh.setMinimumWidth(92)
            vh.setDefaultAlignment(Qt.AlignCenter)
            hh = tw.horizontalHeader()
            hh.setDefaultAlignment(Qt.AlignCenter)
            hh.setMinimumSectionSize(72)

    def _detail_session_key(self):
        try:
            cfg = _rpt_read(os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml"))
        except Exception:
            cfg = {}
        date_s = ""
        cid = ""
        strip_raw = "1"
        if hasattr(self, "show_time") and self.show_time.text().strip():
            date_s = self.show_time.text().strip()
        else:
            date_s = str(cfg.get("time", "")).strip()
        if hasattr(self, "show_id") and self.show_id.text().strip():
            cid = self.show_id.text().strip()
        else:
            cid = str(cfg.get("id", "")).strip()
        if hasattr(self, "show_strip_id") and self.show_strip_id.text().strip():
            strip_raw = self.show_strip_id.text().strip()
        else:
            strip_raw = str(cfg.get("strip_id", "1")).strip()
        try:
            strip_n = str(max(1, int(float(strip_raw))))
        except Exception:
            strip_n = "1"
        return date_s, cid, strip_n

    def _ensure_detail_json_watcher(self):
        if getattr(self, "_detail_opt_watcher", None) is not None:
            return
        try:
            self._detail_opt_watcher = QFileSystemWatcher(self)
            d = os.path.dirname(_DETAIL_OPT_JSON)
            if os.path.isdir(d):
                self._detail_opt_watcher.addPath(d)
            if os.path.isfile(_DETAIL_OPT_JSON):
                self._detail_opt_watcher.addPath(_DETAIL_OPT_JSON)
            self._detail_opt_watcher.directoryChanged.connect(self._on_detail_json_fs_changed)
            self._detail_opt_watcher.fileChanged.connect(self._on_detail_json_fs_changed)
        except Exception:
            self._detail_opt_watcher = None

    def _on_detail_json_fs_changed(self, _path=None):
        if self._ignore_detail_json_reload:
            return
        if self._detail_is_user_editing():
            return
        # 若用户有未保存编辑，不要回读覆盖
        if getattr(self, "_detail_dirty", False):
            return
        QTimer.singleShot(350, self._try_reload_detail_json)

    def _try_reload_detail_json(self):
        if self._ignore_detail_json_reload or self._detail_table_programmatic:
            return
        if self._detail_is_user_editing():
            return
        if getattr(self, "_detail_dirty", False):
            return
        date_s, cid, strip_n = self._detail_session_key()
        if not date_s or not cid:
            return
        obj = _load_json_safe(_DETAIL_OPT_JSON, None)
        if not isinstance(obj, dict):
            return
        if str(obj.get("date", "")) != date_s or str(obj.get("conduct_id", "")) != cid or str(obj.get("strip_id", "")) != strip_n:
            return
        mu, md = obj.get("matrix_up"), obj.get("matrix_down")
        if not isinstance(mu, list) or not isinstance(md, list):
            return
        try:
            self._fill_detail_tables_from_matrices(mu, md, from_json=True)
        except Exception:
            pass

    def _fill_detail_tables_from_matrices(self, matrix_up, matrix_down, from_json=False):
        self._detail_table_programmatic = True
        try:
            for ti, mat in ((self.tableWidget_change, matrix_up), (self.tableWidget_change_2, matrix_down)):
                if ti is None or not mat:
                    continue
                for i in range(min(ti.rowCount(), len(mat))):
                    row = mat[i]
                    if not isinstance(row, (list, tuple)):
                        continue
                    for j in range(min(ti.columnCount(), len(row))):
                        v = row[j]
                        it = ti.item(i, j)
                        if it is None:
                            it = QTableWidgetItem("")
                            ti.setItem(i, j, it)
                        it.setText(str(int(v)) if str(v).strip().isdigit() or (isinstance(v, int) and v >= 0) else "")
        finally:
            self._detail_table_programmatic = False
        if from_json:
            try:
                if hasattr(self, "lbl_sync_status"):
                    self.lbl_sync_status.setText(
                        (self.lbl_sync_status.text() or "")
                        + "  细节表：已从 detail_optimization.json 同步"
                    )
            except Exception:
                pass

    def _detail_is_user_editing(self) -> bool:
        """
        细节优化表格正在编辑时，禁止任何“自动同步/回刷”覆盖用户输入。
        - 正在编辑（EditingState）或控件有焦点时均视为编辑中
        - 编辑后短暂冷却窗口内也禁止回刷（避免 watcher / timer 抢写）
        """
        try:
            if time.time() < float(getattr(self, "_detail_edit_cooldown_until", 0.0) or 0.0):
                return True
        except Exception:
            pass
        try:
            for w in (getattr(self, "tableWidget_change", None), getattr(self, "tableWidget_change_2", None)):
                if w is None:
                    continue
                try:
                    if w.state() == QTableWidget.EditingState:
                        return True
                except Exception:
                    pass
                if w.hasFocus():
                    return True
        except Exception:
            pass
        return False

    def _matrices_from_tables(self):
        mu, md = [], []
        for ti, acc in ((self.tableWidget_change, mu), (self.tableWidget_change_2, md)):
            if ti is None:
                acc.append([])
                continue
            for i in range(ti.rowCount()):
                row = []
                for j in range(ti.columnCount()):
                    it = ti.item(i, j)
                    t = (it.text() if it else "").strip()
                    row.append(int(t) if t.isdigit() else 0)
                acc.append(row)
        return mu, md

    def _flush_detail_optimization_json(self):
        if self._detail_table_programmatic:
            return
        date_s, cid, strip_n = self._detail_session_key()
        mu, md = self._matrices_from_tables()
        self._ignore_detail_json_reload = True
        try:
            os.makedirs(os.path.dirname(_DETAIL_OPT_JSON), exist_ok=True)
            payload = {
                "date": date_s,
                "conduct_id": cid,
                "strip_id": strip_n,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "column_headers": list(getattr(self, "column_headers", []) or []),
                "row_headers": list(getattr(self, "row_headers", []) or []),
                "matrix_up": mu,
                "matrix_down": md,
            }
            with open(_DETAIL_OPT_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[detail_optimization.json] 写入失败: {e}")
        finally:
            QTimer.singleShot(600, lambda: setattr(self, "_ignore_detail_json_reload", False))

    def _schedule_detail_json_persist(self):
        if self._detail_table_programmatic:
            return
        # 编辑不落盘：只标记“未保存”，并开启一个短冷却窗口避免外部回刷覆盖
        try:
            self._detail_edit_cooldown_until = time.time() + 1.2
        except Exception:
            pass
        self._detail_dirty = True
        try:
            if hasattr(self, "lbl_sync_status"):
                t = self.lbl_sync_status.text() or ""
                if "细节优化未保存" not in t:
                    self.lbl_sync_status.setText(t + "  细节优化未保存")
        except Exception:
            pass

    def _refresh_detail_optimization_from_data(self):
        """从检测目录统计缺陷数量填入上下表；优先用 detail_optimization.json 覆盖显示（便于外部改数）。基线矩阵始终来自原始统计，供保存时生成 updates 差分。"""
        if not hasattr(self, "row2") or not hasattr(self, "col2"):
            return
        # 若用户正在编辑，不要自动回刷覆盖输入
        if self._detail_is_user_editing():
            return
        # 若用户有未保存编辑，不要自动重算/覆盖
        if getattr(self, "_detail_dirty", False):
            return
        date_s, cid, strip_n_s = self._detail_session_key()
        try:
            strip_n = max(1, int(float(strip_n_s)))
        except Exception:
            strip_n = 1
        result_root = os.path.join(_DETECT_RESULT_DIR, date_s, cid)
        strip_folder = _strip_paths.resolve_strip_dir_basename(result_root, int(strip_n))
        use_json = False

        cfg_path = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
        rpt_path = os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}
        try:
            with open(rpt_path, "r", encoding="utf-8") as f:
                rptcfg = yaml.safe_load(f) or {}
        except Exception:
            rptcfg = {}

        print_cls = rptcfg.get("print_cls", []) or rptcfg.get("class_list", [])
        if not print_cls:
            print_cls = rptcfg.get("class_list", [])

        df_up0 = df_dn0 = None
        if date_s and cid and os.path.isdir(result_root):
            try:
                stat = _make_statistic_for_detail(config, rptcfg)
                up_cams = config.get("camrea_id_up_report", []) or []
                down_cams = config.get("camrea_id_down_report", []) or []
                df_up0 = _surface_area_table_from_files(result_root, strip_folder, up_cams, stat, print_cls)
                df_dn0 = _surface_area_table_from_files(result_root, strip_folder, down_cams, stat, print_cls)
            except Exception as e:
                print(f"[细节优化] 统计失败: {e}")

        self._detail_table_programmatic = True
        base_up, base_down = [], []
        try:
            for i in range(self.row2):
                r_up, r_dn = [], []
                for j in range(self.col2):
                    lab = self.row_headers[i]
                    ch = self.column_headers[j]
                    bu = _df_to_cell_int(df_up0, lab, ch) if df_up0 is not None else 0
                    bd = _df_to_cell_int(df_dn0, lab, ch) if df_dn0 is not None else 0
                    r_up.append(bu)
                    r_dn.append(bd)
                base_up.append(r_up)
                base_down.append(r_dn)
            self._detail_base_matrix_up = base_up
            self._detail_base_matrix_down = base_down

            obj = _load_json_safe(_DETAIL_OPT_JSON, None)
            # 全 0 的 json 多为误生成或占位，应重新从 detect result 统计，避免表格永远为 0
            use_json = (
                bool(date_s)
                and bool(cid)
                and isinstance(obj, dict)
                and str(obj.get("date", "")) == date_s
                and str(obj.get("conduct_id", "")) == cid
                and str(obj.get("strip_id", "")) == str(strip_n_s)
                and isinstance(obj.get("matrix_up"), list)
                and isinstance(obj.get("matrix_down"), list)
                and _detail_json_matrix_sum(obj) > 0
            )
            if use_json:
                self._detail_table_programmatic = False
                self._fill_detail_tables_from_matrices(obj["matrix_up"], obj["matrix_down"], from_json=True)
                self._detail_table_programmatic = True
            else:
                df_up = _apply_sparse_updates_to_df(df_up0, rptcfg.get("updates_up") or []) if df_up0 is not None else None
                df_dn = _apply_sparse_updates_to_df(df_dn0, rptcfg.get("updates_down") or []) if df_dn0 is not None else None
                for i in range(self.row2):
                    for j in range(self.col2):
                        lab = self.row_headers[i]
                        ch = self.column_headers[j]
                        vu = _df_to_cell_int(df_up, lab, ch) if df_up is not None else 0
                        vd = _df_to_cell_int(df_dn, lab, ch) if df_dn is not None else 0
                        iu = self.tableWidget_change.item(i, j)
                        if iu is None:
                            iu = QTableWidgetItem("")
                            self.tableWidget_change.setItem(i, j, iu)
                        iu.setText(str(vu))
                        idn = self.tableWidget_change_2.item(i, j)
                        if idn is None:
                            idn = QTableWidgetItem("")
                            self.tableWidget_change_2.setItem(i, j, idn)
                        idn.setText(str(vd))
        finally:
            self._detail_table_programmatic = False

        # 编辑不落盘：这里只确保 watcher 存在（便于外部修改 json 时同步）
        self._ensure_detail_json_watcher()

    def _connect_detail_table_edit_signals(self):
        for w in (getattr(self, "tableWidget_change", None), getattr(self, "tableWidget_change_2", None)):
            if w is None:
                continue
            try:
                w.itemChanged.disconnect(self._schedule_detail_json_persist)
            except Exception:
                pass
            w.itemChanged.connect(self._schedule_detail_json_persist)

    def closeEvent(self, event):
        # 关闭窗口时将 update_info 置 False，表示本次报告会话结束；
        # 报告中心下次生成前会重新置 True 并写入当前选择。
        self.justsaveone('update_info', False)
        super().closeEvent(event)

    def print_report(self):
        # 安全闸：当本次会触发“分类推理”(update_info=False)时，必须保证模型与当前类别兼容
        try:
            cfg0 = _rpt_read(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
            update_info = bool(cfg0.get("update_info", False))
        except Exception:
            update_info = True
        if not update_info:
            e = getattr(self, "_selected_model_entry", None)
            if e is None:
                QMessageBox.warning(self, "无法打印", "未选择分类模型。请先在“分类模型”下拉中选择可用模型并启用。")
                return
            if not getattr(e, "classes", None):
                QMessageBox.warning(self, "无法打印", "当前模型缺少 classes.json（类别清单），为避免错位无法进行分类推理。")
                return
            rpt_names = _rptcfg_class_names(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
            ok, remap, diff = _cls_compat_and_remap(model_classes=list(e.classes or []), rptcfg_classes=list(rpt_names or []))
            if not ok:
                QMessageBox.warning(
                    self,
                    "模型不可用",
                    "当前启用/选择的模型与缺陷类别标准不一致，无法进行分类推理并生成报告。\n\n"
                    f"缺少：{', '.join(diff.get('missing') or []) or '-'}\n"
                    f"多出：{', '.join(diff.get('extra') or []) or '-'}\n\n"
                    "请先进入缺陷分类向导按当前类别重新训练，或选择类别一致的模型。",
                )
                return
            # 确保 remap 写入 runtime_state，供分类脚本使用
            try:
                runtime_state_path = os.path.join(_REPO_ROOT, "config", "runtime_state.json")
                try:
                    with open(os.path.join(_REPO_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                except Exception:
                    cfg = {}
                model_path = str((cfg or {}).get("cls_model_path", "") or "")
                _write_runtime_remap(
                    runtime_state_path,
                    model_path=model_path,
                    model_classes=list(e.classes or []),
                    rptcfg_classes=list(rpt_names or []),
                    remap_modelidx_to_rptid_1based=list(remap or []),
                )
            except Exception:
                pass

        self.non_blocking_information(self, "报告打印", "报告打印中，请等待。")
        try:
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
                rptcfg = yaml.safe_load(f) or {}
        except Exception:
            rptcfg = {}
        time = (self.time.text() or "").strip() or str(rptcfg.get("time", ""))
        cid = (self.id.text() or "").strip() or str(rptcfg.get("id", ""))
        strip = (self.strip_id.text() or "").strip() or str(rptcfg.get("strip_id", "1"))
        if not time or not cid:
            QMessageBox.warning(
                self,
                "信息不全",
                "请先填写日期、质保书号（或在上方确认已保存到配置），再打印报告。",
            )
            return
        try:
            strip_n = str(max(1, int(float(strip))))
        except (ValueError, TypeError):
            strip_n = "1"
        args = [
            _PYTHON_EXE,
            "-u",
            "-m",
            _GEN_REPORT_MODULE,
            "--date",
            time,
            "--conduct-id",
            cid,
            "--strip-id",
            strip_n,
            "--update-info",
            "true",
        ]
        self.python_process = subprocess.Popen(args, cwd=os.path.join(_REPO_ROOT))


    # 替代 QMessageBox.information 的非阻塞实现
    def non_blocking_information(self, parent, title, message):
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.show()  # 非阻塞显示
        return msg_box  # 返回消息框对象（如果需要进一步操作）

    def pushButton_id_click(self):
        self.config_data['update_info'] = True
        self.justsaveone('update_info', True)
        try:
            time = self.time.text()
            id = self.id.text()
            strip_id = self.strip_id.text()
            check_cls = self.check_cls.text()
            self.config_data['time'] = time
            self.config_data['id'] = id
            self.config_data['check_cls'] = check_cls
            self.config_data['strip_id'] = strip_id
            self.justsaveone('time', time)
            self.justsaveone('id', id)
            self.justsaveone('check_cls', check_cls)
            self.justsaveone('strip_id', strip_id)

            with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.show_time.setText(str(config['time']))
            self.show_id.setText(str(config['id']))
            self.show_cls.setText(str(config['check_cls']))
            self.show_strip_id.setText(str(config['strip_id']))
            result = self.find_folders_with_id(time, _DETECT_RESULT_DIR, id, strip_id)
            if result:
                self.find_and_open_image(result[0])
            else:
                QMessageBox.warning(self, "目录不存在",
                                    f"未找到报告目录：\n"
                                    f"  日期：{time}\n"
                                    f"  质保书号：{id}\n"
                                    f"  带钢号：{strip_id}\n\n"
                                    "请确认已完成报告生成，且日期/质保书号/带钢号与生成时一致。")
        except Exception as e:
            QMessageBox.information(self, "输入错误", "请输入正确的报告信息！")


    def find_and_open_image(self, folder_path):
        """按优先级搜索报告产物并用系统默认程序打开。"""
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "目录不存在",
                                f"报告目录不存在：\n{folder_path}\n\n"
                                "请确认日期/质保书号/带钢号输入正确，并已完成报告生成。")
            return

        # 按优先级依次尝试打开
        candidates = []
        # 1. 检测报告 PDF
        for fn in os.listdir(folder_path):
            if fn.endswith(".pdf"):
                candidates.append(os.path.join(folder_path, fn))
        # 2. 上表面分布图 PNG
        for fn in os.listdir(folder_path):
            if fn.endswith(".png") and "上表面" in fn and "distribution" in fn:
                candidates.append(os.path.join(folder_path, fn))
        # 3. 任意 PNG
        for fn in os.listdir(folder_path):
            if fn.endswith(".png"):
                candidates.append(os.path.join(folder_path, fn))

        if not candidates:
            QMessageBox.information(self, "暂无产物",
                                    f"目录内未找到报告文件（PDF/PNG）：\n{folder_path}")
            return

        target = candidates[0]
        try:
            os.startfile(target)
            print(f"已打开报告文件: {target}")
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"无法打开文件：\n{target}\n\n{e}")

    def find_folders_with_id(self, date_str, base_path, product_id, strip_id):
        """
        定位报告子目录：{base_path}\\{date_str}\\{product_id}\\report\\strip_{strip_id}
        与 gen_report_cls.py 生成的目录结构保持一致。

        :param date_str:   日期字符串，格式 YYYYMMDD
        :param base_path:  detect result 根目录
        :param product_id: 生产卡号
        :param strip_id:   钢带号（1 基准，字符串或整数均可）
        :return:           [path] 若存在，否则 []
        """
        try:
            strip_n = max(1, int(float(str(strip_id))))
        except Exception:
            strip_n = 1

        roll = os.path.join(base_path, date_str, product_id)
        primary = _strip_paths.resolve_strip_dir_basename(roll, int(strip_n))
        candidates = [
            os.path.join(roll, "report", primary),
            os.path.join(roll, "report", f"strip_{strip_n}"),
        ]
        for target in candidates:
            if os.path.exists(target):
                print(f"找到报告目录: {target}")
                return [target]

        print(f"报告目录不存在: 依次尝试 {candidates}")
        return []

    def initUI3(self):
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as f:
            cfg2 = yaml.safe_load(f)

        # ── 打印面积 多选下拉 ────────────────────────────────────────────────
        if not hasattr(self, '_combo_area') or self._combo_area is None:
            self._combo_area = CheckableComboBox(self.frame_2)
        else:
            while self._combo_area._mdl.rowCount():
                self._combo_area._mdl.removeRow(0)
        area_ranges = cfg2.get('anomaly_area_cls_range', [])
        for a, b in area_ranges:
            self._combo_area.addItem(f"[{a}, {b}]", [a, b])
        if area_ranges:
            last_b = area_ranges[-1][1]
            self._combo_area.addItem(f"> {last_b}", [last_b, None])
        cur_area = config.get('area_range', [])
        if not cur_area:
            self._combo_area.check_all()
        self._combo_area._refresh_text()
        self._combo_area.show()
        if hasattr(self, 'lineEdit_area'):
            self.lineEdit_area.hide()

        # ── 打印类别 多选下拉 ────────────────────────────────────────────────
        if not hasattr(self, '_combo_cls') or self._combo_cls is None:
            self._combo_cls = CheckableComboBox(self.frame_2)
        else:
            while self._combo_cls._mdl.rowCount():
                self._combo_cls._mdl.removeRow(0)
        class_labels = config.get('class_labels', {})
        for k in sorted(class_labels.keys(), key=lambda x: int(x)):
            self._combo_cls.addItem(str(class_labels[k]), int(k))
        cur_cls = config.get('print_cls', [])
        if not cur_cls:
            self._combo_cls.check_all()
        else:
            self._combo_cls.set_checked_values(cur_cls)
        self._combo_cls._refresh_text()
        self._combo_cls.show()
        if hasattr(self, 'lineEdit_cls'):
            self.lineEdit_cls.hide()

        # 生成表格
        self.create_table2()
        self.create_table3()

        try:
            self.change_save.clicked.disconnect()
        except Exception:
            pass
        self.change_save.clicked.connect(self.change_data)
        try:
            self.change_save.setToolTip(
                "保存打印面积/类别筛选与细节优化表格到 rptcfg.yaml，并同步 config/detail_optimization.json。\n"
                "不会自动打印报告；打印请使用窗口左下方「全部完成」等按钮。"
            )
        except Exception:
            pass
        self._refresh_detail_optimization_from_data()
        self._connect_detail_table_edit_signals()

    def create_table2(self):

        # 获取行列数
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        class_labels = data.get("class_labels", {}) or {}
        class_list_order = data.get("class_list", []) or []
        self.row_headers = []
        if class_list_order:
            for c in class_list_order:
                try:
                    ci = int(c)
                except Exception:
                    continue
                v = class_labels.get(str(ci))
                if v is None:
                    v = class_labels.get(ci)
                if v is not None:
                    self.row_headers.append(v)
        else:
            for k in sorted(int(x) for x in class_labels.keys()):
                v = class_labels.get(k) or class_labels.get(str(k))
                if v is not None:
                    self.row_headers.append(v)
        self.row2 = len(self.row_headers)

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        anomaly_area_cls_range = config.get('anomaly_area_cls_range', [])
        last_end = anomaly_area_cls_range[-1][1]  # 获取最后一个范围的结束值
        self.column_headers = [f"[{start}, {end}]" for start, end in anomaly_area_cls_range] + [f"> {last_end}"]
        self.col2 = len(anomaly_area_cls_range)+1

        rows = self.row2
        cols = self.col2

        # 设置表格行列数
        self.tableWidget_change.setRowCount(rows)
        self.tableWidget_change.setColumnCount(cols)

        # 设置行标题和列标题
        self.tableWidget_change.setHorizontalHeaderLabels(self.column_headers)
        self.tableWidget_change.setVerticalHeaderLabels(self.row_headers)
        _apply_matrix_corner_labels(self.tableWidget_change, tr_text="面积", bl_text="缺陷")

        # 初始化表格内容
        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem("")  # 默认值为 空
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_change.setItem(i, j, item)

        # 调整列宽
        for col in range(cols):
            self.tableWidget_change.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(rows):
            self.tableWidget_change.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def create_table3(self):


        # 设置表格行列数
        self.tableWidget_change_2.setRowCount(self.row2)
        self.tableWidget_change_2.setColumnCount(self.col2)

        # 设置行标题和列标题
        self.tableWidget_change_2.setHorizontalHeaderLabels(self.column_headers)
        self.tableWidget_change_2.setVerticalHeaderLabels(self.row_headers)
        _apply_matrix_corner_labels(self.tableWidget_change_2, tr_text="面积", bl_text="缺陷")

        # 初始化表格内容
        for i in range(self.row2):
            for j in range(self.col2):
                item = QTableWidgetItem("")  # 默认值为 1000
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_change_2.setItem(i, j, item)

        # 调整列宽
        for col in range(self.col2):
            self.tableWidget_change_2.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(self.row2):
            self.tableWidget_change_2.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def change_data(self):
        self.updates_up = []
        self.updates_down = []
        # ── 打印面积（从多选下拉读取；全选 = [] 表示不过滤）
        try:
            if hasattr(self, '_combo_area') and self._combo_area is not None:
                sel = self._combo_area.selected_data()
                total = self._combo_area._mdl.rowCount()
                area_range = [] if (len(sel) == 0 or len(sel) == total) else sel
            else:
                area_range = ast.literal_eval(self.lineEdit_area.text().strip())
        except Exception:
            area_range = []
        # ── 打印类别（从多选下拉读取；全选 = [] 表示打印全部类别）
        try:
            if hasattr(self, '_combo_cls') and self._combo_cls is not None:
                sel_cls = self._combo_cls.selected_data()
                total_cls = self._combo_cls._mdl.rowCount()
                print_cls = [] if (len(sel_cls) == 0 or len(sel_cls) == total_cls) else [int(d) for d in sel_cls]
            else:
                print_cls = ast.literal_eval(self.lineEdit_cls.text().strip())
        except Exception:
            print_cls = []
        self.config_data['area_range'] = area_range
        self.config_data['print_cls'] = print_cls

        # 整表写入 updates_*：避免「只存差分」时下次保存覆盖 yaml 把先前改过的格子丢掉，导致打印仍用旧检出数
        for i in range(self.row2):
            for j in range(self.col2):
                item = self.tableWidget_change.item(i, j)
                text = (item.text() if item else "").strip()
                if text.isdigit() or text == "0":
                    nu = int(text)
                else:
                    nu = 0
                if nu < 0:
                    nu = 0
                self.updates_up.append(
                    {
                        "class_label": self.row_headers[i],
                        "area_interval": self.column_headers[j],
                        "new_count": nu,
                    }
                )
        self.config_data['updates_up'] = self.updates_up
        print(self.updates_up)

        for i in range(self.row2):
            for j in range(self.col2):
                item2 = self.tableWidget_change_2.item(i, j)
                text2 = (item2.text() if item2 else "").strip()
                if text2.isdigit() or text2 == "0":
                    nd = int(text2)
                else:
                    nd = 0
                if nd < 0:
                    nd = 0
                self.updates_down.append(
                    {
                        "class_label": self.row_headers[i],
                        "area_interval": self.column_headers[j],
                        "new_count": nd,
                    }
                )
        self.config_data['updates_down'] = self.updates_down
        print(self.updates_down)
        self.justsaveone("area_range", area_range, skip_auto_report=True)
        self.justsaveone("print_cls", print_cls, skip_auto_report=True)
        self.justsaveone("updates_up", self.updates_up, skip_auto_report=True)
        self.justsaveone("updates_down", self.updates_down, skip_auto_report=True)
        # 保存时才落盘 detail_optimization.json
        self._flush_detail_optimization_json()
        self._detail_dirty = False
        try:
            if hasattr(self, "lbl_sync_status"):
                t = self.lbl_sync_status.text() or ""
                if "细节优化已保存" not in t:
                    self.lbl_sync_status.setText(t + "  细节优化已保存（未自动打印）。")
        except Exception:
            pass

    def initUI2(self):
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        key = str(config.get('product_cls', ''))
        # 合格标准中：产品类型改为下拉框选择（来自 rptcfg.yaml）
        try:
            from cls_config import product_combo_entries
            entries = product_combo_entries()
        except Exception:
            entries = []

        # 仍保留 product_cls QLineEdit 作为内部数据载体，但不展示给用户
        self.product_cls.setText(key)
        self.product_cls.setReadOnly(True)
        try:
            self.product_cls.setToolTip("产品型号（来自 rptcfg）。可在下方下拉框选择后刷新表格。")
        except Exception:
            pass
        self.product_cls.setStyleSheet("background: #f5f5f5; color: #555;")

        # 创建下拉框并映射到同一几何
        if not hasattr(self, "product_cls_combo"):
            self.product_cls_combo = QComboBox(self.frame_2)
        # 尺寸/位置由 _polish_report_spacing 统一管理；这里给一个安全的最小宽度兜底
        self.product_cls_combo.setMinimumWidth(220)
        self.product_cls_combo.setMinimumHeight(30)
        try:
            self.product_cls_combo.setStyleSheet("QComboBox{padding:2px 10px;}")
        except Exception:
            pass

        self.product_cls_combo.clear()
        for k, disp in entries:
            if not k:
                continue
            self.product_cls_combo.addItem(str(disp), str(k))

        # 设置初始选择
        ix = -1
        for i in range(self.product_cls_combo.count()):
            if str(self.product_cls_combo.itemData(i)) == key:
                ix = i
                break
        if ix >= 0:
            self.product_cls_combo.setCurrentIndex(ix)

        # 显示下拉框，隐藏原输入框
        self.product_cls_combo.show()
        self.product_cls.hide()

        def _on_cls_changed(_=None):
            selected_key = str(self.product_cls_combo.currentData() or "")
            if not selected_key:
                return
            self.product_cls.setText(selected_key)
            # 切换产品类型后刷新允收矩阵视图
            try:
                self.standard_show_click()
            except Exception:
                # 兜底：至少重建表格
                try:
                    self.create_table()
                except Exception:
                    pass

        # 避免重复连接
        try:
            self.product_cls_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.product_cls_combo.currentIndexChanged.connect(_on_cls_changed)

        # 「类别配置」快捷按钮（放在产品类别行旁边）
        if not hasattr(self, '_btn_open_cls_cfg'):
            from PyQt5.QtWidgets import QPushButton
            from PyQt5.QtGui import QFont
            self._btn_open_cls_cfg = QPushButton("类别配置 →", self.frame_2)
            self._btn_open_cls_cfg.setGeometry(360, 47, 100, 30)
            f = QFont("Arial", 10); f.setBold(True)
            self._btn_open_cls_cfg.setFont(f)
            self._btn_open_cls_cfg.setStyleSheet(
                "QPushButton{background:#E8F5E9;color:#2e7d32;"
                "border:1px solid #a5d6a7;border-radius:4px;}"
                "QPushButton:hover{background:#C8E6C9;}"
            )
            self._btn_open_cls_cfg.setToolTip(
                "打开类别配置窗口（需密码，见 config/auth.yaml 的 cls_config）。"
                "维护型号、缺陷类别和允收矩阵。"
            )
            self._btn_open_cls_cfg.clicked.connect(self._open_cls_config)
            self._btn_open_cls_cfg.show()
            self._cls_config_window = None

        # 缺陷分类向导按钮（frame_2，紧贴「类别配置→」右侧，由 _polish_report_spacing 定位）
        if not hasattr(self, "btnWizard"):
            self.btnWizard = QPushButton("缺陷分类向导", self.frame_2)
            self.btnWizard.setStyleSheet(
                "QPushButton{background:#FFF3E0;color:#E65100;border:1px solid #FFCC80;"
                "border-radius:4px;padding:4px 8px;font-weight:bold;}"
                "QPushButton:hover{background:#FFE0B2;}"
            )
            self.btnWizard.setToolTip("一步一步完成缺陷类型配置与训练，工人小白也能操作。")
            self.btnWizard.clicked.connect(self._open_wizard_from_report)
            self.btnWizard.show()

        # —— 分类模型选择（工人可查看/选择可用模型）——
        # 按你的要求：报告修改界面不提供“模型选择/启用”，只展示当前模型信息（模型选择统一在“类别配置”里完成）
        if not hasattr(self, "lblModel"):
            self.lblModel = QLabel("分类模型：", self.frame_2)
            self.lblModel.setStyleSheet("font-weight:bold;")
        if not hasattr(self, "lblModelStatus"):
            self.lblModelStatus = QLabel("", self.frame_2)
            self.lblModelStatus.setStyleSheet("color:#555;")
            self.lblModelStatus.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.lblModelStatus.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # 兼容旧控件：若存在则隐藏
        for attr in ("comboModel", "btnModelView", "btnModelEnable"):
            if hasattr(self, attr):
                try:
                    getattr(self, attr).hide()
                except Exception:
                    pass
        self._refresh_current_model_info()

        # 生成表格（只读视图；修改请通过「类别配置」）
        self.create_table()
        from PyQt5.QtWidgets import QAbstractItemView
        self.tableWidget_standard.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget_standard.setToolTip(
            "当前型号的允收标准（只读）。\n"
            "如需修改请点击「类别配置 →」按钮，在类别配置窗口中统一维护。"
        )

        # 保留导出按钮但更新说明，提示优先走类别配置
        self.standard_save.clicked.connect(self.export_data)
        self.standard_save.setText("强制同步")
        self.standard_save.setToolTip(
            "将当前表格内容强制写回 rptcfg.yaml（紧急覆盖用途）。\n"
            "正常情况下请通过「类别配置」窗口维护，避免双份编辑。"
        )

    def _refresh_model_registry(self):
        project_root = os.path.join(_REPO_ROOT)
        config_path = os.path.join(project_root, "config", "config.yaml")
        rptcfg_path = os.path.join(project_root, "config", "rptcfg.yaml")
        try:
            self._model_entries = _scan_model_registry(
                project_root=project_root, config_path=config_path, rptcfg_path=rptcfg_path, keep_cache=True
            )
        except Exception:
            self._model_entries = []

        self.comboModel.clear()
        if not self._model_entries:
            self.comboModel.addItem("（未发现可用模型）", None)
            self.btnModelEnable.setEnabled(False)
            self.btnModelView.setEnabled(False)
            self.lblModelStatus.setText("未找到模型产物（请先训练或拷贝模型，并确保同目录有 classes.json）。")
            return

        for e in self._model_entries:
            head = f"{e.trained_at} | {e.num_classes}类 | {e.source}"
            preview = "、".join((e.classes or [])[:4])
            text = head + (f" | {preview}" if preview else "")
            self.comboModel.addItem(text, e)

        self.comboModel.setCurrentIndex(0)
        self._on_model_selected(0)

    def _refresh_current_model_info(self):
        """只显示当前 cls_model_path + 是否具备 classes.json + 与当前标准是否一致。"""
        try:
            with open(os.path.join(_REPO_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        model_path = str((cfg or {}).get("cls_model_path", "") or "").strip()
        if not model_path:
            self.lblModelStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelStatus.setText("未配置分类模型（请到“类别配置”选择可用模型）")
            return
        cj = os.path.join(os.path.dirname(model_path), "classes.json")
        if not os.path.exists(cj):
            self.lblModelStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelStatus.setText(f"当前模型：{model_path}（缺少classes.json，无法保证不错位）")
            return
        # 粗略兼容提示（不阻断）
        try:
            import json

            with open(cj, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
            if isinstance(obj, dict) and "id_to_name" in obj and isinstance(obj["id_to_name"], dict):
                obj = obj["id_to_name"]
            model_classes = []
            if isinstance(obj, dict):
                for kk in sorted(obj.keys(), key=lambda x: int(str(x))):
                    model_classes.append(str(obj.get(kk, "")).strip())
        except Exception:
            model_classes = []
        rpt_names = _rptcfg_class_names(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
        ok, _remap, _diff = _cls_compat_and_remap(model_classes=model_classes, rptcfg_classes=rpt_names) if model_classes else (False, [], {})
        if ok:
            self.lblModelStatus.setStyleSheet("color:#2e7d32; font-weight:bold;")
            self.lblModelStatus.setText(f"当前模型：{model_path}（可用）")
        else:
            self.lblModelStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelStatus.setText(f"当前模型：{model_path}（与当前标准不一致，请到“类别配置”更换）")

    def _on_model_selected(self, _=None):
        e = self.comboModel.currentData()
        self._selected_model_entry = e
        self._selected_model_remap = []

        if e is None:
            self.btnModelEnable.setEnabled(False)
            self.btnModelView.setEnabled(False)
            self.lblModelStatus.setStyleSheet("color:#555;")
            self.lblModelStatus.setText("未选择模型。")
            return

        rpt_names = _rptcfg_class_names(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))

        if not getattr(e, "model_path", "") or not os.path.exists(str(e.model_path)):
            self.btnModelEnable.setEnabled(False)
            self.btnModelView.setEnabled(bool(e.classes))
            self.lblModelStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelStatus.setText("该条目缺少模型权重文件（pth），无法启用。")
            return

        if not getattr(e, "classes", None):
            self.btnModelEnable.setEnabled(False)
            self.btnModelView.setEnabled(False)
            self.lblModelStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelStatus.setText("该模型缺少类别清单（classes.json），为避免错位，不允许启用。")
            return

        ok, remap, diff = _cls_compat_and_remap(model_classes=list(e.classes or []), rptcfg_classes=list(rpt_names or []))
        self._selected_model_remap = list(remap or [])
        self.btnModelView.setEnabled(True)
        self.btnModelEnable.setEnabled(bool(ok and remap))
        if ok:
            self.lblModelStatus.setStyleSheet("color:#2e7d32; font-weight:bold;")
            same_order = remap == list(range(1, len(remap) + 1))
            self.lblModelStatus.setText("可用（顺序一致）" if same_order else "可用（已自动对齐顺序）")
        else:
            self.lblModelStatus.setStyleSheet("color:#c62828; font-weight:bold;")
            miss = diff.get("missing") or []
            extra = diff.get("extra") or []
            msg = "不可用：类别不一致"
            if miss:
                msg += "；缺少：" + "、".join(miss[:6]) + ("…" if len(miss) > 6 else "")
            if extra:
                msg += "；多出：" + "、".join(extra[:6]) + ("…" if len(extra) > 6 else "")
            self.lblModelStatus.setText(msg)

    def _show_selected_model_classes(self):
        e = getattr(self, "_selected_model_entry", None)
        if e is None:
            QMessageBox.information(self, "模型类别", "未选择模型。")
            return
        if not getattr(e, "classes", None):
            # 工人常见困惑：只有一个模型但“点了没反应”。这里给明确指引。
            QMessageBox.information(
                self,
                "模型类别不可查看",
                "该模型缺少类别清单（classes.json）。\n\n"
                "解决办法：\n"
                "1) 进入「缺陷分类向导」按当前类别重新训练（推荐）；或\n"
                "2) 把训练产物目录里的 classes.json 复制到该模型 .pth 同目录。\n\n"
                "为避免分类错位，系统不会允许启用没有类别清单的模型。",
            )
            return
        lines = [f"{i}. {n}" for i, n in enumerate(list(e.classes or []), start=1)]
        QMessageBox.information(self, "模型对应缺陷类型", "\n".join(lines) if lines else "（空）")

    def _enable_selected_model(self):
        e = getattr(self, "_selected_model_entry", None)
        if e is None:
            return
        if not getattr(e, "model_path", "") or not os.path.exists(str(e.model_path)):
            QMessageBox.warning(self, "无法启用", "模型文件不存在。")
            return
        if not getattr(e, "classes", None):
            QMessageBox.warning(self, "无法启用", "缺少 classes.json（类别清单），为避免错位不允许启用。")
            return

        rpt_names = _rptcfg_class_names(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
        ok, remap, diff = _cls_compat_and_remap(model_classes=list(e.classes or []), rptcfg_classes=list(rpt_names or []))
        if not ok:
            QMessageBox.warning(
                self,
                "模型不可用",
                "该模型的类别与当前标准不一致，不能启用。\n\n"
                f"缺少：{', '.join(diff.get('missing') or []) or '-'}\n"
                f"多出：{', '.join(diff.get('extra') or []) or '-'}\n\n"
                "建议：进入「缺陷分类向导」按当前类别重新训练。",
            )
            return

        cfg_path = os.path.join(_REPO_ROOT, "config", "config.yaml")
        cfg = {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        cfg["cls_model_path"] = str(e.model_path)
        try:
            tmp = cfg_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True)
            os.replace(tmp, cfg_path)
        except Exception as ex:
            QMessageBox.warning(self, "写入失败", f"无法写入 config.yaml：\n{ex}")
            return

        try:
            runtime_state_path = os.path.join(_REPO_ROOT, "config", "runtime_state.json")
            _write_runtime_remap(
                runtime_state_path,
                model_path=str(e.model_path),
                model_classes=list(e.classes or []),
                rptcfg_classes=list(rpt_names or []),
                remap_modelidx_to_rptid_1based=list(remap or []),
            )
        except Exception:
            pass

        QMessageBox.information(self, "启用成功", "已启用所选分类模型（并写入自动对齐映射）。")
        self._refresh_model_registry()

    def standard_show_click(self):
        product_cls = self.product_cls.text()
        # 获取行列数
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # 获取 class_labels 的最大 key 值
        class_labels = data.get('class_labels', {})


        row_headers = list(class_labels.values())
        self.row = max(map(int, class_labels.keys()), default=0)

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        anomaly_area_cls_range = config.get('anomaly_area_cls_range', [])
        last_end = anomaly_area_cls_range[-1][1]  # 获取最后一个范围的结束值
        column_headers = [f"[{start}, {end}]" for start, end in anomaly_area_cls_range] + [f"> {last_end}"]
        self.col = len(anomaly_area_cls_range) + 1

        rows = self.row
        cols = self.col

        # 设置表格行列数
        self.tableWidget_standard.setRowCount(rows)
        self.tableWidget_standard.setColumnCount(cols)

        # 设置行标题和列标题
        self.tableWidget_standard.setHorizontalHeaderLabels(column_headers)
        self.tableWidget_standard.setVerticalHeaderLabels(row_headers)

        # 初始化表格内容
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            config2 = yaml.safe_load(file)
        data_key = f"data{product_cls}"
        data = config2.get(data_key)
        for i in range(rows):
            for j in range(cols):
                # 如果数据列表中有值，则使用该值；否则默认为 1000
                value = data[i][j] if i < len(data) and j < len(data[i]) else 1000
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_standard.setItem(i, j, item)

        # 调整列宽
        for col in range(cols):
            self.tableWidget_standard.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(rows):
            self.tableWidget_standard.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def create_table(self):
        # 获取行列数
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # 获取 class_labels 的最大 key 值
        class_labels = data.get('class_labels', {})
        product_cls = data.get("product_cls")


        row_headers = list(class_labels.values())
        self.row = max(map(int, class_labels.keys()), default=0)

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        anomaly_area_cls_range = config.get('anomaly_area_cls_range', [])
        last_end = anomaly_area_cls_range[-1][1]  # 获取最后一个范围的结束值
        column_headers = [f"[{start}, {end}]" for start, end in anomaly_area_cls_range] + [f"> {last_end}"]
        self.col = len(anomaly_area_cls_range)+1

        rows = self.row
        cols = self.col

        # 设置表格行列数
        self.tableWidget_standard.setRowCount(rows)
        self.tableWidget_standard.setColumnCount(cols)

        # 设置行标题和列标题
        self.tableWidget_standard.setHorizontalHeaderLabels(column_headers)
        self.tableWidget_standard.setVerticalHeaderLabels(row_headers)
        _apply_matrix_corner_labels(self.tableWidget_standard, tr_text="面积", bl_text="缺陷")

        # 初始化表格内容
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            config2 = yaml.safe_load(file)
        data_key = f"data{product_cls}"
        data = config2.get(data_key)
        for i in range(rows):
            for j in range(cols):
                # 如果数据列表中有值，则使用该值；否则默认为 1000
                value = data[i][j] if i < len(data) and j < len(data[i]) else 1000
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_standard.setItem(i, j, item)

        # 调整列宽
        for col in range(cols):
            self.tableWidget_standard.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(rows):
            self.tableWidget_standard.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def export_data(self):
        product_cls = self.product_cls.text()
        self.config_data['product_cls'] = product_cls
        rows = self.tableWidget_standard.rowCount()
        cols = self.tableWidget_standard.columnCount()

        # 获取表格数据
        data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                item = self.tableWidget_standard.item(i, j)
                if item and item.text().strip().isdigit():
                    row_data.append(int(item.text().strip()))
                else:
                    row_data.append(0)  # 默认值为 0
            data.append(row_data)

        self.config_data[f"data{product_cls}"] = data
        self.justsaveone('product_cls', self.product_cls.text())
        self.justsaveone(f"data{product_cls}", data)
        self.python_process = subprocess.Popen([_PYTHON_EXE, "-u", _MAKE_STD_SCRIPT], cwd=os.path.join(_REPO_ROOT))
        print(data)

    def _open_cls_config(self):
        """弹出类别配置窗口，关闭后刷新型号展示。"""
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
        """类别配置关闭后刷新当前型号展示。"""
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            self.product_cls.setText(str(cfg.get('product_cls', '')))
            self.create_table()
        except Exception as e:
            print(f'刷新型号展示失败: {e}')

    def initUI1(self):
        self.scrollArea01_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(4, 6, 14, 6)
        self.scroll_layout.setSpacing(10)

        # 初始化四个输入框
        # 从 YAML 文件加载数据并初始化输入框
        self.load_class_labels(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))



        # 设置滚动区域
        self.scrollArea01_widget.setLayout(self.scroll_layout)
        self.scrollArea01.setWidget(self.scrollArea01_widget)
        self.scrollArea01.setWidgetResizable(True)

        # 添加按钮
        self.pushButton_add.clicked.connect(self.add_input_field)

        # 保存按钮
        self.cls_save.clicked.connect(self.save_data)
        # 回滚按钮（仅创建 + 连接信号；位置由 _polish_report_spacing 统一设置到 frame_2）
        if not hasattr(self, "btnRollback"):
            self.btnRollback = QPushButton("撤销上次标准修改")
            self.btnRollback.setStyleSheet(
                "QPushButton{color:#c62828;font-weight:bold;border:1px solid #ef9a9a;"
                "border-radius:4px;padding:4px 8px;background:#fff8f8;}"
                "QPushButton:hover{background:#ffebee;}"
            )
            self.btnRollback.clicked.connect(self.rollback_last_standard_change)

    def _open_wizard_from_report(self):
        if not hasattr(self, "_wizard_window") or self._wizard_window is None:
            self._wizard_window = ClsWizardWindow(self)
        self._wizard_window.show()
        self._wizard_window.raise_()
        self._wizard_window.activateWindow()


    def load_class_labels(self, yaml_file):
        """从 YAML 文件加载 class_labels 并初始化输入框"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                class_labels = data.get("class_labels", {})

                for key, value in class_labels.items():
                    self.add_input_field(key, value)
        except FileNotFoundError:
            print(f"YAML 文件 {yaml_file} 未找到！")
        except yaml.YAMLError as e:
            print(f"解析 YAML 文件时出错: {e}")

    def add_input_field(self, class_id=None, label_text=""):
        """动态添加输入框"""
        if class_id is False:  # 如果未指定 class_id，自动分配一个
            class_id = self.class_counter

        field_layout = QHBoxLayout()
        field_layout.setSpacing(10)
        field_layout.setContentsMargins(0, 2, 0, 2)
        label = QLabel(f"类别 {class_id}:")
        label.setMinimumWidth(76)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        input_field = QLineEdit()
        input_field.setPlaceholderText("请输入类别名称")
        input_field.setText(label_text)  # 初始化输入框内容
        input_field.setMinimumHeight(28)
        remove_button = QPushButton("删除")
        remove_button.setMinimumWidth(52)
        remove_button.setStyleSheet(
            "QPushButton{padding:4px 10px;min-width:48px;}"
        )
        remove_button.clicked.connect(lambda: self.remove_input_field(field_layout))
        try:
            input_field.textChanged.connect(lambda *_: setattr(self, "_has_unsaved_edits", True))
        except Exception:
            pass

        field_layout.addWidget(label, 0)
        field_layout.addWidget(input_field, 1)
        field_layout.addWidget(remove_button, 0)

        self.scroll_layout.addLayout(field_layout)

        # 将输入框和布局记录到 self.input_fields 中
        self.input_fields.append((field_layout, input_field))

        self.class_counter += 1

    def add_input_field01(self, class_id=None, label_text=""):
        """动态添加输入框"""
        if class_id is False:  # 如果未指定 class_id，自动分配一个

            class_id = self.class_counter

        field_layout = QHBoxLayout()
        field_layout.setSpacing(10)
        field_layout.setContentsMargins(0, 2, 0, 2)
        label = QLabel(f"类别 {class_id}:")
        label.setMinimumWidth(76)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        input_field = QLineEdit()
        input_field.setPlaceholderText("请输入类别名称")
        input_field.setText(label_text)  # 初始化输入框内容
        input_field.setMinimumHeight(28)
        remove_button = QPushButton("删除")
        remove_button.setMinimumWidth(52)
        remove_button.setStyleSheet(
            "QPushButton{padding:4px 10px;min-width:48px;}"
        )
        remove_button.clicked.connect(lambda: self.remove_input_field(field_layout))

        field_layout.addWidget(label, 0)
        field_layout.addWidget(input_field, 1)
        field_layout.addWidget(remove_button, 0)

        self.scroll_layout.addLayout(field_layout)
        self.class_counter += 1

    def renumber_fields(self):
        """重新计算并刷新所有类别编号"""
        for index, (layout, input_field) in enumerate(self.input_fields):
            # 新的编号是 列表索引 + 1
            new_id = index + 1

            # 获取 layout 中的第一个控件（即 QLabel "类别 x:"）
            # 注意：itemAt(0) 是 label, itemAt(1) 是 input, itemAt(2) 是 button
            label_item = layout.itemAt(0)
            if label_item:
                label_widget = label_item.widget()
                if isinstance(label_widget, QLabel):
                    label_widget.setText(f"类别 {new_id}:")

        # 更新计数器，确保下一次“添加”时，编号是接着当前的最后一位
        self.class_counter = len(self.input_fields) + 1

    def remove_input_field(self, layout_to_remove):
        """删除指定的输入框"""
        for i, (layout, input_field) in enumerate(self.input_fields):
            if layout == layout_to_remove:
                # 1. 从列表中移除记录
                self.input_fields.pop(i)

                # 2. 从界面 UI 中移除控件
                self.clear_layout(layout_to_remove)

                # 3. 【关键步骤】触发重排，更新剩余的编号
                self.renumber_fields()

                # 找到并删除后即可退出循环
                return

    def save_yaml1(self):
        """保存数据到 YAML 文件"""
        class_labels = {}
        for i in range(self.scroll_layout.count()):
            layout = self.scroll_layout.itemAt(i)
            if isinstance(layout, QHBoxLayout):
                label_widget = layout.itemAt(0).widget()  # 获取标签
                input_widget = layout.itemAt(1).widget()  # 获取输入框

                if isinstance(label_widget, QLabel) and isinstance(input_widget, QLineEdit):
                    class_id = int(label_widget.text().split(" ")[1])  # 提取 class_id
                    label_text = input_widget.text()  # 获取用户输入的类别名称

                    class_labels[class_id] = label_text

    @staticmethod
    def clear_layout(layout):
        """清除布局中的所有小部件"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def save_data(self):
        """将用户输入的数据保存为字典格式"""
        self.class_labels = {}
        for idx, (_, input_field) in enumerate(self.input_fields, start=1):
            text = input_field.text().strip()
            if text:  # 忽略空值
                self.class_labels[idx] = text

        if not self.class_labels:
            QMessageBox.warning(self, "警告", "没有输入有效的类别名称！")
            return
        self.config_data["class_labels"] = self.class_labels
        self.class_list = list(self.class_labels.keys())
        self.config_data["class_list"] = self.class_list
        print(self.class_list)
        # 两段式提交：先提示影响范围（全局）
        ok = QMessageBox.question(
            self,
            "影响提示",
            "你将修改【全局缺陷类型标准】。\n\n"
            "- 会影响后续所有报告的统计与允收判定\n"
            "- 系统会自动备份，可回滚\n\n"
            "确认保存吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return

        # 保存到全局 rptcfg.yaml（带 revision/备份）
        self.justsaveone("class_labels", self.class_labels)
        self.justsaveone("class_list", self.class_list)
        self._has_unsaved_edits = False

        self.initUI2()
        self.initUI3()

    def justsaveone(self, name, data, *, skip_auto_report=False):
        try:
            # 冲突保护：若外部 revision 已变化且本窗口也有未保存编辑，禁止覆盖
            meta = _rpt_read_meta(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
            if meta.revision > int(getattr(self, "_rptcfg_revision_at_load", 0) or 0) and getattr(self, "_has_unsaved_edits", False):
                QMessageBox.warning(self, "冲突", "标准已在其他界面更新，请先刷新后再保存。")
                return

            _rpt_update(
                os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"),
                {name: data},
                updated_by=getattr(self, "_updated_by", None),
            )
            self._rptcfg_revision_at_load = _rpt_read_meta(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml")).revision
            self._refresh_sync_label()
            # 保存标准后：自动触发重生成本次报告（修改模式）；细节优化「保存」显式跳过
            if not skip_auto_report:
                try:
                    cfg = _rpt_read(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))
                    trigger = bool(cfg.get("update_info", False)) and (
                        name in ("class_labels", "class_list", "print_cls", "colors")
                        or str(name).startswith("data")
                    )
                    if trigger:
                        self.print_report()
                except Exception:
                    pass

            #QMessageBox.information(self, "保存成功", "数据已成功保存到 rptcfg.yaml 文件！")
            print(name, data)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存数据时出错:\n{e}")
            print(f"保存数据时出错: {e}")

    def rollback_last_standard_change(self):
        backups = _rpt_list_backups(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), limit=3)
        if not backups:
            QMessageBox.information(self, "无备份", "未找到标准备份文件，无法回滚。")
            return
        # 小白模式：只提供回滚到最近一次
        target = backups[0]
        ok = QMessageBox.question(
            self,
            "确认回滚",
            f"确认回滚到最近一次标准备份？\n\n{os.path.basename(target)}\n\n回滚会影响后续所有报告。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        try:
            _rpt_rollback(
                os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"),
                target,
                updated_by=getattr(self, "_updated_by", None),
            )
            self._rptcfg_revision_at_load = _rpt_read_meta(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml")).revision
            self._has_unsaved_edits = False
            self.initUI2()
            self.initUI3()
            QMessageBox.information(self, "已回滚", "已回滚到最近一次标准备份。")
        except Exception as e:
            QMessageBox.warning(self, "回滚失败", f"回滚失败：{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReportWindow()
    window.show()
    sys.exit(app.exec_())