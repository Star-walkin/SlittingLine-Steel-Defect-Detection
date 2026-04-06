"""
类别配置窗口：产品型号管理 + 缺陷类别命名 + 允收矩阵维护

单一读写层：所有修改均通过本窗口写入 rptcfg.yaml，
并可触发 make_standard.py 生成 table.json 供报告链使用。
"""

import subprocess
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import yaml
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QSplitter, QWidget, QScrollArea, QSizePolicy, QHeaderView,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPainter, QPen
from PyQt5 import QtGui

from rptcfg_store import (
    read_yaml as _rpt_read,
    write_with_revision as _rpt_write_with_rev,
    update_keys as _rpt_update,
    read_meta as _rpt_read_meta,
)
from cls_wizard import ClsWizardWindow
from cls_model_registry import (
    compat_and_remap as _cls_compat_and_remap,
    rptcfg_class_names as _rptcfg_class_names,
    scan_model_registry as _scan_model_registry,
    write_runtime_remap as _write_runtime_remap,
)

_RPTCFG_PATH = os.path.join(_REPO_ROOT, "config", "rptcfg.yaml")
_CONFIG_PATH  = os.path.join(_REPO_ROOT, "config", "config.yaml")
_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)
_MAKE_STD     = os.path.join(_REPO_ROOT, "F_mainui", "F_mainui", "make_standard.py")


# ── 角标（表格左上角空白格）：对角线 + “缺陷/面积” ─────────────────────────────


class _DiagonalCornerWidget(QWidget):
    def __init__(self, table: QTableWidget, *, tr_text: str = "面积", bl_text: str = "缺陷"):
        super().__init__(table)
        self._table = table
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
        p.fillRect(self.rect(), self._table.palette().base())
        pen = QPen(self._table.palette().mid().color())
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(0, 0, self.width() - 1, self.height() - 1)
        font = QFont(self._table.font())
        font.setPointSize(max(8, int(font.pointSize() * 0.9)))
        p.setFont(font)
        p.setPen(self._table.palette().text().color())
        pad = 4
        p.drawText(pad, pad, self.width() - pad, self.height() - pad, Qt.AlignRight | Qt.AlignTop, self._tr)
        p.drawText(pad, pad, self.width() - pad, self.height() - pad, Qt.AlignLeft | Qt.AlignBottom, self._bl)
        p.end()


def _apply_matrix_corner_labels(table: QTableWidget, *, tr_text: str = "面积", bl_text: str = "缺陷") -> None:
    if table is None:
        return
    try:
        try:
            table.setCornerButtonEnabled(False)
        except Exception:
            pass
        # 方案A：cornerWidget
        try:
            w = _DiagonalCornerWidget(table, tr_text=tr_text, bl_text=bl_text)
            table.setCornerWidget(w)
            w.setAutoFillBackground(True)
            w.show()
            w.raise_()
        except Exception:
            w = None

        # 方案B：覆盖层 overlay（强制显示）
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
            from PyQt5.QtCore import QTimer as _QT
            _QT.singleShot(0, _sync_any)
        except Exception:
            pass
    except Exception:
        pass


# ─────────────────────── rptcfg 读写工具 ───────────────────────────────────

def _read_rptcfg() -> dict:
    return _rpt_read(_RPTCFG_PATH)


def _write_rptcfg(cfg: dict) -> None:
    _rpt_write_with_rev(_RPTCFG_PATH, cfg)


def _rptcfg_set(key: str, value) -> None:
    _rpt_update(_RPTCFG_PATH, {key: value})


def available_product_types(cfg: dict = None) -> list:
    """返回 rptcfg 中已有的 data{N} 型号键列表（仅数字部分），按数值排序。"""
    if cfg is None:
        cfg = _read_rptcfg()
    return sorted(
        (k[4:] for k in cfg if k.startswith("data") and k[4:].isdigit()),
        key=int,
    )


def product_cls_display_label(cfg: dict, key: str) -> str:
    """
    主界面/列表展示用：有命名则「名称 [编号]」，否则「型号 编号」。
    实际存储与 data{N}、config0.product_cls 仍为纯数字编号字符串。
    """
    key = str(key).strip()
    if not key:
        return ""
    names = cfg.get("product_cls_names") or {}
    raw = names.get(key)
    if raw is None and key.isdigit():
        raw = names.get(str(int(key)))
    name = (str(raw).strip() if raw is not None else "")
    if name:
        return f"{name}  [{key}]"
    return f"型号 {key}"


def product_cls_key_from_combo_text(text: str) -> str:
    """从下拉展示文案或纯数字中解析出型号编号（供可编辑 QComboBox 兜底）。"""
    import re

    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"\[(\d+)\]\s*$", t)
    if m:
        return m.group(1)
    if t.isdigit():
        return t
    return t


def product_combo_entries() -> List[Tuple[str, str]]:
    """[(编号, 展示文案), ...] 供主界面 QComboBox 使用。"""
    cfg = _read_rptcfg()
    return [(k, product_cls_display_label(cfg, k)) for k in available_product_types(cfg)]


# ─────────────────────── 主窗口 ────────────────────────────────────────────

class ClsConfigWindow(QDialog):
    """产品型号、缺陷类别、允收矩阵的统一配置窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别配置 — 产品型号与允收标准")
        # 初始窗口更大，减少首次打开的遮挡；同时给最小尺寸，避免缩到看不清
        self.resize(1380, 780)
        self.setMinimumSize(QSize(1100, 720))
        self.setModal(False)
        self._col_headers: list = []
        self._cls_input_fields: list = []  # [(QHBoxLayout, QLineEdit)]
        self._rptcfg_revision_at_load = 0
        self._has_unsaved_edits = False
        self._wizard_window = None
        self._build_ui()
        self._load_all()
        # 同步监控：避免与报告修改窗口/其他入口互相覆盖
        from PyQt5.QtCore import QTimer
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(2000)
        self._sync_timer.timeout.connect(self._poll_rptcfg_revision)
        self._sync_timer.start()

    # ── UI 构建 ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── 左侧：型号列表 + 缺陷类别编辑 ────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_lay.setSpacing(8)

        # 型号管理：列表展示「显示名称 [编号]」，内部键仍为 data{N} 的 N
        grp_type = QGroupBox("产品型号（编号 + 显示名称）")
        vt = QVBoxLayout(grp_type)
        tip_types = QLabel(
            "列表中选型号后编辑矩阵；「显示名称」仅便于识别，报告与 data 键仍用编号。"
        )
        tip_types.setStyleSheet("color: #666; font-size: 10px;")
        tip_types.setWordWrap(True)
        vt.addWidget(tip_types)
        self.type_list = QListWidget()
        self.type_list.setFixedHeight(120)
        self.type_list.currentItemChanged.connect(self._on_type_list_current_changed)
        vt.addWidget(self.type_list)
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("显示名称："))
        self.type_name_edit = QLineEdit()
        self.type_name_edit.setPlaceholderText("如：冷轧基板-内贸1号线（可选）")
        self.type_name_edit.setEnabled(False)
        h_name.addWidget(self.type_name_edit, 1)
        btn_save_name = QPushButton("保存名称")
        btn_save_name.setToolTip("将当前选中型号的显示名称写入 rptcfg（不改变编号与 data 键）")
        btn_save_name.clicked.connect(self._save_type_display_name)
        h_name.addWidget(btn_save_name)
        vt.addLayout(h_name)
        h_new = QHBoxLayout()
        h_new.addWidget(QLabel("新编号"))
        self.new_type_edit = QLineEdit()
        self.new_type_edit.setFixedWidth(44)
        self.new_type_edit.setPlaceholderText("4")
        h_new.addWidget(self.new_type_edit)
        h_new.addWidget(QLabel("名称"))
        self.new_type_name_edit = QLineEdit()
        self.new_type_name_edit.setPlaceholderText("添加时可选填")
        h_new.addWidget(self.new_type_name_edit, 1)
        btn_add_type = QPushButton("添加型号")
        btn_add_type.setToolTip("新建 data{编号} 及默认允收矩阵；可同时填写显示名称。")
        btn_add_type.clicked.connect(self._add_type)
        h_new.addWidget(btn_add_type)
        btn_del_type = QPushButton("删除型号")
        btn_del_type.setStyleSheet("color: #c62828;")
        btn_del_type.setToolTip("删除该编号对应的 data 与显示名称（需确认）")
        btn_del_type.clicked.connect(self._del_type)
        h_new.addWidget(btn_del_type)
        vt.addLayout(h_new)
        left_lay.addWidget(grp_type)

        # 缺陷类别名称（按型号分别保存，避免“改一个影响全部型号”的误解）
        grp_cls = QGroupBox("缺陷类别名称（按当前选中型号）")
        vc = QVBoxLayout(grp_cls)
        vc.setSpacing(4)
        tip_cls = QLabel(
            "说明：这里的“类别名称”按产品型号分别保存（切换型号可分别命名）。\n"
            "编号仍与分类模型输出一一对应：训练有几类就只能映射几类。\n"
            "若你增加类别编号超过模型输出类数，超过部分在报告分类结果中不会出现（需更换/重训模型）。"
        )
        tip_cls.setStyleSheet("color: #666; font-size: 10px;")
        tip_cls.setWordWrap(True)
        vc.addWidget(tip_cls)
        self._cls_scroll_area = QScrollArea()
        self._cls_scroll_area.setWidgetResizable(True)
        self._cls_scroll_content = QWidget()
        self._cls_scroll_layout = QVBoxLayout(self._cls_scroll_content)
        self._cls_scroll_layout.setAlignment(Qt.AlignTop)
        self._cls_scroll_layout.setSpacing(4)
        self._cls_scroll_area.setWidget(self._cls_scroll_content)
        vc.addWidget(self._cls_scroll_area)
        h_cls_btn = QHBoxLayout()
        btn_add_cls = QPushButton("+ 添加类别")
        btn_add_cls.setToolTip(
            "仅扩展配置与允收表占位；分类模型仍只输出训练时的 K 个类号。\n"
            "新编号若无模型映射，报告里该类别统计恒为空。"
        )
        btn_add_cls.clicked.connect(lambda: self._add_class_field())
        h_cls_btn.addWidget(btn_add_cls)
        btn_save_cls = QPushButton("保存到当前型号")
        btn_save_cls.setStyleSheet("font-weight: bold;")
        btn_save_cls.clicked.connect(self._save_class_labels)
        h_cls_btn.addWidget(btn_save_cls)
        btn_save_cls_global = QPushButton("保存为全局标准")
        btn_save_cls_global.setToolTip("将当前编辑的类别名称保存到全局（会影响所有型号）。")
        btn_save_cls_global.clicked.connect(self._save_class_labels_global)
        h_cls_btn.addWidget(btn_save_cls_global)
        vc.addLayout(h_cls_btn)
        left_lay.addWidget(grp_cls)
        left.setMinimumWidth(280)
        splitter.addWidget(left)

        # ── 右侧：允收矩阵 ────────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_lay.setSpacing(8)
        grp_mat = QGroupBox("允收矩阵（按当前选中型号）")
        vm = QVBoxLayout(grp_mat)
        # 顶部栏：提示 + 右上角两个关键按钮（按你要求缩到右上角）
        top = QHBoxLayout()
        self.lbl_type_hint = QLabel("请在左侧选择产品型号")
        self.lbl_type_hint.setStyleSheet("color: #5c6bc0; font-size: 11px; font-weight: bold;")
        top.addWidget(self.lbl_type_hint, 1)
        top.addStretch(1)
        self.btn_gen_table = QPushButton("生成 table.json")
        self.btn_gen_table.setToolTip(
            "将当前允收矩阵编译为 table.json（供报告链 gen_report_cls.py 使用）。\n"
            "会先保存矩阵再运行 make_standard.py。"
        )
        self.btn_gen_table.clicked.connect(self._generate_table)
        self.btn_save_mat = QPushButton("保存允收矩阵")
        self.btn_save_mat.setStyleSheet("font-weight: bold;")
        self.btn_save_mat.clicked.connect(self._save_matrix)
        # 右上角：先保存后生成
        top.addWidget(self.btn_save_mat)
        top.addWidget(self.btn_gen_table)
        vm.addLayout(top)
        self.lbl_mat_desc = QLabel(
            "行 = 缺陷类别，列 = 面积区间，单元格 = 该类别在该区间内允许出现的最大缺陷数。"
        )
        self.lbl_mat_desc.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_mat_desc.setWordWrap(True)
        vm.addWidget(self.lbl_mat_desc)
        self.matrix_table = QTableWidget()
        self.matrix_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 表头/行标题内边距与最小宽度，避免中文贴边或被裁切（不依赖窗口拉伸）
        self.matrix_table.setStyleSheet(
            "QTableWidget{gridline-color:#c8c8c8;}"
            "QHeaderView::section{padding:6px 10px; min-height:28px;}"
        )
        vh = self.matrix_table.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setMinimumWidth(120)
        vh.setDefaultAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hh = self.matrix_table.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignCenter)
        hh.setMinimumSectionSize(72)
        vm.addWidget(self.matrix_table)
        right_lay.addWidget(grp_mat)

        # ── 右下角：分类模型选择（按当前选中型号）──────────────────────────
        grp_model = QGroupBox("分类模型（按当前选中型号选择）")
        gm = QVBoxLayout(grp_model)
        tip_m = QLabel(
            "说明：一个型号可绑定一个分类模型。\n"
            "- 你可以看到所有模型及其对应缺陷类别\n"
            "- 只有“模型类别=当前缺陷类别标准”时，才允许选择（避免错位）"
        )
        tip_m.setStyleSheet("color:#666;font-size:10px;")
        tip_m.setWordWrap(True)
        gm.addWidget(tip_m)

        self.lbl_model_bind = QLabel("当前型号绑定：未设置")
        self.lbl_model_bind.setStyleSheet("color:#1a237e;font-weight:bold;")
        self.lbl_model_bind.setWordWrap(True)
        gm.addWidget(self.lbl_model_bind)

        self.tbl_models = QTableWidget()
        self.tbl_models.setColumnCount(4)
        self.tbl_models.setHorizontalHeaderLabels(["模型", "类别预览", "状态", "模型路径"])
        self.tbl_models.horizontalHeader().setStretchLastSection(True)
        self.tbl_models.verticalHeader().setVisible(False)
        self.tbl_models.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_models.setSelectionMode(QTableWidget.SingleSelection)
        gm.addWidget(self.tbl_models, 1)

        h_mbtn = QHBoxLayout()
        self.btn_model_refresh = QPushButton("刷新模型清单")
        self.btn_model_view = QPushButton("查看该模型类别")
        self.btn_model_apply = QPushButton("设为当前型号使用")
        self.btn_model_apply.setStyleSheet("font-weight:bold;")
        h_mbtn.addWidget(self.btn_model_refresh)
        h_mbtn.addWidget(self.btn_model_view)
        h_mbtn.addStretch(1)
        h_mbtn.addWidget(self.btn_model_apply)
        gm.addLayout(h_mbtn)

        self.btn_model_refresh.clicked.connect(self._refresh_models_table)
        self.btn_model_view.clicked.connect(self._view_selected_model_classes)
        self.btn_model_apply.clicked.connect(self._apply_selected_model_to_type)
        try:
            self.tbl_models.itemSelectionChanged.connect(self._on_models_selection_changed)
        except Exception:
            pass

        right_lay.addWidget(grp_model, stretch=0)
        splitter.addWidget(right)
        splitter.setSizes([300, 640])

        # ── 底部状态条 ────────────────────────────────────────────────────
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #2e7d32; font-size: 10px; padding: 2px 4px;")
        root.addWidget(self.lbl_status)

        self.lbl_sync = QLabel("")
        self.lbl_sync.setStyleSheet("color: #555; font-size: 10px; padding: 2px 4px;")
        root.addWidget(self.lbl_sync)

        # 小白入口：放在底部状态条附近，不与主体区域抢空间
        h_entry = QHBoxLayout()
        h_entry.addStretch(1)
        self.btn_wizard = QPushButton("进入缺陷分类向导")
        self.btn_wizard.setStyleSheet(
            "QPushButton{background:#FFF3E0;color:#E65100;border:1px solid #FFCC80;"
            "border-radius:4px;padding:6px 12px;font-weight:bold;}"
            "QPushButton:hover{background:#FFE0B2;}"
        )
        self.btn_wizard.setToolTip("推荐：工人小白模式，一步一步完成缺陷类型配置与训练准备。")
        self.btn_wizard.clicked.connect(self._open_wizard)
        h_entry.addWidget(self.btn_wizard)
        root.addLayout(h_entry)

    def _open_wizard(self):
        if self._wizard_window is None or not self._wizard_window.isVisible():
            self._wizard_window = ClsWizardWindow(self.parent() if hasattr(self, "parent") else None)
        self._wizard_window.show()
        self._wizard_window.raise_()
        self._wizard_window.activateWindow()

    # ── 数据加载 ─────────────────────────────────────────────────────────────

    def _load_all(self):
        cfg = _read_rptcfg()
        try:
            self._rptcfg_revision_at_load = int(cfg.get("rptcfg_revision", 0) or 0)
        except Exception:
            self._rptcfg_revision_at_load = 0
        self._has_unsaved_edits = False
        self._refresh_sync_label(cfg)
        # 分类模型类数（用于防止“新增类别”超出模型输出，造成现场误解）
        self._cls_model_num_classes = self._try_get_cls_model_num_classes()
        # 先从 config.yaml 读面积区间作为列标题（避免首次选中型号时矩阵列数为 0）
        self._reload_col_headers()
        # 型号列表（展示名 + UserRole 存编号）
        self.type_list.blockSignals(True)
        self.type_list.clear()
        for key in available_product_types(cfg):
            it = QListWidgetItem(product_cls_display_label(cfg, key))
            it.setData(Qt.UserRole, key)
            self.type_list.addItem(it)
        self.type_list.blockSignals(False)
        if self.type_list.count():
            self.type_list.setCurrentRow(0)
        # 缺陷类别
        self._populate_class_labels(cfg.get("class_labels", {}))
        # 模型清单（右下角）
        try:
            self._refresh_models_table()
        except Exception:
            pass
        # 兜底：若首次 setCurrentRow 时信号未触发或列标题刚加载，强制刷新一次矩阵
        cur = self.type_list.currentItem()
        if cur:
            self._on_type_selected(self._type_key_from_item(cur))

    # ── 分类模型选择（右下角）──────────────────────────────────────────────

    def _type_key_current(self) -> str:
        it = self.type_list.currentItem()
        return self._type_key_from_item(it) if it else ""

    def _get_cls_model_by_product(self, cfg: dict) -> dict:
        m = cfg.get("cls_model_by_product")
        return dict(m) if isinstance(m, dict) else {}

    def _refresh_models_table(self):
        cfg = _read_rptcfg()
        type_key = self._type_key_current()
        binds = self._get_cls_model_by_product(cfg)
        bind = binds.get(str(type_key), {}) if type_key else {}
        bind_path = str((bind or {}).get("model_path", "") or "").strip()
        self.lbl_model_bind.setText(f"当前型号绑定：{bind_path or '未设置'}")

        rpt_names = _rptcfg_class_names(_RPTCFG_PATH)
        # 扫描所有模型
        entries = _scan_model_registry(
            project_root=os.path.join(_REPO_ROOT),
            config_path=_CONFIG_PATH,
            rptcfg_path=_RPTCFG_PATH,
            keep_cache=True,
        )

        self._model_entries = list(entries or [])
        self.tbl_models.setRowCount(0)
        for i, e in enumerate(self._model_entries):
            self.tbl_models.insertRow(i)
            model_title = f"{e.trained_at} | {e.num_classes}类 | {e.source}"
            it0 = QTableWidgetItem(model_title)
            it0.setFlags(it0.flags() & ~Qt.ItemIsEditable)
            self.tbl_models.setItem(i, 0, it0)

            preview = "、".join((e.classes or [])[:6]) + ("…" if len(e.classes or []) > 6 else "")
            it1 = QTableWidgetItem(preview)
            it1.setFlags(it1.flags() & ~Qt.ItemIsEditable)
            self.tbl_models.setItem(i, 1, it1)

            status = "未知"
            ok = False
            diff = {}
            if not e.model_path or not os.path.exists(str(e.model_path)):
                status = "缺少权重"
            elif not e.classes:
                status = "缺少classes.json"
            else:
                ok, _remap, diff = _cls_compat_and_remap(model_classes=list(e.classes), rptcfg_classes=list(rpt_names))
                status = "可用" if ok else "类别不一致"
            it2 = QTableWidgetItem(status)
            it2.setFlags(it2.flags() & ~Qt.ItemIsEditable)
            if ok:
                it2.setForeground(QtGui.QBrush(QtGui.QColor("#2e7d32")))
            else:
                it2.setForeground(QtGui.QBrush(QtGui.QColor("#c62828")))
            it2.setData(Qt.UserRole, {"ok": ok, "diff": diff})
            self.tbl_models.setItem(i, 2, it2)

            it3 = QTableWidgetItem(str(e.model_path or ""))
            it3.setFlags(it3.flags() & ~Qt.ItemIsEditable)
            self.tbl_models.setItem(i, 3, it3)

        self.tbl_models.resizeColumnsToContents()
        self._on_models_selection_changed()

        # 若当前绑定存在，尝试选中对应行
        if bind_path:
            for i, e in enumerate(self._model_entries):
                if os.path.normpath(str(e.model_path or "")) == os.path.normpath(bind_path):
                    self.tbl_models.selectRow(i)
                    break

    def _selected_model_entry(self):
        r = self.tbl_models.currentRow()
        if r < 0:
            return None
        try:
            return self._model_entries[r]
        except Exception:
            return None

    def _on_models_selection_changed(self):
        e = self._selected_model_entry()
        if e is None:
            self.btn_model_apply.setEnabled(False)
            self.btn_model_view.setEnabled(False)
            return
        self.btn_model_view.setEnabled(bool(e.classes))
        # 只有兼容才允许选择
        rpt_names = _rptcfg_class_names(_RPTCFG_PATH)
        ok, remap, _diff = (False, [], {})
        if e.model_path and os.path.exists(str(e.model_path)) and e.classes:
            ok, remap, _diff = _cls_compat_and_remap(model_classes=list(e.classes), rptcfg_classes=list(rpt_names))
        self.btn_model_apply.setEnabled(bool(ok and remap))

    def _view_selected_model_classes(self):
        e = self._selected_model_entry()
        if e is None:
            return
        if not e.classes:
            QMessageBox.information(self, "模型类别不可查看", "该模型缺少 classes.json（类别清单）。")
            return
        txt = "\n".join([f"{i}. {n}" for i, n in enumerate(list(e.classes), start=1)])
        QMessageBox.information(self, "模型对应缺陷类型", txt)

    def _apply_selected_model_to_type(self):
        type_key = self._type_key_current()
        if not type_key:
            QMessageBox.information(self, "提示", "请先选择一个产品型号。")
            return
        e = self._selected_model_entry()
        if e is None:
            QMessageBox.information(self, "提示", "请先在下方列表选择一个模型。")
            return
        rpt_names = _rptcfg_class_names(_RPTCFG_PATH)
        ok, remap, diff = _cls_compat_and_remap(model_classes=list(e.classes or []), rptcfg_classes=list(rpt_names))
        if not ok:
            QMessageBox.warning(
                self,
                "模型不可用",
                "该模型的类别与当前缺陷类别标准不一致，不能绑定到该型号。\n\n"
                f"缺少：{', '.join(diff.get('missing') or []) or '-'}\n"
                f"多出：{', '.join(diff.get('extra') or []) or '-'}",
            )
            return

        cfg = _read_rptcfg()
        binds = self._get_cls_model_by_product(cfg)
        binds[str(type_key)] = {
            "model_path": str(e.model_path or ""),
            "trained_at": str(e.trained_at or ""),
            "source": str(e.source or ""),
            "remap_modelidx_to_rptid_1based": list(remap or []),
        }
        cfg["cls_model_by_product"] = binds
        _write_rptcfg(cfg)

        # 同步写 runtime_state remap（让报告生成/分类推理立即可用）
        try:
            _write_runtime_remap(
                os.path.join(os.path.dirname(_RPTCFG_PATH), "runtime_state.json"),
                model_path=str(e.model_path or ""),
                model_classes=list(e.classes or []),
                rptcfg_classes=list(rpt_names or []),
                remap_modelidx_to_rptid_1based=list(remap or []),
            )
        except Exception:
            pass

        self._set_status(f"已将型号 {type_key} 绑定到分类模型：{e.model_path}")
        self._refresh_models_table()

    def _refresh_sync_label(self, cfg: dict = None):
        if cfg is None:
            cfg = _read_rptcfg()
        try:
            rev = int(cfg.get("rptcfg_revision", 0) or 0)
        except Exception:
            rev = 0
        ts = str(cfg.get("rptcfg_updated_at", "") or "")
        self.lbl_sync.setText(f"标准同步：revision={rev}  更新时间={ts or '-'}")

    def _poll_rptcfg_revision(self):
        meta = _rpt_read_meta(_RPTCFG_PATH)
        if meta.revision <= int(getattr(self, "_rptcfg_revision_at_load", 0) or 0):
            return
        # 外部更新发生
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
        self._load_all()

    def _reload_col_headers(self):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            ranges = c.get("anomaly_area_cls_range", [])
            self._col_headers = [f"[{a},{b}]" for a, b in ranges]
            if ranges:
                self._col_headers.append(f">{ranges[-1][1]}")
        except Exception:
            self._col_headers = []

    # ── 缺陷类别编辑 ─────────────────────────────────────────────────────────

    def _populate_class_labels(self, cls_dict: dict):
        for lyt, _ in self._cls_input_fields:
            while lyt.count():
                w = lyt.takeAt(0).widget()
                if w:
                    w.deleteLater()
        self._cls_input_fields.clear()
        for k, v in sorted(cls_dict.items(), key=lambda x: int(x[0])):
            self._add_class_field(class_id=k, label_text=str(v))

    def _effective_class_labels_for_type(self, cfg: dict, key: str) -> dict:
        """按型号取类别名称：优先 class_labels_by_product[key]，否则回退全局 class_labels。"""
        key = str(key or "").strip()
        by = cfg.get("class_labels_by_product")
        if key and isinstance(by, dict):
            v = by.get(key)
            if v is None and key.isdigit():
                v = by.get(str(int(key)))
            if isinstance(v, dict) and v:
                return v
        return cfg.get("class_labels", {}) or {}

    def _add_class_field(self, class_id=None, label_text=""):
        h = QHBoxLayout()
        idx = len(self._cls_input_fields) + 1
        if class_id is None:
            class_id = idx
        # 若模型类数已知，给出“超过模型输出”的即时提示（不强拦截，避免阻断现场临时编辑）
        try:
            n = int(getattr(self, "_cls_model_num_classes", 0) or 0)
            if n > 0 and int(class_id) > n:
                self._set_status(f"提示：已超过分类模型输出类数（模型={n}类），新增类别将不会在报告分类结果中出现。")
        except Exception:
            pass
        lbl = QLabel(f"类别 {class_id}:")
        lbl.setFixedWidth(64)
        edt = QLineEdit()
        edt.setPlaceholderText("类别名称")
        edt.setText(str(label_text))
        btn_rm = QPushButton("\u00d7")  # 乘号 ×，作删除示意
        btn_rm.setFixedWidth(32)
        btn_rm.setToolTip("删除此类别")
        btn_rm.setStyleSheet(
            "QPushButton { color: #c62828; font-weight: bold; font-size: 18px;"
            " padding: 2px 4px; border: 1px solid #e0e0e0; border-radius: 6px; background: #ffffff; }"
            "QPushButton:hover { background: #ffebee; border-color: #ef9a9a; color: #b71c1c; }"
        )
        btn_rm.clicked.connect(lambda: self._remove_class_field(h))
        edt.textChanged.connect(lambda *_: setattr(self, "_has_unsaved_edits", True))
        h.addWidget(lbl)
        h.addWidget(edt)
        h.addWidget(btn_rm)
        self._cls_scroll_layout.addLayout(h)
        self._cls_input_fields.append((h, edt))

    def _remove_class_field(self, target_layout):
        for i, (lyt, _) in enumerate(self._cls_input_fields):
            if lyt is target_layout:
                self._cls_input_fields.pop(i)
                while lyt.count():
                    w = lyt.takeAt(0).widget()
                    if w:
                        w.deleteLater()
                self._cls_scroll_layout.removeItem(lyt)
                break
        self._renumber_class_fields()

    def _renumber_class_fields(self):
        for idx, (lyt, _) in enumerate(self._cls_input_fields, 1):
            item = lyt.itemAt(0)
            if item and item.widget():
                item.widget().setText(f"类别 {idx}:")

    def _save_class_labels(self):
        # 保存到“当前型号”
        item = self.type_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选择型号。")
            return
        cur_key = self._type_key_from_item(item)
        cls_labels = {}
        for idx, (_, edt) in enumerate(self._cls_input_fields, 1):
            t = edt.text().strip()
            if t:
                cls_labels[idx] = t
        if not cls_labels:
            QMessageBox.warning(self, "警告", "请至少输入一个缺陷类别名称。")
            return
        # 若超过模型类数：明确弹窗确认，防止“以为模型会同步多一类”的误操作
        try:
            n = int(getattr(self, "_cls_model_num_classes", 0) or 0)
        except Exception:
            n = 0
        if n > 0 and len(cls_labels) > n:
            ok = QMessageBox.question(
                self,
                "类别数超过模型",
                f"当前保存 {len(cls_labels)} 个类别，但分类模型输出为 {n} 类。\n"
                f"超过部分不会在报告分类结果中出现（除非更换/重训分类模型）。\n\n仍要保存吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return
        # 写入前：若外部 revision 已变化且本窗口也有编辑，阻止覆盖
        meta = _rpt_read_meta(_RPTCFG_PATH)
        if meta.revision > self._rptcfg_revision_at_load and self._has_unsaved_edits:
            QMessageBox.warning(self, "冲突", "标准已在其他界面更新，请先刷新后再保存。")
            return
        cfg = _read_rptcfg()
        by = cfg.get("class_labels_by_product")
        by = dict(by) if isinstance(by, dict) else {}
        by[str(cur_key)] = cls_labels
        _rpt_update(_RPTCFG_PATH, {"class_labels_by_product": by, "class_list": list(cls_labels.keys())})
        self._set_status(f"已保存型号 {cur_key} 的 {len(cls_labels)} 个缺陷类别名称。")
        self._has_unsaved_edits = False
        self._rptcfg_revision_at_load = _rpt_read_meta(_RPTCFG_PATH).revision
        self._refresh_sync_label()
        # 刷新矩阵行标题（本型号）
        self._on_type_selected(str(cur_key))

    def _save_class_labels_global(self):
        """保存为全局标准（会影响所有型号）。"""
        cls_labels = {}
        for idx, (_, edt) in enumerate(self._cls_input_fields, 1):
            t = edt.text().strip()
            if t:
                cls_labels[idx] = t
        if not cls_labels:
            QMessageBox.warning(self, "警告", "请至少输入一个缺陷类别名称。")
            return
        ok = QMessageBox.question(
            self,
            "影响提示",
            "你将修改【全局缺陷类别名称】。\n"
            "这会影响所有型号的显示与后续报告统计口径。\n\n确认保存为全局标准吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        meta = _rpt_read_meta(_RPTCFG_PATH)
        if meta.revision > self._rptcfg_revision_at_load and self._has_unsaved_edits:
            QMessageBox.warning(self, "冲突", "标准已在其他界面更新，请先刷新后再保存。")
            return
        _rpt_update(_RPTCFG_PATH, {"class_labels": cls_labels, "class_list": list(cls_labels.keys())})
        self._set_status(f"已保存全局 {len(cls_labels)} 个缺陷类别名称。")
        self._has_unsaved_edits = False
        self._rptcfg_revision_at_load = _rpt_read_meta(_RPTCFG_PATH).revision
        self._refresh_sync_label()

    def _try_get_cls_model_num_classes(self) -> int:
        """
        从 config.yaml 的 cls_model_path 读取 checkpoint 的 out.weight.shape[0]。
        失败则返回 0（不阻断 UI）。
        """
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            model_path = str(c.get("cls_model_path", "") or "").strip()
            if not model_path:
                return 0
            if not os.path.exists(model_path):
                return 0
            import torch  # 延迟导入，避免打开窗口时拖慢
            w = torch.load(model_path, map_location="cpu")
            return int(w["out.weight"].shape[0])
        except Exception:
            return 0

    # ── 型号管理 ─────────────────────────────────────────────────────────────

    def _type_key_from_item(self, item: QListWidgetItem) -> str:
        if not item:
            return ""
        k = item.data(Qt.UserRole)
        return str(k).strip() if k is not None else product_cls_key_from_combo_text(item.text())

    def _on_type_list_current_changed(self, current, _previous):
        if not current:
            self.type_name_edit.clear()
            self.type_name_edit.setEnabled(False)
            return
        key = self._type_key_from_item(current)
        self.type_name_edit.setEnabled(True)
        cfg = _read_rptcfg()
        names = cfg.get("product_cls_names") or {}
        self.type_name_edit.setText(str(names.get(key, "") or ""))
        self._on_type_selected(key)
        # 型号切换时刷新右下角“分类模型绑定/可用性”
        try:
            self._refresh_models_table()
        except Exception:
            pass

    def _save_type_display_name(self):
        item = self.type_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选择型号。")
            return
        key = self._type_key_from_item(item)
        name = self.type_name_edit.text().strip()
        cfg = _read_rptcfg()
        names = dict(cfg.get("product_cls_names") or {})
        if name:
            names[str(key)] = name
        else:
            names.pop(str(key), None)
        cfg["product_cls_names"] = names
        # 防覆盖
        meta = _rpt_read_meta(_RPTCFG_PATH)
        if meta.revision > self._rptcfg_revision_at_load and self._has_unsaved_edits:
            QMessageBox.warning(self, "冲突", "标准已在其他界面更新，请先刷新后再保存。")
            return
        _write_rptcfg(cfg)
        self._has_unsaved_edits = False
        self._rptcfg_revision_at_load = _rpt_read_meta(_RPTCFG_PATH).revision
        self._refresh_sync_label()
        item.setText(product_cls_display_label(cfg, key))
        self._set_status(f"已保存型号 {key} 的显示名称。")

    def _add_type(self):
        key = self.new_type_edit.text().strip()
        if not key or not key.isdigit():
            QMessageBox.warning(self, "无效", "型号编号必须为纯数字（如 4）。")
            return
        cfg = _read_rptcfg()
        data_key = f"data{key}"
        disp_name = self.new_type_name_edit.text().strip()
        if data_key in cfg:
            QMessageBox.information(self, "已存在", f"型号 {key} 已存在，可直接在列表中选择编辑。")
        else:
            class_labels = cfg.get("class_labels", {})
            n_rows = max((int(k) for k in class_labels), default=0)
            n_cols = len(self._col_headers)
            cfg[data_key] = [[1000] * n_cols for _ in range(n_rows)]
            if disp_name:
                names = dict(cfg.get("product_cls_names") or {})
                names[str(key)] = disp_name
                cfg["product_cls_names"] = names
            _write_rptcfg(cfg)
            cfg = _read_rptcfg()
            it = QListWidgetItem(product_cls_display_label(cfg, key))
            it.setData(Qt.UserRole, key)
            self.type_list.addItem(it)
            self.type_list.setCurrentItem(it)
            self._set_status(f"已添加型号 {key}（data{key}），允收初值 1000。")
        self.new_type_edit.clear()
        self.new_type_name_edit.clear()

    def _del_type(self):
        item = self.type_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先在列表中选择要删除的型号。")
            return
        key = self._type_key_from_item(item)
        cfg = _read_rptcfg()
        label = product_cls_display_label(cfg, key)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确认删除 {label} ？\n将删除 data{key} 及该型号的显示名称，且不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        cfg.pop(f"data{key}", None)
        names = dict(cfg.get("product_cls_names") or {})
        names.pop(str(key), None)
        cfg["product_cls_names"] = names
        _write_rptcfg(cfg)
        self.type_list.takeItem(self.type_list.row(item))
        self.matrix_table.clearContents()
        self.matrix_table.setRowCount(0)
        self.type_name_edit.clear()
        self.type_name_edit.setEnabled(False)
        self.lbl_type_hint.setText("型号已删除，请重新选择。")
        self._set_status(f"已删除型号 {key}。", ok=False)

    # ── 允收矩阵 ─────────────────────────────────────────────────────────────

    def _on_type_selected(self, key: str):
        if not key:
            return
        cfg = _read_rptcfg()
        self.lbl_type_hint.setText(f"当前：{product_cls_display_label(cfg, key)}")
        # 型号切换：同时切换“该型号的缺陷类别名称”
        class_labels = self._effective_class_labels_for_type(cfg, key)
        try:
            self._populate_class_labels(class_labels)
        except Exception:
            pass
        row_headers = [
            str(class_labels.get(k, k))
            for k in sorted(class_labels, key=int)
        ]
        col_headers = self._col_headers
        data = cfg.get(f"data{key}", [])
        n_rows, n_cols = len(row_headers), len(col_headers)
        self.matrix_table.setRowCount(n_rows)
        self.matrix_table.setColumnCount(n_cols)
        self.matrix_table.setHorizontalHeaderLabels(col_headers)
        self.matrix_table.setVerticalHeaderLabels(row_headers)
        _apply_matrix_corner_labels(self.matrix_table, tr_text="面积", bl_text="缺陷")
        for r in range(n_rows):
            for c in range(n_cols):
                val = data[r][c] if r < len(data) and c < len(data[r]) else 1000
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                self.matrix_table.setItem(r, c, it)
        for c in range(n_cols):
            self.matrix_table.setColumnWidth(c, 82)
        for r in range(n_rows):
            self.matrix_table.setRowHeight(r, 38)
        # 行标题宽度按内容兜底（某些字体/缩放下固定 120 仍可能略紧）
        try:
            self.matrix_table.resizeRowsToContents()
            self.matrix_table.verticalHeader().resizeSections(QHeaderView.ResizeToContents)
            self.matrix_table.verticalHeader().setMinimumWidth(
                max(120, self.matrix_table.verticalHeader().sizeHint().width() + 12)
            )
        except Exception:
            pass

    def _save_matrix(self) -> bool:
        item = self.type_list.currentItem()
        if not item:
            QMessageBox.warning(self, "未选择", "请先在左侧选择产品型号。")
            return False
        key = self._type_key_from_item(item)
        rows = self.matrix_table.rowCount()
        cols = self.matrix_table.columnCount()
        data = []
        for r in range(rows):
            row_d = []
            for c in range(cols):
                cell = self.matrix_table.item(r, c)
                try:
                    row_d.append(int(cell.text()) if cell else 1000)
                except (ValueError, AttributeError):
                    row_d.append(1000)
            data.append(row_d)
        meta = _rpt_read_meta(_RPTCFG_PATH)
        if meta.revision > self._rptcfg_revision_at_load and self._has_unsaved_edits:
            QMessageBox.warning(self, "冲突", "标准已在其他界面更新，请先刷新后再保存。")
            return False
        _rpt_update(_RPTCFG_PATH, {f"data{key}": data, "product_cls": key})
        self._set_status(f"型号 {key} 允收矩阵已保存，rptcfg.product_cls 同步为 {key}。")
        self._has_unsaved_edits = False
        self._rptcfg_revision_at_load = _rpt_read_meta(_RPTCFG_PATH).revision
        self._refresh_sync_label()
        return True

    def _generate_table(self):
        if not self._save_matrix():
            return
        try:
            subprocess.Popen(
                [_PYTHON_EXE, "-u", _MAKE_STD],
                cwd=os.path.join(_REPO_ROOT),
            )
            self._set_status("已触发 make_standard.py，正在生成 table.json，请稍候。")
        except Exception as exc:
            self._set_status(f"生成失败：{exc}", ok=False)

    # ── 辅助 ─────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, ok: bool = True):
        color = "#2e7d32" if ok else "#c62828"
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px; padding: 2px 4px;")
        self.lbl_status.setText(msg)

    def refresh(self):
        """外部可调用：重新从 rptcfg 加载数据（如其他窗口保存后需刷新）。"""
        self._load_all()
