import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
"""
类别配置窗口：产品型号管理 + 缺陷类别命名 + 允收矩阵维护

单一读写层：所有修改均通过本窗口写入 rptcfg.yaml，
并可触发 make_standard.py 生成 table.json 供报告链使用。
"""

import subprocess
import yaml
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QSplitter, QWidget, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys

_RPTCFG_PATH = os.path.join(_REPO_ROOT, "config", "rptcfg.yaml")
_CONFIG_PATH  = os.path.join(_REPO_ROOT, "config", "config.yaml")
_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)
_MAKE_STD     = os.path.join(_REPO_ROOT, "F_mainui", "F_mainui", "make_standard.py")


# ─────────────────────── rptcfg 读写工具 ───────────────────────────────────

def _read_rptcfg() -> dict:
    try:
        with open(_RPTCFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_rptcfg(cfg: dict) -> None:
    with open(_RPTCFG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)


def _rptcfg_set(key: str, value) -> None:
    cfg = _read_rptcfg()
    cfg[key] = value
    _write_rptcfg(cfg)


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
        self.resize(960, 660)
        self.setModal(False)
        self._col_headers: list = []
        self._cls_input_fields: list = []  # [(QHBoxLayout, QLineEdit)]
        self._build_ui()
        self._load_all()

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

        # 缺陷类别名称
        grp_cls = QGroupBox("缺陷类别名称 (class_labels)")
        vc = QVBoxLayout(grp_cls)
        vc.setSpacing(4)
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
        btn_add_cls.clicked.connect(lambda: self._add_class_field())
        h_cls_btn.addWidget(btn_add_cls)
        btn_save_cls = QPushButton("保存类别名称")
        btn_save_cls.setStyleSheet("font-weight: bold;")
        btn_save_cls.clicked.connect(self._save_class_labels)
        h_cls_btn.addWidget(btn_save_cls)
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
        self.lbl_type_hint = QLabel("请在左侧选择产品型号")
        self.lbl_type_hint.setStyleSheet("color: #5c6bc0; font-size: 11px; font-weight: bold;")
        vm.addWidget(self.lbl_type_hint)
        self.lbl_mat_desc = QLabel(
            "行 = 缺陷类别，列 = 面积区间，单元格 = 该类别在该区间内允许出现的最大缺陷数。"
        )
        self.lbl_mat_desc.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_mat_desc.setWordWrap(True)
        vm.addWidget(self.lbl_mat_desc)
        self.matrix_table = QTableWidget()
        self.matrix_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vm.addWidget(self.matrix_table)
        h_mat_btn = QHBoxLayout()
        btn_save_mat = QPushButton("保存允收矩阵")
        btn_save_mat.setStyleSheet("font-weight: bold;")
        btn_save_mat.clicked.connect(self._save_matrix)
        h_mat_btn.addWidget(btn_save_mat)
        btn_gen = QPushButton("生成 table.json")
        btn_gen.setToolTip(
            "将当前允收矩阵编译为 table.json（供报告链 gen_report_cls.py 使用）。\n"
            "会先保存矩阵再运行 make_standard.py。"
        )
        btn_gen.clicked.connect(self._generate_table)
        h_mat_btn.addWidget(btn_gen)
        vm.addLayout(h_mat_btn)
        right_lay.addWidget(grp_mat)
        splitter.addWidget(right)
        splitter.setSizes([300, 640])

        # ── 底部状态条 ────────────────────────────────────────────────────
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #2e7d32; font-size: 10px; padding: 2px 4px;")
        root.addWidget(self.lbl_status)

    # ── 数据加载 ─────────────────────────────────────────────────────────────

    def _load_all(self):
        cfg = _read_rptcfg()
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
        # 从 config.yaml 读面积区间作为列标题
        self._reload_col_headers()

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

    def _add_class_field(self, class_id=None, label_text=""):
        h = QHBoxLayout()
        idx = len(self._cls_input_fields) + 1
        if class_id is None:
            class_id = idx
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
        cls_labels = {}
        for idx, (_, edt) in enumerate(self._cls_input_fields, 1):
            t = edt.text().strip()
            if t:
                cls_labels[idx] = t
        if not cls_labels:
            QMessageBox.warning(self, "警告", "请至少输入一个缺陷类别名称。")
            return
        _rptcfg_set("class_labels", cls_labels)
        _rptcfg_set("class_list", list(cls_labels.keys()))
        self._set_status(f"已保存 {len(cls_labels)} 个缺陷类别。")
        # 刷新矩阵行标题
        cur_item = self.type_list.currentItem()
        if cur_item:
            k = str(cur_item.data(Qt.UserRole) or cur_item.text())
            self._on_type_selected(k)

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
        _write_rptcfg(cfg)
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
        class_labels = cfg.get("class_labels", {})
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
        for r in range(n_rows):
            for c in range(n_cols):
                val = data[r][c] if r < len(data) and c < len(data[r]) else 1000
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                self.matrix_table.setItem(r, c, it)
        for c in range(n_cols):
            self.matrix_table.setColumnWidth(c, 82)
        for r in range(n_rows):
            self.matrix_table.setRowHeight(r, 36)

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
        _rptcfg_set(f"data{key}", data)
        _rptcfg_set("product_cls", key)
        self._set_status(f"型号 {key} 允收矩阵已保存，rptcfg.product_cls 同步为 {key}。")
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
