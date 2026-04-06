from __future__ import annotations

import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import time
from typing import Dict, List, Optional

import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rptcfg_store import read_yaml as _rpt_read, update_keys as _rpt_update
from cls_train_window import ClsTrainWindow
from cls_model_registry import compat_and_remap as _cls_compat_and_remap, rptcfg_class_names as _rptcfg_class_names


_PROJECT_ROOT = os.path.join(_REPO_ROOT)
_RPTCFG_PATH = os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml")
_RUNTIME_STATE_PATH = os.path.join(_PROJECT_ROOT, "config", "runtime_state.json")


def _default_templates() -> Dict[str, List[str]]:
    return {
        "3类（常用）": ["划痕", "擦伤", "破洞"],
        "5类（常用）": ["划痕", "擦伤", "破洞", "油污", "褶皱"],
        "8类（扩展）": ["划痕", "擦伤", "破洞", "油污", "褶皱", "压伤", "氧化皮", "异物"],
    }


def _to_class_labels(names: List[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for i, n in enumerate(names, start=1):
        t = str(n or "").strip()
        if not t:
            t = f"类别{i}"
        out[i] = t
    return out


class ClsWizardWindow(QtWidgets.QMainWindow):
    """
    工人小白模式：3 步向导
    - Step1 选择模板/当前标准
    - Step2 只改“缺陷类型名字”
    - Step3 创建训练集文件夹结构（训练/启用由下一 todo 接入）
    """

    def __init__(self, parent=None, *, is_detect_running_fn=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("缺陷分类向导（推荐）")
        self.resize(860, 620)

        self._class_names: List[str] = []
        self._dataset_root: str = ""
        self._is_detect_running_fn = is_detect_running_fn
        self._train_window: Optional[ClsTrainWindow] = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.lblTitle = QLabel("缺陷分类向导")
        self.lblTitle.setStyleSheet("font-size:18px;font-weight:bold;color:#1a237e;")
        root.addWidget(self.lblTitle)

        self.lblHint = QLabel("按提示一步一步操作即可。不会改坏系统：每次保存都会自动备份，可随时回滚。")
        self.lblHint.setStyleSheet("color:#555;")
        self.lblHint.setWordWrap(True)
        root.addWidget(self.lblHint)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # bottom nav
        nav = QHBoxLayout()
        root.addLayout(nav)
        self.btnBack = QPushButton("上一步")
        self.btnNext = QPushButton("下一步")
        self.btnBack.clicked.connect(self._back)
        self.btnNext.clicked.connect(self._next)
        nav.addWidget(self.btnBack)
        nav.addStretch(1)
        nav.addWidget(self.btnNext)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self.stack.setCurrentIndex(0)
        self._update_nav()

    # ---------------- step1 ----------------
    def _build_step1(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("第 1 步：选择缺陷类型方案")
        g = QVBoxLayout(grp)

        self.grpChoices = QButtonGroup(self)
        self.grpChoices.setExclusive(True)

        # current standard option
        cfg = _rpt_read(_RPTCFG_PATH)
        cur_labels = cfg.get("class_labels") or {}
        cur_names = []
        try:
            for k in sorted(cur_labels, key=lambda x: int(x)):
                cur_names.append(str(cur_labels[k]))
        except Exception:
            cur_names = [str(v) for v in cur_labels.values()]
        updated_at = str(cfg.get("rptcfg_updated_at", "") or "")
        rb_cur = QRadioButton(f"当前标准（推荐）：{len(cur_names)} 类  更新时间：{updated_at or '-'}")
        rb_cur.setProperty("mode", "current")
        rb_cur.setChecked(True)
        self.grpChoices.addButton(rb_cur)
        g.addWidget(rb_cur)

        # templates
        for title, names in _default_templates().items():
            rb = QRadioButton(f"{title}：{len(names)} 类")
            rb.setProperty("mode", "template")
            rb.setProperty("template_title", title)
            self.grpChoices.addButton(rb)
            g.addWidget(rb)

        tip = QLabel("说明：如果只改名字/允收标准，不训练也能生效；如果类别数量变化，建议训练后启用模型。")
        tip.setStyleSheet("color:#666;font-size:10px;")
        tip.setWordWrap(True)
        g.addWidget(tip)

        lay.addWidget(grp)
        lay.addStretch(1)
        self.stack.addWidget(w)

    def _step1_selected_names(self) -> List[str]:
        btn = self.grpChoices.checkedButton()
        if btn is None:
            return []
        mode = btn.property("mode")
        if mode == "current":
            cfg = _rpt_read(_RPTCFG_PATH)
            cur_labels = cfg.get("class_labels") or {}
            out = []
            try:
                for k in sorted(cur_labels, key=lambda x: int(x)):
                    out.append(str(cur_labels[k]))
            except Exception:
                out = [str(v) for v in cur_labels.values()]
            return [x.strip() for x in out if str(x).strip()]
        if mode == "template":
            title = str(btn.property("template_title") or "")
            return list(_default_templates().get(title, []))
        return []

    # ---------------- step2 ----------------
    def _build_step2(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("第 2 步：设置缺陷类型名称（只改名字）")
        g = QVBoxLayout(grp)

        self.listNames = QListWidget()
        g.addWidget(self.listNames, 1)

        row = QHBoxLayout()
        self.edtNew = QLineEdit()
        self.edtNew.setPlaceholderText("输入一个新的缺陷类型名称，例如：油污")
        self.btnAdd = QPushButton("添加")
        self.btnDel = QPushButton("删除选中")
        self.btnSaveGlobal = QPushButton("保存为全局标准（会影响后续所有报告）")
        self.btnSaveGlobal.setStyleSheet("font-weight:bold;")
        row.addWidget(self.edtNew, 1)
        row.addWidget(self.btnAdd)
        row.addWidget(self.btnDel)
        g.addLayout(row)
        g.addWidget(self.btnSaveGlobal)

        self.btnAdd.clicked.connect(self._add_name)
        self.btnDel.clicked.connect(self._del_name)
        self.btnSaveGlobal.clicked.connect(self._save_global_labels)

        lay.addWidget(grp, 1)
        self.stack.addWidget(w)

    def _load_step2_from_selected(self):
        self.listNames.clear()
        names = self._step1_selected_names()
        if not names:
            names = ["划痕", "擦伤", "破洞"]
        for n in names:
            self.listNames.addItem(QListWidgetItem(str(n)))
        self._class_names = names

    def _add_name(self):
        t = (self.edtNew.text() or "").strip()
        if not t:
            return
        self.listNames.addItem(QListWidgetItem(t))
        self.edtNew.clear()

    def _del_name(self):
        row = self.listNames.currentRow()
        if row < 0:
            return
        name = self.listNames.item(row).text()
        ok = QMessageBox.question(
            self,
            "确认删除",
            f"确认删除缺陷类型：{name}？\n\n删除会影响统计与允收判定。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ok != QMessageBox.Yes:
            return
        self.listNames.takeItem(row)

    def _save_global_labels(self):
        names: List[str] = []
        for i in range(self.listNames.count()):
            t = self.listNames.item(i).text().strip()
            if t:
                names.append(t)
        if len(names) < 2:
            QMessageBox.warning(self, "类别过少", "至少需要 2 个缺陷类型。")
            return

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

        class_labels = _to_class_labels(names)
        class_list = list(class_labels.keys())
        # print_cls：默认全选
        patch = {"class_labels": class_labels, "class_list": class_list, "print_cls": class_list}
        _rpt_update(_RPTCFG_PATH, patch)
        QMessageBox.information(self, "已保存", "已保存为全局缺陷类型标准。")
        self._class_names = names

    # ---------------- step3 ----------------
    def _build_step3(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("第 3 步：准备训练图片（可选）")
        g = QVBoxLayout(grp)

        self.lblDataset = QLabel("训练集目录：未选择")
        self.lblDataset.setStyleSheet("color:#333;")
        self.lblDataset.setWordWrap(True)
        g.addWidget(self.lblDataset)

        # 当前模型与当前标准是否匹配（工人不需要懂原理，只看“能用/不能用”）
        self.lblModelCompat = QLabel("")
        self.lblModelCompat.setWordWrap(True)
        self.lblModelCompat.setStyleSheet("color:#555;")
        g.addWidget(self.lblModelCompat)

        row = QHBoxLayout()
        self.btnChoose = QPushButton("选择训练集目录")
        self.btnCreate = QPushButton("创建训练文件夹结构")
        self.btnOpen = QPushButton("打开训练文件夹")
        row.addWidget(self.btnChoose)
        row.addWidget(self.btnCreate)
        row.addWidget(self.btnOpen)
        row.addStretch(1)
        g.addLayout(row)

        self.lblCheck = QLabel(
            "数据集要求（请务必按以下方式准备，否则训练会失败或效果很差）：\n"
            "1) 目录结构：训练集目录下必须有多个“类别文件夹”，文件夹名=缺陷类型名称。\n"
            "2) 图片格式：建议 jpg/png（其它格式可能无法读取）。\n"
            "3) 每类数量：建议每类 ≥ 30 张；太少会导致模型不稳定。\n"
            "4) 图片内容：尽量是缺陷小图/缺陷截图，同一类尽量风格一致。\n"
            "5) 不要把不同类别混放到同一文件夹。\n\n"
            "操作方法：点「创建训练文件夹结构」→ 打开文件夹 → 把图片拖到对应类别目录。\n"
        )
        self.lblCheck.setStyleSheet("color:#666;font-size:10px;")
        self.lblCheck.setWordWrap(True)
        g.addWidget(self.lblCheck)

        self.btnTrainNow = QPushButton("开始训练并自动启用（推荐）")
        self.btnTrainNow.setStyleSheet("font-weight:bold;")
        self.btnTrainNow2 = QPushButton("仅更新标准（不训练）")
        g.addWidget(self.btnTrainNow)
        g.addWidget(self.btnTrainNow2)

        lay.addWidget(grp)
        lay.addStretch(1)
        self.stack.addWidget(w)

        self.btnChoose.clicked.connect(self._choose_dataset_root)
        self.btnCreate.clicked.connect(self._create_dataset_structure)
        self.btnOpen.clicked.connect(self._open_dataset_root)
        self.btnTrainNow.clicked.connect(self._start_train_flow)
        self.btnTrainNow2.clicked.connect(lambda: QMessageBox.information(self, "已完成", "已更新全局标准（未训练）。"))
        self._refresh_model_compat_status()

    def _refresh_model_compat_status(self):
        # 读取当前启用模型与当前标准类别，给出“可用/不可用”提示
        cfg_path = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
        model_path = ""
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_path = str((cfg or {}).get("cls_model_path", "") or "").strip()
        except Exception:
            model_path = ""

        rpt_names = _rptcfg_class_names(_RPTCFG_PATH)
        if not model_path:
            self.lblModelCompat.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelCompat.setText("当前未启用分类模型：建议训练后再启用。")
            return
        if not os.path.exists(model_path):
            self.lblModelCompat.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelCompat.setText("当前分类模型文件不存在：请重新启用或训练。")
            return

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
            self.lblModelCompat.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelCompat.setText("当前模型缺少类别清单（classes.json）：为避免分类错位，建议重新训练。")
            return

        ok, remap, diff = _cls_compat_and_remap(model_classes=model_classes, rptcfg_classes=rpt_names)
        if ok:
            same_order = remap == list(range(1, len(remap) + 1))
            self.lblModelCompat.setStyleSheet("color:#2e7d32; font-weight:bold;")
            self.lblModelCompat.setText("当前模型可用（顺序一致）。" if same_order else "当前模型可用（已自动对齐顺序）。")
        else:
            miss = diff.get("missing") or []
            extra = diff.get("extra") or []
            msg = "当前模型不可用：类别不一致。"
            if miss:
                msg += " 缺少：" + "、".join(miss[:6]) + ("…" if len(miss) > 6 else "")
            if extra:
                msg += " 多出：" + "、".join(extra[:6]) + ("…" if len(extra) > 6 else "")
            self.lblModelCompat.setStyleSheet("color:#c62828; font-weight:bold;")
            self.lblModelCompat.setText(msg + "（建议按当前类别重新训练并启用）")

    def _choose_dataset_root(self):
        d = QFileDialog.getExistingDirectory(self, "选择训练集目录", _PROJECT_ROOT)
        if d:
            self._dataset_root = d
            self.lblDataset.setText(f"训练集目录：{d}")

    def _create_dataset_structure(self):
        if not self._dataset_root:
            QMessageBox.information(self, "提示", "请先选择训练集目录。")
            return
        names = self._class_names or self._step1_selected_names()
        if not names:
            QMessageBox.warning(self, "提示", "未找到缺陷类型，请先完成第 2 步保存。")
            return
        try:
            for n in names:
                os.makedirs(os.path.join(self._dataset_root, str(n)), exist_ok=True)
            QMessageBox.information(self, "已创建", "已创建训练文件夹结构。请把图片拖进去即可。")
        except Exception as e:
            QMessageBox.warning(self, "创建失败", f"创建失败：{e}")

    def _open_dataset_root(self):
        if not self._dataset_root:
            QMessageBox.information(self, "提示", "请先选择训练集目录。")
            return
        try:
            os.startfile(self._dataset_root)  # type: ignore[attr-defined]
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"打开失败：{e}")

    def _is_detect_running(self) -> bool:
        try:
            if callable(self._is_detect_running_fn):
                return bool(self._is_detect_running_fn())
        except Exception:
            pass
        # fallback：runtime_state.json 里只有 paused，不可靠；默认认为不在运行
        return False

    def _start_train_flow(self):
        # 互斥保护
        if self._is_detect_running():
            QMessageBox.warning(self, "无法训练", "检测正在运行。为避免影响产线，请先停止检测后再训练。")
            return
        if not self._dataset_root:
            QMessageBox.information(self, "提示", "请先选择训练集目录。")
            return

        # 训练集强校验：每类目录存在 + 每类最少样本数
        names = self._class_names or self._step1_selected_names()
        if not names:
            QMessageBox.information(self, "提示", "请先在第 2 步保存缺陷类型名称。")
            return
        min_per_class = 30
        missing_dirs = []
        low_counts = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for n in names:
            p = os.path.join(self._dataset_root, str(n))
            if not os.path.isdir(p):
                missing_dirs.append(n)
                continue
            cnt = 0
            try:
                for fn in os.listdir(p):
                    if os.path.splitext(fn)[1].lower() in exts:
                        cnt += 1
            except Exception:
                cnt = 0
            if cnt < min_per_class:
                low_counts.append(f"{n}({cnt}张)")

        if missing_dirs:
            QMessageBox.warning(
                self,
                "训练集不完整",
                "以下类别文件夹不存在：\n"
                + "\n".join(f"- {x}" for x in missing_dirs)
                + "\n\n请点击「创建训练文件夹结构」，并把图片放入对应文件夹。",
            )
            return
        if low_counts:
            ok = QMessageBox.question(
                self,
                "样本较少",
                "以下类别样本数偏少，训练效果可能不稳定：\n"
                + "\n".join(f"- {x}" for x in low_counts)
                + f"\n\n建议每类 ≥ {min_per_class} 张。仍要继续训练吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return

        # 打开训练窗口并预填目录（工人不需要理解参数）
        if self._train_window is None or not self._train_window.isVisible():
            self._train_window = ClsTrainWindow(self, is_detect_running_fn=self._is_detect_running_fn)
        self._train_window.show()
        self._train_window.raise_()
        self._train_window.activateWindow()
        try:
            self._train_window.edtDataset.setText(self._dataset_root)
            self._train_window._scan_dataset()
            self._train_window._log("来自向导：已自动扫描训练集，请直接点击「开始训练」。")
        except Exception:
            pass

    # ---------------- nav ----------------
    def _back(self):
        ix = self.stack.currentIndex()
        if ix <= 0:
            return
        self.stack.setCurrentIndex(ix - 1)
        self._update_nav()

    def _next(self):
        ix = self.stack.currentIndex()
        if ix == 0:
            self._load_step2_from_selected()
        if ix >= self.stack.count() - 1:
            self.close()
            return
        self.stack.setCurrentIndex(ix + 1)
        # 进入第3步时刷新“当前模型是否匹配”提示
        try:
            if self.stack.currentIndex() == 2:
                self._refresh_model_compat_status()
        except Exception:
            pass
        self._update_nav()

    def _update_nav(self):
        ix = self.stack.currentIndex()
        self.btnBack.setEnabled(ix > 0)
        self.btnNext.setText("完成" if ix == self.stack.count() - 1 else "下一步")

