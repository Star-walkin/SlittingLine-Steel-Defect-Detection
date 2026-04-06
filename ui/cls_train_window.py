from __future__ import annotations

import json
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


_PROJECT_ROOT = os.path.join(_REPO_ROOT)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
_RPTCFG_PATH = os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml")
_RUNTIME_STATE_PATH = os.path.join(_PROJECT_ROOT, "config", "runtime_state.json")
_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)

_CLS_TRAIN_SCRIPT = os.path.join(_PROJECT_ROOT, "cls_model", "train_cls_model.py")
_CLS_MODEL_DIR = os.path.join(_PROJECT_ROOT, "cls_model")
_CLS_TRAIN_RESULT_DIR = os.path.join(_CLS_MODEL_DIR, "train-result")


def _read_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_yaml(path: str, cfg: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_int(s: str, default: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return default


def _scan_dataset_by_folders(dataset_root: str) -> Tuple[List[str], Dict[str, int], List[str]]:
    """
    约定：dataset_root/<class_name>/*.jpg|png|bmp|jpeg
    返回：class_names, class_counts, warnings
    """
    warnings: List[str] = []
    dataset_root = str(dataset_root or "").strip()
    if not dataset_root or not os.path.isdir(dataset_root):
        return [], {}, ["训练集目录不存在或不可访问。"]

    class_names: List[str] = []
    counts: Dict[str, int] = {}
    for name in sorted(os.listdir(dataset_root)):
        p = os.path.join(dataset_root, name)
        if not os.path.isdir(p):
            continue
        if name.startswith("."):
            continue
        if name.strip() == "":
            continue
        class_names.append(name)
        n = 0
        for fn in os.listdir(p):
            if not isinstance(fn, str):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                n += 1
        counts[name] = n

    if not class_names:
        warnings.append("未发现任何类别子目录（需要 dataset_root/类别名/图片）。")
    else:
        zeros = [c for c in class_names if counts.get(c, 0) <= 0]
        if zeros:
            warnings.append(f"以下类别没有图片样本：{', '.join(zeros)}")
        if len(class_names) < 2:
            warnings.append("类别数少于 2：分类训练通常至少需要 2 类。")
        # Windows 路径常见误操作：用 data1/data2 当类别名没问题，但用纯数字会给映射/展示带来混淆
        digit_folders = [c for c in class_names if re.fullmatch(r"\d+", c or "")]
        if digit_folders:
            warnings.append(f"注意：以下类别文件夹名为纯数字，建议用中文/英文更易读：{', '.join(digit_folders)}")
    return class_names, counts, warnings


def _default_id_mapping(class_names: List[str]) -> Dict[str, int]:
    # 默认 1..K，按文件夹名排序
    return {name: i + 1 for i, name in enumerate(class_names)}


def _rptcfg_set_class_labels_from_mapping(mapping: Dict[str, int]) -> None:
    """
    用 mapping(类别名->1based_id) 重建 rptcfg.class_labels/class_list/print_cls 的最小一致集。
    """
    rptcfg = _read_yaml(_RPTCFG_PATH)
    # 反向：id->name
    id_to_name: Dict[int, str] = {}
    for name, cid in mapping.items():
        try:
            ic = int(cid)
        except Exception:
            continue
        if ic <= 0:
            continue
        id_to_name[ic] = str(name)
    if not id_to_name:
        return
    class_labels = {int(k): str(v) for k, v in sorted(id_to_name.items(), key=lambda x: int(x[0]))}
    kmax = max(class_labels.keys())
    class_list = list(range(1, kmax + 1))

    # 保留用户原来 print_cls 的交集；若为空则默认全打印
    old_print = rptcfg.get("print_cls")
    old_print_list: List[int] = []
    if isinstance(old_print, list):
        for it in old_print:
            try:
                old_print_list.append(int(it))
            except Exception:
                pass
    keep = [c for c in old_print_list if c in set(class_list)]
    rptcfg["class_labels"] = class_labels
    rptcfg["class_list"] = class_list
    rptcfg["print_cls"] = keep if keep else class_list
    _write_yaml(_RPTCFG_PATH, rptcfg)


def _write_runtime_state_extra(extra: dict) -> None:
    cur = {}
    try:
        if os.path.exists(_RUNTIME_STATE_PATH):
            with open(_RUNTIME_STATE_PATH, "r", encoding="utf-8") as f:
                cur = json.load(f) or {}
    except Exception:
        cur = {}
    cur.update(extra or {})
    _atomic_write_json(_RUNTIME_STATE_PATH, cur)


@dataclass(frozen=True)
class TrainArtifacts:
    out_dir: str
    checkpoint_path: str
    classes_json_path: str
    train_config_path: str
    metrics_path: str


def _make_artifacts_dir() -> TrainArtifacts:
    tag = _now_tag()
    out_dir = os.path.join(_CLS_TRAIN_RESULT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    return TrainArtifacts(
        out_dir=out_dir,
        checkpoint_path=os.path.join(out_dir, "cls_checkpoint.pth"),
        classes_json_path=os.path.join(out_dir, "classes.json"),
        train_config_path=os.path.join(out_dir, "train_config.yaml"),
        metrics_path=os.path.join(out_dir, "metrics.json"),
    )


class ClsTrainWindow(QtWidgets.QMainWindow):
    """
    分类训练/模型管理窗口：
    - 训练集：按文件夹扫描、映射编辑
    - 训练：后台启动训练脚本，显示日志
    - 启用：一键写 config.yaml 的 cls_model_path，并同步 rptcfg.class_labels
    """

    def __init__(self, parent=None, *, is_detect_running_fn=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("分类训练与模型管理")
        self.resize(980, 680)
        self._is_detect_running_fn = is_detect_running_fn

        self._proc: Optional[QProcess] = None
        self._last_artifacts: Optional[TrainArtifacts] = None
        self._dataset_root: str = ""
        self._mapping: Dict[str, int] = {}

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- current model summary
        self.lblCurrent = QLabel("")
        self.lblCurrent.setStyleSheet("color:#1a237e;font-weight:bold;")
        root.addWidget(self.lblCurrent)

        # ---- dataset picker
        row = QHBoxLayout()
        root.addLayout(row)
        row.addWidget(QLabel("训练集目录"))
        self.edtDataset = QLineEdit()
        self.edtDataset.setPlaceholderText(r"例如：D:\dataset_cls\（包含多个类别子目录）")
        row.addWidget(self.edtDataset, 1)
        self.btnBrowse = QPushButton("选择...")
        row.addWidget(self.btnBrowse)
        self.btnScan = QPushButton("扫描训练集")
        row.addWidget(self.btnScan)

        # ---- dataset table
        self.tbl = QTableWidget()
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels(["类别(文件夹名)", "样本数", "类别ID(1-based)"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        root.addWidget(self.tbl, 1)

        # ---- train params + actions
        row2 = QHBoxLayout()
        root.addLayout(row2)
        row2.addWidget(QLabel("Epoch"))
        self.spinEpoch = QSpinBox()
        self.spinEpoch.setRange(1, 200)
        self.spinEpoch.setValue(15)
        row2.addWidget(self.spinEpoch)
        row2.addWidget(QLabel("Batch"))
        self.spinBatch = QSpinBox()
        self.spinBatch.setRange(1, 256)
        self.spinBatch.setValue(32)
        row2.addWidget(self.spinBatch)
        row2.addWidget(QLabel("Img"))
        self.spinImg = QSpinBox()
        self.spinImg.setRange(64, 512)
        self.spinImg.setValue(128)
        row2.addWidget(self.spinImg)
        row2.addStretch(1)

        self.btnTrain = QPushButton("开始训练")
        self.btnStop = QPushButton("停止训练")
        self.btnEnable = QPushButton("启用最新模型")
        self.btnOpenOut = QPushButton("打开产物目录")
        self.btnSyncOnly = QPushButton("仅同步类别到报告配置")
        row2.addWidget(self.btnTrain)
        row2.addWidget(self.btnStop)
        row2.addWidget(self.btnEnable)
        row2.addWidget(self.btnOpenOut)
        row2.addWidget(self.btnSyncOnly)

        # ---- log
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QtGui.QFont("Consolas", 9))
        root.addWidget(self.txt, 2)

        # ---- wiring
        self.btnBrowse.clicked.connect(self._browse_dataset)
        self.btnScan.clicked.connect(self._scan_dataset)
        self.btnTrain.clicked.connect(self._start_train)
        self.btnStop.clicked.connect(self._stop_train)
        self.btnEnable.clicked.connect(self._enable_latest_model)
        self.btnOpenOut.clicked.connect(self._open_out_dir)
        self.btnSyncOnly.clicked.connect(self._sync_labels_only)

        self._refresh_current_model_status()
        self._update_buttons()

    def _log(self, s: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt.append(f"[{ts}] {s}")

    def _is_detect_running(self) -> bool:
        try:
            if callable(self._is_detect_running_fn):
                return bool(self._is_detect_running_fn())
        except Exception:
            pass
        return False

    def _refresh_current_model_status(self) -> None:
        cfg = _read_yaml(_CONFIG_PATH)
        model_path = str(cfg.get("cls_model_path", "") or "").strip()
        rpt = _read_yaml(_RPTCFG_PATH)
        n_labels = len(rpt.get("class_labels") or {})
        exists = "存在" if (model_path and os.path.exists(model_path)) else "缺失"

        k = None
        if model_path and os.path.exists(model_path):
            try:
                import torch  # 延迟导入，避免打开窗口卡顿

                w = torch.load(model_path, map_location="cpu")
                k = int(w["out.weight"].shape[0])
            except Exception:
                k = None
        k_text = f"{k}" if isinstance(k, int) and k > 0 else "未知"
        consistent = ""
        if isinstance(k, int) and k > 0:
            consistent = "一致" if n_labels == k else "不一致"
        self.lblCurrent.setText(
            f"当前分类模型：{model_path or '(未配置)'}（{exists}）  |  "
            f"模型输出类数K：{k_text}  |  "
            f"当前类别数(class_labels)：{n_labels}"
            + (f"（{consistent}）" if consistent else "")
        )

    def _update_buttons(self) -> None:
        running = self._proc is not None and self._proc.state() != QProcess.NotRunning
        self.btnStop.setEnabled(running)
        self.btnTrain.setEnabled(not running)
        self.btnEnable.setEnabled((not running) and (self._last_artifacts is not None))

    def _browse_dataset(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择分类训练集目录", self.edtDataset.text().strip() or _PROJECT_ROOT)
        if d:
            self.edtDataset.setText(d)

    def _scan_dataset(self) -> None:
        dataset_root = self.edtDataset.text().strip()
        class_names, counts, warns = _scan_dataset_by_folders(dataset_root)
        self._dataset_root = dataset_root
        self._mapping = _default_id_mapping(class_names)

        self.tbl.setRowCount(0)
        for i, name in enumerate(class_names):
            self.tbl.insertRow(i)
            it0 = QTableWidgetItem(str(name))
            it0.setFlags(it0.flags() & ~Qt.ItemIsEditable)
            self.tbl.setItem(i, 0, it0)

            it1 = QTableWidgetItem(str(counts.get(name, 0)))
            it1.setTextAlignment(Qt.AlignCenter)
            it1.setFlags(it1.flags() & ~Qt.ItemIsEditable)
            self.tbl.setItem(i, 1, it1)

            it2 = QTableWidgetItem(str(self._mapping.get(name, i + 1)))
            it2.setTextAlignment(Qt.AlignCenter)
            self.tbl.setItem(i, 2, it2)

        self.tbl.resizeColumnsToContents()
        if warns:
            self._log("训练集扫描提示：")
            for w in warns:
                self._log(f"- {w}")
        else:
            self._log(f"训练集扫描完成：{len(class_names)} 类。")

        # 追加一个“强校验”提示：样本数过少/极度不均衡时，提示但不阻断
        try:
            nonzero = [counts.get(n, 0) for n in class_names if counts.get(n, 0) > 0]
            if nonzero:
                mn, mx = min(nonzero), max(nonzero)
                if mn < 10:
                    self._log("提示：存在样本数 < 10 的类别，训练效果可能不稳定。")
                if mx >= 5 * max(1, mn):
                    self._log("提示：类别样本数差异较大（最大/最小>=5），建议补齐小类或做采样策略。")
        except Exception:
            pass

    def _collect_mapping_from_table(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for r in range(self.tbl.rowCount()):
            name = (self.tbl.item(r, 0).text() if self.tbl.item(r, 0) else "").strip()
            cid_txt = (self.tbl.item(r, 2).text() if self.tbl.item(r, 2) else "").strip()
            if not name:
                continue
            cid = _safe_int(cid_txt, 0)
            if cid <= 0:
                continue
            mapping[name] = cid
        return mapping

    def _validate_mapping(self, mapping: Dict[str, int]) -> Optional[str]:
        if not self._dataset_root or not os.path.isdir(self._dataset_root):
            return "请先选择并扫描训练集目录。"
        if not mapping:
            return "未配置任何类别映射。"
        ids = list(mapping.values())
        if len(set(ids)) != len(ids):
            return "类别ID存在重复，请修改为 1..K 且不重复。"
        if min(ids) <= 0:
            return "类别ID必须为正整数（1-based）。"
        # 连续性：建议 1..K 连续，避免“中间缺号”导致报告/允收矩阵理解困难
        ids_sorted = sorted(set(ids))
        if ids_sorted and ids_sorted[0] != 1:
            return "类别ID建议从 1 开始连续编号（1..K）。"
        if ids_sorted and ids_sorted != list(range(1, ids_sorted[-1] + 1)):
            return "类别ID建议连续（1..K），请补齐或重新编号。"
        return None

    def _start_train(self) -> None:
        if self._is_detect_running():
            QMessageBox.warning(self, "无法训练", "当前在线检测正在运行。为避免占用 GPU/CPU，请先停止检测后再训练。")
            return

        mapping = self._collect_mapping_from_table()
        err = self._validate_mapping(mapping)
        if err:
            QMessageBox.warning(self, "配置不完整", err)
            return

        if not os.path.exists(_CLS_TRAIN_SCRIPT):
            QMessageBox.critical(self, "缺少训练脚本", f"未找到训练脚本：\n{_CLS_TRAIN_SCRIPT}\n请先部署训练脚本。")
            return

        artifacts = _make_artifacts_dir()
        self._last_artifacts = artifacts
        self._mapping = mapping

        # 写出 classes.json / train_config.yaml，训练脚本直接读取
        # classes.json: {"1": "划痕", "2": "..."}
        id_to_name = {str(cid): name for name, cid in mapping.items()}
        _atomic_write_json(artifacts.classes_json_path, id_to_name)
        train_cfg = {
            "dataset_root": self._dataset_root,
            "mapping": mapping,
            "epochs": int(self.spinEpoch.value()),
            "batch_size": int(self.spinBatch.value()),
            "img_size": int(self.spinImg.value()),
            "out_checkpoint": artifacts.checkpoint_path,
            "out_metrics": artifacts.metrics_path,
            "out_classes_json": artifacts.classes_json_path,
        }
        _write_yaml(artifacts.train_config_path, train_cfg)

        self._log("准备开始训练。")
        self._log(f"产物目录：{artifacts.out_dir}")
        self._log(f"训练集：{self._dataset_root}")
        self._log(f"类别数：{len(mapping)}")

        # 启动进程
        if self._proc is None:
            self._proc = QProcess(self)
            self._proc.setProcessChannelMode(QProcess.MergedChannels)
            self._proc.readyReadStandardOutput.connect(self._on_proc_output)
            self._proc.finished.connect(self._on_proc_finished)

        self._proc.setWorkingDirectory(_PROJECT_ROOT)
        args = ["-u", _CLS_TRAIN_SCRIPT, "--config", artifacts.train_config_path]
        self._proc.start(_PYTHON_EXE, args)
        if not self._proc.waitForStarted(3000):
            QMessageBox.critical(self, "启动失败", "训练进程启动失败，请检查 Python 环境与路径。")
        else:
            self._log("训练进程已启动。")

        self._update_buttons()

    def _stop_train(self) -> None:
        if self._proc is None:
            return
        if self._proc.state() == QProcess.NotRunning:
            return
        self._log("正在请求停止训练进程...")
        self._proc.terminate()
        QtCore.QTimer.singleShot(2500, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        if self._proc is None:
            return
        if self._proc.state() != QProcess.NotRunning:
            self._log("训练进程未退出，执行强制结束。")
            self._proc.kill()

    def _on_proc_output(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput())
        if not data:
            return
        try:
            text = data.decode("utf-8")
        except Exception:
            text = data.decode("gbk", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self._log(line.rstrip())

    def _on_proc_finished(self, exit_code: int, _status) -> None:
        self._log(f"训练进程结束，exit_code={exit_code}")
        self._update_buttons()
        self._refresh_current_model_status()
        if exit_code == 0 and self._last_artifacts:
            ckpt = self._last_artifacts.checkpoint_path
            if os.path.exists(ckpt):
                self._log("训练完成：已生成 checkpoint，可点击「启用最新模型」。")
            else:
                self._log("训练结束但未找到 checkpoint，请检查日志。")

    def _enable_latest_model(self) -> None:
        if not self._last_artifacts:
            QMessageBox.information(self, "提示", "暂无可启用的训练产物。请先完成一次训练。")
            return
        ckpt = self._last_artifacts.checkpoint_path
        if not os.path.exists(ckpt):
            QMessageBox.warning(self, "无法启用", f"未找到模型权重：\n{ckpt}")
            return

        # 1) 写 config.yaml
        cfg = _read_yaml(_CONFIG_PATH)
        cfg["cls_model_path"] = ckpt
        _write_yaml(_CONFIG_PATH, cfg)

        # 2) 同步 rptcfg.class_labels（以训练集 mapping 为准）
        mapping = self._mapping or self._collect_mapping_from_table()
        if mapping:
            _rptcfg_set_class_labels_from_mapping(mapping)

        # 3) 写 runtime_state 额外信息（不影响 paused）
        try:
            k = max(mapping.values()) if mapping else None
        except Exception:
            k = None
        _write_runtime_state_extra(
            {
                "cls_model_path": ckpt,
                "cls_enabled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cls_num_classes": int(k) if isinstance(k, int) else None,
                "cls_artifacts_dir": self._last_artifacts.out_dir,
            }
        )

        # 4) 写入“类别对齐映射”（顺序允许不同，但集合必须一致）
        try:
            from cls_model_registry import compat_and_remap, rptcfg_class_names, write_runtime_remap

            # model classes：来自本次训练产物 classes.json（按 key 排序）
            model_classes = []
            try:
                with open(self._last_artifacts.classes_json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                if isinstance(obj, dict):
                    for kk in sorted(obj.keys(), key=lambda x: int(str(x))):
                        model_classes.append(str(obj.get(kk, "")).strip())
            except Exception:
                model_classes = []

            rpt_names = rptcfg_class_names(_RPTCFG_PATH)
            ok, remap, _diff = compat_and_remap(model_classes=model_classes, rptcfg_classes=rpt_names)
            if ok and remap:
                write_runtime_remap(
                    _RUNTIME_STATE_PATH,
                    model_path=ckpt,
                    model_classes=model_classes,
                    rptcfg_classes=rpt_names,
                    remap_modelidx_to_rptid_1based=list(remap),
                )
        except Exception:
            pass

        self._log("已启用最新分类模型，并同步类别配置。")
        self._refresh_current_model_status()
        QMessageBox.information(self, "启用成功", "已启用最新分类模型。\n建议：生成报告前先做一次分类推理与报告生成验证。")

    def _open_out_dir(self) -> None:
        if self._last_artifacts and os.path.isdir(self._last_artifacts.out_dir):
            os.startfile(self._last_artifacts.out_dir)  # type: ignore[attr-defined]
        else:
            os.makedirs(_CLS_TRAIN_RESULT_DIR, exist_ok=True)
            os.startfile(_CLS_TRAIN_RESULT_DIR)  # type: ignore[attr-defined]

    def _sync_labels_only(self) -> None:
        """
        仅把当前表格中的类别映射同步到 rptcfg.class_labels/class_list/print_cls。
        用于：先对齐报告侧显示与允收，再决定是否训练/启用新模型。
        """
        mapping = self._collect_mapping_from_table()
        err = self._validate_mapping(mapping)
        if err:
            QMessageBox.warning(self, "同步失败", err)
            return
        _rptcfg_set_class_labels_from_mapping(mapping)
        self._log("已同步类别到 rptcfg.yaml（未改动 cls_model_path）。")
        self._refresh_current_model_status()

