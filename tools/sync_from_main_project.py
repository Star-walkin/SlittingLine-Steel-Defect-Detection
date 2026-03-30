# -*- coding: utf-8 -*-
"""
从“主工程 steeldefect”同步必要文件到本备份仓库，并进行可移植化改写。

约束：
- 只写入本仓库（SlittingLine-Steel-Defect-Detection）内的文件
- 可读取主工程目录作为输入源

用法（推荐）：
  python tools/sync_from_main_project.py --source-root "d:\\pycharm_project\\steeldefect"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
from pathlib import Path


DEST_ROOT = Path(__file__).resolve().parents[1]

# 主工程绝对路径前缀（允许正斜杠/反斜杠）
ABS_PREFIX_RE = re.compile(
    r"[A-Za-z]:[\\/]+pycharm_project[\\/]+steeldefect[\\/]*",
    re.IGNORECASE,
)


UI_SRC_REL_DIR = Path("F_mainui") / "F_mainui"
UI_DST_REL_DIR = Path("ui")

UI_FILES = [
    "main.py",
    "mainui.py",
    "para.py",
    "report_change.py",
    "report_center.py",
    "cls_config.py",
    "report.py",
    "make_standard.py",
    "theme.qss",
    "requirements.txt",
]

ROOT_FILES = [
    "detect_anomalies_online.py",
    "gen_report_cls.py",
    "function_bank.py",
    "util.py",
    "speed_monitor.py",
    "cls_anomalies.py",
    "cls_model.py",
    "table.json",
]

CONFIG_FILES = [
    "config.yaml",
    "config0.yaml",
    "config_default.yaml",
    "auth.yaml",
    "rptcfg.yaml",
    "rpt01.yaml",
    "3.json",
]

DIRS_TO_SYNC = [
    "patchcore_model",
    "cls_model",
    # PatchCore-only：det_model 仅保留 PatchCore 复用的预处理代码
    "det_model",
]

DETMODEL_KEEP_FILES = {
    "__init__.py",
    "infer.py",
    "prepare_dataset_det.py",
    "PREPROCESS_README.md",
}


def _ignore_pyc(_dir: str, names: list[str]) -> set[str]:
    out: set[str] = set()
    for n in names:
        if n == "__pycache__" or n.endswith(".pyc"):
            out.add(n)
    return out


def _rmtree_force(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_onerror)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _strip_abs_prefix_for_yaml_json(text: str) -> str:
    # 仅剥离工程根前缀，得到相对路径（例如 config.yaml 中各权重路径）
    return ABS_PREFIX_RE.sub("", text)


def _ensure_repo_root_assignment(py: str, *, ui_subpackage: bool) -> str:
    """
    确保存在 _REPO_ROOT 定义：
    - UI 子包（F_mainui/F_mainui）：向上两级
    - 根目录脚本：当前文件所在目录
    """
    if "_REPO_ROOT" in py and re.search(r"^_REPO_ROOT\s*=", py, flags=re.M):
        return py

    assign = (
        "_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\n"
        if ui_subpackage
        else "_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))\n"
    )

    lines = py.splitlines(keepends=True)

    # 插在 import os 之后；若无 import os，则插在首个 import 区域顶部并补 import os
    import_os_idx = None
    first_import_idx = None
    for i, ln in enumerate(lines):
        if first_import_idx is None and (ln.startswith("import ") or ln.startswith("from ")):
            first_import_idx = i
        if ln.startswith("import os") or ln.startswith("import os,"):
            import_os_idx = i
            break

    if import_os_idx is not None:
        lines.insert(import_os_idx + 1, assign)
        return "".join(lines)

    if first_import_idx is None:
        return "import os\n" + assign + py

    lines.insert(first_import_idx, "import os\n")
    lines.insert(first_import_idx + 1, assign)
    return "".join(lines)


def _portableize_python_paths(py: str) -> str:
    """
    将硬编码的工程绝对路径字符串替换为基于 _REPO_ROOT 的 os.path.join(...)。
    仅处理字符串字面量里的路径。
    """

    def repl(m: re.Match) -> str:
        quote = m.group(1)
        full = m.group(2)
        tail = ABS_PREFIX_RE.sub("", full)
        tail = tail.replace("\\", "/").strip("/")
        if not tail:
            return f"os.path.join(_REPO_ROOT)"
        segs = [f"{quote}{p}{quote}" for p in tail.split("/") if p]
        return "os.path.join(_REPO_ROOT, " + ", ".join(segs) + ")"

    # raw string
    py = re.sub(
        r'r([\'"])([A-Za-z]:[\\/]+pycharm_project[\\/]+steeldefect(?:[\\/][^\'"]*)?)\1',
        repl,
        py,
    )
    # normal string
    py = re.sub(
        r'(?<!r)([\'"])([A-Za-z]:[\\/]+pycharm_project[\\/]+steeldefect(?:[\\/][^\'"]*)?)\1',
        repl,
        py,
    )

    # 解释器 / 外部程序：统一改为环境变量（让 UI 与报告脚本一致）
    py = re.sub(
        r"python_exe\s*=\s*r?[\"'].*?python\.exe[\"']",
        "python_exe = os.environ.get('STEEL_PYTHON_EXE', sys.executable)",
        py,
    )
    py = re.sub(
        r"_PYTHON_EXE\s*=\s*r?[\"'].*?python\.exe[\"']",
        "_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)",
        py,
    )
    return py


def _sync_file(src: Path, dst: Path, *, kind: str) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))

    if kind == "raw":
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    if kind == "py_ui":
        py = _read_text(src)
        py = _portableize_python_paths(py)
        py = _ensure_repo_root_assignment(py, ui_subpackage=True)
        _write_text(dst, py)
        return

    if kind == "py_root":
        py = _read_text(src)
        py = _portableize_python_paths(py)
        py = _ensure_repo_root_assignment(py, ui_subpackage=False)
        _write_text(dst, py)
        return

    if kind == "yaml_json":
        t = _read_text(src)
        t = _strip_abs_prefix_for_yaml_json(t)
        _write_text(dst, t)
        return

    raise ValueError(f"unknown kind={kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, help="主工程根目录，例如 d:\\pycharm_project\\steeldefect")
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    if not source_root.is_dir():
        raise SystemExit(f"[ERROR] source-root not found: {source_root}")

    # 1) UI 子包文件
    for fn in UI_FILES:
        _sync_file(
            source_root / UI_SRC_REL_DIR / fn,
            DEST_ROOT / UI_DST_REL_DIR / fn,
            kind="py_ui" if fn.endswith(".py") else "raw",
        )

    # 2) 根目录脚本/模块
    for fn in ROOT_FILES:
        src = source_root / fn
        dst = DEST_ROOT / fn
        if fn.endswith(".py"):
            _sync_file(src, dst, kind="py_root")
        elif fn.endswith(".json"):
            _sync_file(src, dst, kind="yaml_json")
        else:
            _sync_file(src, dst, kind="raw")

    # 3) config
    for fn in CONFIG_FILES:
        _sync_file(
            source_root / "config" / fn,
            DEST_ROOT / "config" / fn,
            kind="yaml_json" if (fn.endswith(".yaml") or fn.endswith(".yml") or fn.endswith(".json")) else "raw",
        )

    # 4) 目录同步（先整体覆盖，后续 patchcore-only 会进一步裁剪）
    for d in DIRS_TO_SYNC:
        src_dir = source_root / d
        dst_dir = DEST_ROOT / d
        if not src_dir.is_dir():
            continue
        if dst_dir.exists():
            _rmtree_force(dst_dir)
        shutil.copytree(src_dir, dst_dir, ignore=_ignore_pyc, dirs_exist_ok=True)

    # 4.1) PatchCore-only：裁剪 det_model 到“仅预处理子集”
    det_dst = DEST_ROOT / "det_model"
    if det_dst.is_dir():
        for p in list(det_dst.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(det_dst).as_posix()
            if "/" in rel:
                # 不保留子目录（例如 train-result/）
                try:
                    p.unlink()
                except Exception:
                    pass
                continue
            if p.name not in DETMODEL_KEEP_FILES:
                try:
                    p.unlink()
                except Exception:
                    pass
        # 清理空目录
        for d in sorted([x for x in det_dst.rglob("*") if x.is_dir()], key=lambda x: len(str(x)), reverse=True):
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass

    # 5) 基础运行目录占位
    (DEST_ROOT / "detect result").mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / "detect result" / ".gitkeep").write_text("", encoding="utf-8")

    print("[OK] synced from", str(source_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

