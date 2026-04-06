"""
桥接模块：
你通常在 `F_mainui/F_mainui` 目录下直接运行 `python main.py`，
此时 Python 的 import 搜索路径不包含项目根目录 `steeldefect/`，
会导致根目录的 `cls_model_registry.py` 无法导入。

这个文件把根目录实现动态加载进来，并重新导出其公共函数/类，
从而让 `from cls_model_registry import ...` 在 UI 目录下也能工作。
"""

from __future__ import annotations

import importlib.util
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
from types import ModuleType


def _load_root_module() -> ModuleType:
    # 源码：从 F_mainui/F_mainui 上溯两级到工程根。
    # PyInstaller：根目录的 cls_model_registry.py 由 spec 以 datas 打入 _MEIPASS（与可执行体同包）。
    if getattr(sys, "frozen", False):
        project_root = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    else:
        here = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", ".."))
    root_path = os.path.join(project_root, "cls_model_registry.py")
    if not os.path.exists(root_path):
        raise ModuleNotFoundError(f"未找到根目录模块：{root_path}")

    # 将 project_root 放入 sys.path，便于根模块内部的相对 import（若未来有）正常工作
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    spec = importlib.util.spec_from_file_location("_cls_model_registry_root", root_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"无法加载根目录模块：{root_path}")
    mod = importlib.util.module_from_spec(spec)
    # 关键：先注册到 sys.modules，避免 dataclasses/typing 在执行期间取不到模块命名空间
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_root = _load_root_module()

# 重新导出公共 API
scan_model_registry = getattr(_root, "scan_model_registry")
compat_and_remap = getattr(_root, "compat_and_remap")
write_runtime_remap = getattr(_root, "write_runtime_remap")
rptcfg_class_names = getattr(_root, "rptcfg_class_names")
ModelEntry = getattr(_root, "ModelEntry")

__all__ = [
    "scan_model_registry",
    "compat_and_remap",
    "write_runtime_remap",
    "rptcfg_class_names",
    "ModelEntry",
]

