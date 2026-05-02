"""
检测结果中“每条带钢”目录命名规则（与 UI 输入的带钢卡号对齐）。

约定：
- 每条带钢在相机目录（例如 上表面/ 下表面/）下是一个子文件夹。
- 新数据：优先使用 config0 中的 strip_card_* / strip_card_list，经 sanitize 后作为文件夹名。
- 兼容旧数据：仍支持 strip_1、strip_2 ... 目录名。
- 为避免 Windows 非法字符/重名/空名等问题，实际落盘目录名可能与原始卡号略有不同；
  因此在每次开卷时把最终目录名列表写入卷根 config0_snapshot.yaml 的 strip_dir_list 字段，供 UI/报告侧稳定解析。
"""

from __future__ import annotations

import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
import re
from typing import Any, Dict, List, Optional

import yaml

_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"", ".", ".."}


def sanitize_strip_dir_name(raw: str) -> str:
    s = str(raw or "").strip()
    s = _WIN_INVALID.sub("_", s)
    s = s.rstrip(" .")
    if s in _RESERVED:
        return ""
    # Windows 不建议以空格结尾；上面已 rstrip
    return s


def _raw_strip_cards(config0: Optional[Dict[str, Any]]) -> List[str]:
    cfg = config0 or {}
    cards = cfg.get("strip_card_list")
    out: List[str] = []
    if isinstance(cards, (list, tuple)):
        out = [str(x or "").strip() for x in cards]
    if not out:
        out = [
            str(cfg.get("strip_card_1", "") or "").strip(),
            str(cfg.get("strip_card_2", "") or "").strip(),
            str(cfg.get("strip_card_3", "") or "").strip(),
            str(cfg.get("strip_card_4", "") or "").strip(),
        ]
    return out


def build_strip_dir_names(config0: Optional[Dict[str, Any]], strip_count: int) -> List[str]:
    """
    返回长度=strip_count 的目录名列表（仅 basename，不含路径）。
    """
    n = int(strip_count or 0)
    if n <= 0:
        return []

    raw = (_raw_strip_cards(config0) + [""] * n)[:n]
    used: set[str] = set()
    out: List[str] = []

    for i in range(n):
        card = str(raw[i] or "").strip()
        base = sanitize_strip_dir_name(card) if card else ""
        if not base:
            base = f"strip_{i + 1}"

        cand = base
        k = 2
        while cand.lower() in used:
            cand = f"{base}_{k}"
            k += 1
        used.add(cand.lower())
        out.append(cand)

    return out


def read_result_roll_config0(result_all_path: str) -> Dict[str, Any]:
    snap = os.path.join(str(result_all_path or ""), "config0_snapshot.yaml")
    try:
        if os.path.exists(snap):
            with open(snap, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f) or {}
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
    return {}


def strip_dir_list_from_roll(result_all_path: str, strip_count: int) -> List[str]:
    cfg0 = read_result_roll_config0(result_all_path)
    xs = cfg0.get("strip_dir_list")
    if isinstance(xs, list) and xs and all(isinstance(x, (str, int, float)) for x in xs):
        names = [str(x) for x in xs]
        if len(names) >= int(strip_count or 0):
            return names[: int(strip_count or 0)]
    # fallback：按快照里的卡号规则重建（对旧快照也尽量一致）
    return build_strip_dir_names(cfg0, int(strip_count or 0))


def resolve_strip_dir_basename(result_all_path: str, strip_index_1based: int, strip_count: Optional[int] = None) -> str:
    """
    将“条带序号(1 起)”解析为实际目录 basename。
    """
    sid = int(strip_index_1based)
    if sid < 1:
        sid = 1

    n_hint = int(strip_count or 0)
    names = strip_dir_list_from_roll(result_all_path, n_hint) if n_hint > 0 else []
    if not names:
        # 尽量从快照 strip_count 推断长度
        cfg0 = read_result_roll_config0(result_all_path)
        n2 = int(cfg0.get("strip_count", 0) or 0)
        if n2 <= 0:
            # 最后兜底：保持旧规则
            return f"strip_{sid}"
        names = strip_dir_list_from_roll(result_all_path, n2)

    if 1 <= sid <= len(names):
        return str(names[sid - 1])

    # 若快照缺失但目录仍存在旧 strip_k，则回退
    return f"strip_{sid}"


def discover_strip_dir_basenames_under_cam(cam_dir: str) -> List[str]:
    """
    在相机目录下发现条带目录：
    - 优先：strip_*
    - 否则：包含 fukuan.json 或 defect_events_center.jsonl 的子目录（排除常见非条带目录）
    """
    cam_dir = str(cam_dir or "")
    if not cam_dir or not os.path.isdir(cam_dir):
        return []

    try:
        subs = [d for d in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir, d))]
    except Exception:
        return []

    legacy = [d for d in subs if d.startswith("strip_")]
    if legacy:
        def _key(name: str):
            # strip_12 -> 12
            try:
                return int(str(name).split("_", 1)[1])
            except Exception:
                return 10**9

        legacy.sort(key=_key)
        return legacy

    skip = {
        "split_vis",
        "debug_visuals",
        "report",
        "__pycache__",
    }
    found: List[str] = []
    for d in subs:
        if d in skip:
            continue
        p = os.path.join(cam_dir, d)
        if os.path.exists(os.path.join(p, "fukuan.json")) or os.path.exists(os.path.join(p, "defect_events_center.jsonl")):
            found.append(d)
    found.sort()
    return found
