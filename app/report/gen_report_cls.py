import matplotlib.pyplot as plt

from function_bank import Anomaly_info_List, Statistic_anomaly, find_folders_with_id
from datetime import datetime, timedelta
from PIL import Image
import argparse
import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
import json
import yaml
import pandas as pd
from cls_anomalies import cls_anomalies
import matplotlib

_DETECT_ROOT_DEFAULT = os.path.join(_REPO_ROOT, "detect result")
_PROJECT_ROOT = os.path.join(_REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import strip_result_paths as _strip_paths


def _stable_color_for_text(text: str) -> str:
    # 稳定的默认配色（避免配置缺项导致报告崩溃；同名类别跨次运行颜色保持一致）
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]
    return palette[abs(hash(str(text))) % len(palette)]


def _normalize_rptcfg_for_report(rptcfg: dict) -> dict:
    """
    最稳妥策略：报告生成前做一次配置自检/归一化，避免小配置问题中断产线流程。
    - 保证 class_labels/class_list/print_cls 类型正确
    - colors 缺项自动补齐（按 class_labels 的中文名）
    """
    rptcfg = dict(rptcfg or {})
    # class_labels：优先按当前 product_cls 取覆盖（避免“改一个型号影响所有型号”的现场困惑）
    product_cls = str(rptcfg.get("product_cls", "") or "").strip()
    by = rptcfg.get("class_labels_by_product")
    if product_cls and isinstance(by, dict) and isinstance(by.get(product_cls), dict):
        class_labels = by.get(product_cls) or {}
    else:
        class_labels = rptcfg.get("class_labels") or {}
    # class_labels: yaml 可能把 key 读成 int 或 str，统一为 int->strLabel
    norm_labels = {}
    for k, v in (class_labels or {}).items():
        try:
            ik = int(k)
        except Exception:
            continue
        norm_labels[ik] = str(v)
    rptcfg["class_labels"] = norm_labels

    # class_list / print_cls: 统一为 int list
    def _to_int_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = []
            for it in x:
                try:
                    out.append(int(it))
                except Exception:
                    pass
            return out
        try:
            return [int(x)]
        except Exception:
            return []

    rptcfg["class_list"] = _to_int_list(rptcfg.get("class_list"))
    # UI「全选类别」会写入 print_cls=[] 表示不限制；报告侧必须展开为全部类别，否则 select_anomalies 空循环 → 零缺陷
    _pcl = _to_int_list(rptcfg.get("print_cls"))
    rptcfg["print_cls"] = list(rptcfg["class_list"]) if not _pcl else _pcl

    # colors: 用中文类别名作为 key；缺项补齐
    colors = rptcfg.get("colors")
    if not isinstance(colors, dict):
        colors = {}
    else:
        colors = dict(colors)

    # 若 colors 使用了旧标签（比如“穿孔”）而 class_labels 改成了“破洞”，这里不会强制改名，
    # 但会确保“破洞”至少有一个可用颜色。
    for _cid, _name in norm_labels.items():
        if _name and _name not in colors:
            colors[_name] = _stable_color_for_text(_name)
    rptcfg["colors"] = colors
    return rptcfg


def _normalize_report_cam_ids(ids):
    """与报告修改界面一致：支持 list、标量 '2'、'2,3'。"""
    if ids is None:
        return []
    if isinstance(ids, (list, tuple)):
        return [str(x).strip() for x in ids if str(x).strip()]
    s = str(ids).strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _write_report_snapshots(strip_dir: str, *, rptcfg_raw: dict, config: dict, config0: dict) -> None:
    """
    报告可追溯性：在每条带钢 report 目录写入当次使用的配置快照（不影响主流程）。
    - rptcfg_snapshot.yaml：当次读取到的原始 rptcfg
    - model_snapshot.json：cls_model_path + K + class_labels（normalize 后，报告实际使用口径）
    """
    try:
        os.makedirs(strip_dir, exist_ok=True)
    except Exception:
        return

    # 1) rptcfg 原始快照
    try:
        snap_path = os.path.join(strip_dir, "rptcfg_snapshot.yaml")
        with open(snap_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(rptcfg_raw or {}), f, allow_unicode=True)
    except Exception:
        pass

    # 1.5) config0 快照（质保书号/幅宽/带钢卡号等来自主界面“确认”）
    try:
        snap_path = os.path.join(strip_dir, "config0_snapshot.yaml")
        with open(snap_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(config0 or {}), f, allow_unicode=True)
    except Exception:
        pass

    # 2) model snapshot（包含类别对齐信息，便于追溯）
    model_path = str((config or {}).get("cls_model_path", "") or "").strip()
    k = None
    if model_path and os.path.exists(model_path):
        try:
            import torch

            w = torch.load(model_path, map_location="cpu")
            k = int(w["out.weight"].shape[0])
        except Exception:
            k = None

    try:
        rptcfg_norm = _normalize_rptcfg_for_report(rptcfg_raw or {})
        class_labels = rptcfg_norm.get("class_labels", {})
        class_labels_json = {str(_k): str(_v) for _k, _v in (class_labels or {}).items()}
    except Exception:
        class_labels_json = {}

    # remap / compat info（不阻断主流程）
    compat_ok = None
    remap = None
    diff = None
    try:
        from cls_model_registry import compat_and_remap as _compat_and_remap

        # 读取 model classes：优先同目录 classes.json
        model_classes = []
        cj = os.path.join(os.path.dirname(model_path), "classes.json") if model_path else ""
        if cj and os.path.exists(cj):
            try:
                with open(cj, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                if isinstance(obj, dict) and "id_to_name" in obj and isinstance(obj["id_to_name"], dict):
                    obj = obj["id_to_name"]
                if isinstance(obj, dict):
                    for kk in sorted(obj.keys(), key=lambda x: int(str(x))):
                        model_classes.append(str(obj.get(kk, "")).strip())
            except Exception:
                model_classes = []

        # rptcfg classes：按 1..K 顺序
        rpt_classes = []
        try:
            for kk in sorted((class_labels_json or {}).keys(), key=lambda x: int(str(x))):
                rpt_classes.append(str(class_labels_json.get(kk, "")).strip())
        except Exception:
            rpt_classes = list(class_labels_json.values())

        if model_classes and rpt_classes:
            compat_ok, remap, diff = _compat_and_remap(model_classes=model_classes, rptcfg_classes=rpt_classes)
    except Exception:
        compat_ok, remap, diff = None, None, None

    try:
        p = os.path.join(strip_dir, "model_snapshot.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cls_model_path": model_path,
                    "cls_num_classes": k,
                    "class_labels": class_labels_json,
                    "cls_compat_ok": compat_ok,
                    "cls_label_remap_modelidx_to_rptid_1based": remap,
                    "cls_compat_diff": diff,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass


def _fukuan_at(fukuan0, idx):
    if isinstance(fukuan0, (list, tuple)):
        if not fukuan0:
            return 0.0
        if 0 <= idx < len(fukuan0):
            return float(fukuan0[idx])
        return float(fukuan0[-1])
    return float(fukuan0)


def filter_classes(A, class_list):
    result = {cls: [] for cls in class_list}
    # 你的数据是 [[...]] 嵌套格式，这里扁平化是正确的
    A = [item for sublist in A for item in sublist]
    class_set = set(class_list)

    for item in A:
        parts = item.split('_')
        if len(parts) < 4:
            continue

        try:
            class_value = int(parts[0])
            # 兼容历史：分类脚本曾写出 0-based 类号，报告侧 class_list 多为 1-based
            if class_value not in class_set and (class_value + 1) in class_set:
                class_value = class_value + 1
            if class_value not in class_set:
                continue

            # 关键修改：转换为数值类型，否则后面的 float() 转换可能会因为空字符报错
            # 并且确保 parts 的索引对应 json 里的 x, y, area
            x = float(parts[1])
            y = float(parts[2])
            area = float(parts[3])

            result[class_value].append([x, y, area])
        except Exception as e:
            print(f"解析条目 {item} 出错: {e}")
            continue

    return result


def _coerce_positive_float_list(seq, fallback_nominal: float):
    """将幅宽序列转为 float；非法项用标称幅宽兜底（避免绘图断线）。"""
    fb = float(fallback_nominal or 0.0)
    out = []
    for v in seq or []:
        try:
            fv = float(v)
            if fv == fv and fv > 0:
                out.append(fv)
            else:
                out.append(fb if fb > 0 else 1.0)
        except Exception:
            out.append(fb if fb > 0 else 1.0)
    return out


def _load_fukuan_base_series_for_report(json_list, meta_path: str, fallback_nominal: float):
    """
    与 UI/检测端一致：若存在 fukuan_meta.json 且与 fukuan.json 等长，则用每条记录的 stable 作为曲线基准；
    否则退回 fukuan.json 浮点序列（与 WaveformThread 读取的 fukuan.json 同源）。
    """
    base = _coerce_positive_float_list(json_list, fallback_nominal)
    if not base or not meta_path or not os.path.isfile(meta_path):
        return base
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, list) or len(meta) != len(base):
            return base
        stable = []
        for row in meta:
            if isinstance(row, dict) and "stable" in row:
                try:
                    fv = float(row["stable"])
                    stable.append(fv if fv == fv and fv > 0 else None)
                except Exception:
                    stable.append(None)
            else:
                stable.append(None)
        if all(v is not None for v in stable):
            return stable  # type: ignore[return-value]
    except Exception:
        pass
    return base


def _rolling_median_1d(values, window: int):
    from statistics import median

    n = len(values)
    if n == 0:
        return []
    w = max(3, int(window) | 1)
    w = min(w, n if (n % 2 == 1) else n - 1)
    if w < 3:
        w = 3
    half = w // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(float(median(values[lo:hi])))
    return out


def _rolling_mean_1d(values, window: int):
    n = len(values)
    if n == 0:
        return []
    w = max(3, int(window) | 1)
    w = min(w, n if (n % 2 == 1) else n - 1)
    if w < 3:
        w = 3
    half = w // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sl = values[lo:hi]
        out.append(sum(sl) / float(len(sl)))
    return out


def _soft_limit_deviation_from_nominal(vals: list, nominal_mm: float, band_mm: float = 40.0) -> list:
    """
    对标称幅宽做软限幅：越界点按比例拉回，避免整段被硬砍成平台；band 外保留约 12% 原始偏离，曲线仍自然。
    """
    c = float(nominal_mm or 0.0)
    if c <= 0 or not vals:
        return vals
    lo, hi = c - float(band_mm), c + float(band_mm)
    k = 0.12
    out = []
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            fv = c
        if fv < lo:
            out.append(lo + (fv - lo) * k)
        elif fv > hi:
            out.append(hi + (fv - hi) * k)
        else:
            out.append(fv)
    return out


def _prepare_fukuan_curve_for_report_plot(base: list, nominal_mm: float) -> list:
    """
    报告专用：抑制带钢晃动/切分不稳造成的孤立尖峰（与 UI 侧 subsample+interp 观感接近），
    不改变磁盘原始 fukuan*.json，仅影响幅宽曲线图 PNG。
    """
    if not base:
        return []
    n = len(base)
    nom = float(nominal_mm or 0.0)
    xs = list(base)
    if n < 5:
        return xs
    # 窗口随长度略增：成段 422/476 交替时，较大窗口才能把曲线拉回「趋势中线」附近，避免整段竖条毛刺感
    w_med = max(7, min(35, 9 + n // 40))
    if w_med % 2 == 0:
        w_med += 1
    med = _rolling_median_1d(xs, w_med)
    try:
        from statistics import median as _med

        resid = sorted(abs(xs[i] - med[i]) for i in range(n))
        mad = _med(resid) if resid else 0.0
        # 阈值略宽：422↔476 交替时单点与中值的差常在 20~35mm，需一次压回中值避免「栅栏」观感
        spike_thr = max(28.0, nom * 0.055, 4.0 * (mad if mad > 1e-6 else 22.0))
    except Exception:
        spike_thr = max(30.0, nom * 0.06)
    merged = [med[i] if abs(xs[i] - med[i]) > spike_thr else xs[i] for i in range(n)]
    merged = _rolling_median_1d(merged, min(21, w_med))
    w_ma = max(3, min(9, 5 + n // 400))
    if w_ma % 2 == 0:
        w_ma += 1
    smoothed = _rolling_mean_1d(merged, w_ma)
    # 将明显脱离标称幅宽的「晃动尖峰」压回标称附近（仅影响曲线观感，不修改磁盘数据）
    smoothed = _soft_limit_deviation_from_nominal(smoothed, nom, band_mm=40.0)
    return _rolling_mean_1d(smoothed, min(5, max(3, w_ma - 2)))


def _fukuan_fig_ylim(plot_vals: list, nominal_mm: float):
    """
    纵轴从 0 起完整显示，上沿按「输入幅宽 / 测量序列」取较大值后留余量并取整到 50mm 刻度。
    这样 ~50mm 级偏差相对整轴高度占比小，观感更接近甲方对「全量程」报告的阅读习惯。
    """
    import math

    nom = float(nominal_mm or 0.0)
    nums = [float(v) for v in (plot_vals or []) if v == v and v > 0]
    mx_meas = max(nums) if nums else 0.0
    ceiling = max(mx_meas, nom, 1.0)
    # 上沿：相对峰值约 +6% 后向上取整到 50mm，常见结果如 528→600、400→450
    hi = math.ceil(ceiling * 1.06 / 50.0) * 50.0
    hi = max(hi, 150.0)  # 窄带钢时仍保留可读高度
    hi = min(hi, 3000.0)  # 极大规格安全上界 (mm)
    return (0.0, float(hi))


def fukuan_fig(list_data, value, save_path, strip_id, ratio):
    """
    list_data：已为「展示用」平滑后的幅宽序列（与磁盘原始列表等长）。
    value：输入幅宽（标称，mm）。
    """
    value = float(value)
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    x = list(range(len(list_data)))
    x = [i * ratio * 4096 / 1000000 for i in x]
    y_lo, y_hi = _fukuan_fig_ylim(list_data, value)

    ax.plot(
        x,
        list_data,
        label="测量幅宽",
        color="b",
        linestyle="-",
        linewidth=1.15,
        solid_capstyle="round",
        antialiased=True,
    )
    ax.plot(x, [value] * len(list_data), label="输入幅宽", color="g", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("幅宽变化曲线", fontname="SimHei", fontweight="bold", fontsize=11)
    ax.set_xlabel("钢带长度:Km", fontname="SimHei", fontweight="bold", fontsize=10)
    ax.set_ylabel("钢带幅宽:mm", fontname="SimHei", fontweight="bold", fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    ax.legend(loc="upper right", fontsize=9)
    out_png = os.path.join(save_path, f"strip_{strip_id + 1}_fukuanfig.png")
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.12, dpi=120)
    plt.close(fig)

def gen_detect_report(test_path,fukuan0,conduct_id,anomaly_area_cls_range,remove_threshold,
                      steel_length_range,calibrat_cam_id,camrea_id_up_report,camrea_id_down_report,updates_up,
                      updates_down,print_cls,class_list,update_info,standard_area_tables,colors,
                      class_labels,area_range,strip_id,result_all_path,fukuan_ratio,
                      *, rptcfg_raw_snapshot=None, config_snapshot=None, config0_snapshot=None, strip_card_no: str = ""):

    '''cam_id = "1"  # 此处的id为标定幅宽的相机编号'''

    #print("DEBUG fukuan0 =", fukuan0, type(fukuan0))
    # ===========================
    #   创建报告层级目录
    # ===========================

    # 在 conduct_id 下创建 report 文件夹
    report_root = os.path.join(result_all_path, "report")
    os.makedirs(report_root, exist_ok=True)

    strip_folder_basename = _strip_paths.resolve_strip_dir_basename(result_all_path, int(strip_id) + 1)
    # 针对当前条带，再建一个独立子目录（与检测结果条带目录同名，便于归档/检索）
    strip_dir = os.path.join(str(report_root), strip_folder_basename)
    os.makedirs(strip_dir, exist_ok=True)

    # 以后所有保存路径，都从 strip_dir 开始

    # 快照（失败不影响报告生成）
    try:
        _write_report_snapshots(
            strip_dir,
            rptcfg_raw=dict(rptcfg_raw_snapshot or {}),
            config=dict(config_snapshot or {}),
            config0=dict(config0_snapshot or {}),
        )
    except Exception:
        pass


    start_time = datetime.now()
    start_time_str_re = start_time.strftime("%Y-%m-%d %H:%M")

    def load_json_safe(p, default):
        if not os.path.exists(p):
            print(f"[WARN] JSON 不存在：{p}")
            return default
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] JSON 读取失败：{p} err={e}")
            return default

    # ====== 1) 读取幅宽（从标定相机的 strip 目录里读）======
    calib_key = _strip_paths.resolve_roll_cam_subdir(
        result_all_path, str(calibrat_cam_id), "上表面", "2", "1", "下表面", "3", "4"
    )
    calib_strip_dir = os.path.join(result_all_path, calib_key, strip_folder_basename)
    fukuan_path = os.path.join(calib_strip_dir, "fukuan.json")
    fukuan_meta_path = os.path.join(calib_strip_dir, "fukuan_meta.json")
    fukuan_list = load_json_safe(fukuan_path, [])
    if len(fukuan_list) == 0:
        print(f"[WARN] fukuan_list 为空：{fukuan_path}")
    fukuan_max = max(fukuan_list) if fukuan_list else float(fukuan0)
    fukuan_min = min(fukuan_list) if fukuan_list else float(fukuan0)

    # 幅宽图：优先 meta.stable，再做报告专用平滑（不改变 fukuan_max/min 统计口径）
    try:
        fukuan0_f = float(fukuan0)
    except Exception:
        fukuan0_f = float(fukuan_max) if fukuan_list else 0.0
    _base_curve = _load_fukuan_base_series_for_report(fukuan_list, fukuan_meta_path, fukuan0_f)
    fukuan_plot_curve = _prepare_fukuan_curve_for_report_plot(_base_curve, fukuan0_f)
    if not fukuan_plot_curve:
        fukuan_plot_curve = _coerce_positive_float_list(fukuan_list, fukuan0_f)

    # 幅宽图：使用与缺陷图一致的长度换算系数
    fukuan_fig(fukuan_plot_curve, fukuan0_f, strip_dir, strip_id, fukuan_ratio)

    # 缺陷图横坐标统一策略：
    # - steel_length_range 为空时，用当前幅宽图的 xmax_km 作为缺陷图 xlim 的 xmax
    # - 避免缺陷点集为空时 fallback 到 1.0km 造成横坐标范围错位
    steel_length_range_plot = steel_length_range
    if not steel_length_range_plot:
        if isinstance(fukuan_list, list) and len(fukuan_list) > 1:
            xmax_km = (len(fukuan_list) - 1) * fukuan_ratio * 4096 / 1_000_000
        else:
            xmax_km = 0.0
        steel_length_range_plot = [0.0, float(xmax_km)]

    Statistic = Statistic_anomaly(
        conduct_id=conduct_id,
        fukuan=fukuan0,
        range=anomaly_area_cls_range,
        result_path=strip_dir,  # ✨ 改这里
        start_time=start_time_str_re,
        remove_threshold=remove_threshold,
        steel_length_range=steel_length_range_plot,
        update_info=update_info,
        standard_area_tables=standard_area_tables,
        colors=colors,
        class_labels=class_labels,
        cls_all=class_list,
        area_range=area_range,
        strip_card_no=str(strip_card_no or "").strip(),
    )

    # ====== 2) 读取上表面各相机分类结果 ======
    up_anomaly_info_all = []
    _seen_up_json: set = set()
    for cam_id in _normalize_report_cam_ids(camrea_id_up_report):
        key = _strip_paths.resolve_roll_cam_subdir(result_all_path, str(cam_id), "上表面", "2", "1")
        p = os.path.join(result_all_path, key, strip_folder_basename, "anomaly_info_result.json")
        if p in _seen_up_json:
            continue
        _seen_up_json.add(p)
        data = load_json_safe(p, [])
        up_anomaly_info_all.append(data)

    # ====== 3) 读取下表面各相机分类结果 ======
    down_anomaly_info_all = []
    _seen_dn_json: set = set()
    for cam_id in _normalize_report_cam_ids(camrea_id_down_report):
        key = _strip_paths.resolve_roll_cam_subdir(result_all_path, str(cam_id), "下表面", "3", "4")
        p = os.path.join(result_all_path, key, strip_folder_basename, "anomaly_info_result.json")
        if p in _seen_dn_json:
            continue
        _seen_dn_json.add(p)
        data = load_json_safe(p, [])
        down_anomaly_info_all.append(data)

    # 扁平化（你后面 filter_classes 期望的是 list[list[str]] 或更扁的结构）
    up_anomaly_info_all = [item for sublist in up_anomaly_info_all for item in sublist]
    down_anomaly_info_all = [item for sublist in down_anomaly_info_all for item in sublist]

    # 避免终端输出超长列表刷屏（报告中心会卡顿）
    try:
        print(f"[DEBUG] up/down raw items: {len(up_anomaly_info_all)} / {len(down_anomaly_info_all)}")
    except Exception:
        pass
    up_anomaly_info_all_for_class=filter_classes(up_anomaly_info_all,class_list=class_list)
    down_anomaly_info_all_for_class=filter_classes(down_anomaly_info_all,class_list=class_list)
    try:
        up_cnt = {k: len(v) for k, v in (up_anomaly_info_all_for_class or {}).items()}
        down_cnt = {k: len(v) for k, v in (down_anomaly_info_all_for_class or {}).items()}
        print(f"[DEBUG] up/down by class counts: {up_cnt} / {down_cnt}")
    except Exception:
        pass
    Statistic.select_anomalies(up_anomaly_info_all_for_class,strip_id,print_cls=print_cls,surface="上",updates=updates_up)
    Statistic.select_anomalies(down_anomaly_info_all_for_class,strip_id,print_cls=print_cls,surface="下",updates=updates_down)



    up_dis = Image.open(os.path.join(strip_dir, f"strip_{strip_id+1}_anomaly_distribution_上表面.png"))
    up_cls = Image.open(os.path.join(strip_dir, f"strip_{strip_id+1}_anomaly_area_cls_上表面.png"))

    down_dis = Image.open(os.path.join(strip_dir, f"strip_{strip_id+1}_anomaly_distribution_下表面.png"))
    down_cls = Image.open(os.path.join(strip_dir, f"strip_{strip_id+1}_anomaly_area_cls_下表面.png"))

    print("up_dis type:", type(up_dis))
    print("down_dis type:", type(down_dis))

    fukuanfig=Image.open(os.path.join(strip_dir, f"strip_{strip_id+1}_fukuanfig.png"))
    report=Statistic.gen_report(up_dis, up_cls, down_dis, down_cls,fukuan_max,fukuan_min,fukuanfig)
    Statistic.judje_product(report,strip_id)
    print("[DEBUG] calib fukuan_path:", fukuan_path, "len=", len(fukuan_list))
    print("[DEBUG] up/down raw lens:", len(up_anomaly_info_all), len(down_anomaly_info_all))


def gen_multi_strip_report(base_path, single_reports):
    """
    汇总多带钢报告：将多条带钢的结果生成一份总览报告
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    summary_path = os.path.join(base_path, "多带钢汇总报告.pdf")
    doc = SimpleDocTemplate(summary_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>多带钢检测报告汇总</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    for i, rpt in enumerate(single_reports, start=1):
        story.append(Paragraph(f"{i}. {os.path.basename(os.path.dirname(rpt))} 报告路径：{rpt}", styles["Normal"]))
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 30))
    story.append(Paragraph("汇总生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["Italic"]))

    doc.build(story)
    print(f"多带钢汇总报告已生成：{summary_path}")


def gen_all_reports(result_all_path, fukuan0, conduct_id, anomaly_area_cls_range, remove_threshold,
                    steel_length_range, calibrat_cam_id, camrea_id_up_report, camrea_id_down_report,
                    updates_up, updates_down, print_cls, class_list, update_info,
                    standard_area_tables, colors, class_labels, area_range,
                    camrea_id_up_list=None, strip_only_0based=None, fukuan_ratio=0.1575,
                    rptcfg_raw_snapshot=None, config_snapshot=None, config0_snapshot=None, strip_card_list=None):
    """
    多带钢检测报告生成器：
    - 自动检测 strip_ 文件夹数量（在第一个上表面相机目录下扫描）
    - 为每条带钢生成单独报告
    - 汇总生成总报告
    strip_only_0based: 仅生成该索引对应条带（0 起）；None 表示全部。
    """
    if not camrea_id_up_list:
        print("[ERROR] camrea_id_up_list 为空，无法定位条带目录。", file=sys.stderr)
        sys.exit(1)

    first_cam = _strip_paths.pick_strip_scan_root_subdir(result_all_path, camrea_id_up_list)
    cam_dir = os.path.join(result_all_path, first_cam)
    if not os.path.isdir(cam_dir):
        print(f"[ERROR] 相机目录不存在：{cam_dir}", file=sys.stderr)
        sys.exit(1)

    bases = _strip_paths.discover_strip_dir_basenames_under_cam(cam_dir)
    strip_paths = [os.path.join(cam_dir, b) for b in bases]

    if len(strip_paths) == 0:
        # 单条带钢，维持原逻辑
        print("检测为单条带钢，按原逻辑生成报告。")
        fw = _fukuan_at(fukuan0, 0)
        gen_detect_report(
            result_all_path, fw, conduct_id, anomaly_area_cls_range,
            remove_threshold, steel_length_range, calibrat_cam_id,
            camrea_id_up_report, camrea_id_down_report, updates_up, updates_down,
            print_cls, class_list, update_info, standard_area_tables,
            colors, class_labels, area_range, 0, result_all_path, fukuan_ratio,
            rptcfg_raw_snapshot=rptcfg_raw_snapshot, config_snapshot=config_snapshot,
            config0_snapshot=config0_snapshot,
        )
        print("报告生成完成。")
        return

    if strip_only_0based is not None and not (0 <= strip_only_0based < len(strip_paths)):
        print(
            f"[ERROR] --strip_id 超出范围：有效 1~{len(strip_paths)}，当前索引 {strip_only_0based + 1}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"检测到 {len(strip_paths)} 条带钢，正在分别生成报告...")

    single_reports = []

    for strip_id, strip_path in enumerate(strip_paths):
        if strip_only_0based is not None and strip_id != strip_only_0based:
            continue
        strip_name = os.path.basename(strip_path)
        print(f"\n正在生成 {strip_name} 的检测报告...")

        fw = _fukuan_at(fukuan0, strip_id)
        card = ""
        try:
            if isinstance(strip_card_list, (list, tuple)) and strip_id < len(strip_card_list):
                card = str(strip_card_list[strip_id] or "").strip()
        except Exception:
            card = ""
        gen_detect_report(
            strip_path + "/", fw, conduct_id, anomaly_area_cls_range,
            remove_threshold, steel_length_range, calibrat_cam_id,
            camrea_id_up_report, camrea_id_down_report, updates_up, updates_down,
            print_cls, class_list, update_info, standard_area_tables,
            colors, class_labels, area_range, strip_id, result_all_path, fukuan_ratio,
            rptcfg_raw_snapshot=rptcfg_raw_snapshot, config_snapshot=config_snapshot, config0_snapshot=config0_snapshot, strip_card_no=card,
        )

        single_reports.append(strip_path + "/检测报告.pdf")

    if len(single_reports) > 1:
        gen_multi_strip_report(result_all_path, single_reports)
    elif len(single_reports) == 1:
        print(f"单条带钢报告已生成：{single_reports[0]}")

    print("报告生成完成。")





def _parse_bool(s):
    if isinstance(s, bool):
        return s
    if s is None:
        return None
    return str(s).strip().lower() in ("1", "true", "yes", "y")


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser(description="钢带缺陷检测报告生成")
    parser.add_argument("--date", type=str, default=None, help="检测结果日期文件夹 YYYYMMDD")
    parser.add_argument("--conduct-id", type=str, default=None, dest="conduct_id_cli", help="质保书号")
    parser.add_argument(
        "--strip-id",
        type=int,
        default=None,
        dest="strip_id_1based",
        help="仅生成第 N 条带钢（1 起）；省略则生成全部",
    )
    parser.add_argument(
        "--update-info",
        type=str,
        default=None,
        help="true=修改重打模式（不跑分类）；false=先分类再生成报告",
    )
    parser.add_argument(
        "--detect-root",
        type=str,
        default=_DETECT_ROOT_DEFAULT,
        help="detect result 根目录",
    )
    args = parser.parse_args()

    _DETECT_ROOT = os.path.normpath(args.detect_root)

    with open(os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
        rptcfg_raw = yaml.safe_load(f) or {}
    rptcfg = _normalize_rptcfg_for_report(rptcfg_raw)
    updates_up = rptcfg.get("updates_up", [])
    updates_down = rptcfg.get("updates_down", [])
    print_cls = rptcfg.get("print_cls", [])
    class_list = rptcfg.get("class_list", [])
    update_info = rptcfg.get("update_info", False)
    colors = rptcfg.get("colors", {})
    class_labels = rptcfg.get("class_labels", {})
    area_range = rptcfg.get("area_range", [])

    if args.update_info is not None:
        update_info = _parse_bool(args.update_info)

    with open(os.path.join(_PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    anomaly_area_cls_range = config["anomaly_area_cls_range"]
    remove_threshold = config["remove_threshold"]
    steel_length_range = config["steel_length_range"]
    calibrat_cam_id = config["calibrat_cam_id"]
    camrea_id_up = config["camrea_id_up"]
    camrea_id_down = config["camrea_id_down"]

    # 统一幅宽图横坐标换算系数：
    # 尽量使用与检测阶段一致的标准比例（cam*_standard_ratio_x），
    # 否则 fallback 到历史硬编码值，确保脚本可运行。
    try:
        _calib_id = int(float(str(calibrat_cam_id).strip()))
    except Exception:
        _calib_id = 2
    _fukuan_ratio = config.get(f"cam{_calib_id}_standard_ratio_x", None)
    if _fukuan_ratio is None:
        _fukuan_ratio = config.get(f"cam{_calib_id + 1}_standard_ratio_x", None)
    if _fukuan_ratio is None:
        _fukuan_ratio = 0.1575

    cls_model_path = config["cls_model_path"]
    camrea_id_up_cls = config["camrea_id_up_cls"]
    camrea_id_down_cls = config["camrea_id_down_cls"]
    camrea_id_down_report = config["camrea_id_down_report"]
    camrea_id_up_report = config["camrea_id_up_report"]

    strip_only_0based = None
    if args.strip_id_1based is not None:
        if args.strip_id_1based < 1:
            print("[ERROR] --strip-id 须为 >= 1 的整数", file=sys.stderr)
            sys.exit(1)
        strip_only_0based = args.strip_id_1based - 1

    with open(os.path.join(_PROJECT_ROOT, "config", "config0.yaml"), "r", encoding="utf-8") as f:
        config0 = yaml.safe_load(f)
    fukuan0 = [
        config0.get("fukuan_1", 0),
        config0.get("fukuan_2", 0),
        config0.get("fukuan_3", 0),
        config0.get("fukuan_4", 0),
    ]

    explicit_path = (
        args.date is not None
        and str(args.date).strip() != ""
        and args.conduct_id_cli is not None
        and str(args.conduct_id_cli).strip() != ""
    )

    if explicit_path:
        result_all_path = os.path.join(_DETECT_ROOT, args.date.strip(), args.conduct_id_cli.strip())
        conduct_id = args.conduct_id_cli.strip()
        if not os.path.isdir(result_all_path):
            print(
                f"[ERROR] 检测结果目录不存在：\n  {result_all_path}\n"
                "请在「报告中心」选择已有日期与卡号，或确认路径是否正确。",
                file=sys.stderr,
            )
            sys.exit(1)
    elif update_info is False:
        conduct_id = config0["conduct_id"]
        print_cls = class_list
        folders = find_folders_with_id(base_path=_DETECT_ROOT, product_id=conduct_id)
        if len(folders) == 0:
            now = datetime.now()
            today = now.strftime("%Y%m%d")
            yday = (now - timedelta(days=1)).strftime("%Y%m%d")
            print(
                f"[ERROR] 未找到质保书号「{conduct_id}」的检测结果目录。\n"
                f"  已按规则查找：{_DETECT_ROOT}\\{today}\\{conduct_id} "
                f"（及凌晨跨日 {yday} 等）。\n"
                "  请使用「报告中心」选择已有日期，或命令行指定：\n"
                f"    --date YYYYMMDD --conduct-id {conduct_id}",
                file=sys.stderr,
            )
            sys.exit(1)
        result_all_path = folders[0]
    else:
        conduct_id = rptcfg["id"]
        _ = rptcfg["check_cls"]
        time_key = rptcfg["time"]
        result_all_path = os.path.join(_DETECT_ROOT, str(time_key), str(conduct_id))
        if not os.path.isdir(result_all_path):
            print(
                f"[ERROR] 修改模式下列目录不存在：\n  {result_all_path}\n"
                "请先在 rptcfg 或报告中心填写正确的日期与卡号。",
                file=sys.stderr,
            )
            sys.exit(1)

    if update_info is False:
        # 安全闸：分类推理前必须确认“模型类别”与“当前标准类别”兼容（集合一致，顺序可不同）。
        # 不兼容直接退出，避免写出错误类号导致报告统计错位。
        # 优先：按“型号绑定模型”选择（类别配置窗口维护），否则 fallback 到 config.yaml 的 cls_model_path
        model_path = cls_model_path
        try:
            # 当前型号：优先 config0.product_cls；其次 rptcfg.product_cls
            try:
                with open(os.path.join(_PROJECT_ROOT, "config", "config0.yaml"), "r", encoding="utf-8") as f:
                    cfg0 = yaml.safe_load(f) or {}
            except Exception:
                cfg0 = {}
            product_cls = str(cfg0.get("product_cls", "") or "").strip() or str(rptcfg_raw.get("product_cls", "") or "").strip()
            binds = rptcfg_raw.get("cls_model_by_product")
            if product_cls and isinstance(binds, dict):
                b = binds.get(str(product_cls)) or binds.get(str(int(product_cls))) if str(product_cls).isdigit() else None
                if isinstance(b, dict):
                    mp = str(b.get("model_path", "") or "").strip()
                    if mp and os.path.exists(mp):
                        model_path = mp
        except Exception:
            pass
        try:
            from cls_model_registry import compat_and_remap as _compat_and_remap

            cj = os.path.join(os.path.dirname(model_path), "classes.json") if model_path else ""
            model_classes = []
            if cj and os.path.exists(cj):
                try:
                    with open(cj, "r", encoding="utf-8") as f:
                        obj = json.load(f) or {}
                    if isinstance(obj, dict) and "id_to_name" in obj and isinstance(obj["id_to_name"], dict):
                        obj = obj["id_to_name"]
                    if isinstance(obj, dict):
                        for kk in sorted(obj.keys(), key=lambda x: int(str(x))):
                            model_classes.append(str(obj.get(kk, "")).strip())
                except Exception:
                    model_classes = []

            rpt_classes = []
            try:
                for kk in sorted((class_labels or {}).keys(), key=lambda x: int(str(x))):
                    rpt_classes.append(str(class_labels.get(kk, "")).strip())
            except Exception:
                rpt_classes = [str(v).strip() for v in (class_labels or {}).values()]

            if not model_classes:
                print("[ERROR] 当前分类模型缺少 classes.json（类别清单）。为避免错位，禁止分类推理。", file=sys.stderr)
                sys.exit(2)

            ok, remap, diff = _compat_and_remap(model_classes=model_classes, rptcfg_classes=rpt_classes)
            if not ok:
                print(
                    "[ERROR] 分类模型与当前缺陷类别标准不一致，禁止分类推理。\n"
                    f"  缺少：{', '.join(diff.get('missing') or []) or '-'}\n"
                    f"  多出：{', '.join(diff.get('extra') or []) or '-'}\n"
                    "请在 UI 中选择类别一致的模型，或按当前类别重新训练。",
                    file=sys.stderr,
                )
                sys.exit(2)

            # 写 runtime_state remap，供 cls_anomalies 使用（合并写入）
            try:
                from cls_model_registry import write_runtime_remap as _write_runtime_remap

                _write_runtime_remap(
                    os.path.join(_PROJECT_ROOT, "config", "runtime_state.json"),
                    model_path=model_path,
                    model_classes=model_classes,
                    rptcfg_classes=rpt_classes,
                    remap_modelidx_to_rptid_1based=list(remap or []),
                )
            except Exception:
                pass
        except Exception:
            # 保守：若校验模块不可用，直接退出（避免静默错位）
            print("[ERROR] 无法完成分类模型兼容性校验，已中止。", file=sys.stderr)
            sys.exit(2)
        _up_cls_dir = _strip_paths.resolve_roll_cam_subdir(
            result_all_path, str(camrea_id_up_cls), "上表面", "2", "1"
        )
        _dn_cls_dir = _strip_paths.resolve_roll_cam_subdir(
            result_all_path, str(camrea_id_down_cls), "下表面", "3", "4"
        )
        cls_anomalies(model_path, result_all_path, _up_cls_dir)
        print("上表面缺陷分类已完成.")
        cls_anomalies(model_path, result_all_path, _dn_cls_dir)
        print("下表面缺陷分类已完成.")

    standard_area_tables = pd.read_json(
        os.path.join(_PROJECT_ROOT, "table.json"), orient="split"
    )

    print("报告打印中，请等待......")
    strip_card_list = []
    try:
        raw = config0.get("strip_card_list")
        if isinstance(raw, (list, tuple)):
            strip_card_list = [str(x or "").strip() for x in raw]
        else:
            strip_card_list = [
                str(config0.get("strip_card_1", "") or "").strip(),
                str(config0.get("strip_card_2", "") or "").strip(),
                str(config0.get("strip_card_3", "") or "").strip(),
                str(config0.get("strip_card_4", "") or "").strip(),
            ]
    except Exception:
        strip_card_list = []
    gen_all_reports(
        result_all_path,
        fukuan0,
        conduct_id,
        anomaly_area_cls_range,
        remove_threshold,
        steel_length_range,
        calibrat_cam_id,
        camrea_id_up_report,
        camrea_id_down_report,
        updates_up,
        updates_down,
        print_cls,
        class_list,
        update_info,
        standard_area_tables,
        colors,
        class_labels,
        area_range,
        camrea_id_up_list=camrea_id_up,
        strip_only_0based=strip_only_0based,
        fukuan_ratio=_fukuan_ratio,
        rptcfg_raw_snapshot=rptcfg_raw,
        config_snapshot=config,
        config0_snapshot=config0,
        strip_card_list=strip_card_list,
    )

