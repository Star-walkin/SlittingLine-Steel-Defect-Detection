
import numpy as np
import queue
import threading
import time
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import json
import yaml
import socket
import struct
import cv2
import glob
from app.common.function_bank import (
    test_one_image,
    get_one_image_list,
    Consecutive_anomaly_Checker,
    Anomaly_info_List,
    find_folders_with_id,
    fix_json,
    split_multi_strips,
    dummy_context,
    dummy_lock,
    crop_square_from_image_u8,
    DEFECT_PATCH_SIDE,
)
import torch
from datetime import datetime
import traceback
from app.common import speed_monitor
from app.common.speed_monitor import init_monitor, get_monitor, StageTimer

# PatchCore / 局部对比度（与 detectoutline02 一致）
try:
    from models.patchcore_model.online_detector import PatchCoreDetector, InferEngine
    from models.patchcore_model.gradient_defect import detect_defects_by_gradient
    from models.patchcore_model.local_contrast_defect import detect_defects_by_local_contrast
    PATCHCORE_AVAILABLE = True
except ImportError:
    PatchCoreDetector = None
    InferEngine = None
    detect_defects_by_gradient = None
    detect_defects_by_local_contrast = None
    PATCHCORE_AVAILABLE = False


def _get_safe_device():
    """
    安全获取 torch 设备：
    - 在未编译 CUDA 或 CUDA 不可用时，统一回退到 CPU，避免
      'Torch not compiled with CUDA enabled' 异常。
    """
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

# 全局写队列与锁（最小侵入）
write_queue = queue.Queue()   # 全局写任务队列

# 与 detectoutline02 一致：优雅退出、debug 目录去重
shutdown_event = threading.Event()
pause_event = threading.Event()
_created_dirs = set()

# 运行态控制（由主界面写入）：暂停/继续
_RUNTIME_STATE_PATH = os.path.join(_REPO_ROOT, "config", "runtime_state.json")

# 产线状态心跳（由接收端写入）：收到图片即刷新时间戳，UI 读取判断运行/静止
_LINE_HEARTBEAT_PATH = os.path.join(_REPO_ROOT, "config", "line_heartbeat.json")
_hb_lock = threading.Lock()
_hb_last_recv_ts = {}  # cam_id -> epoch seconds


def _read_paused_state() -> bool:
    try:
        with open(_RUNTIME_STATE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        return bool(d.get("paused", False))
    except Exception:
        return False


def _runtime_state_watcher():
    """
    运行态监控线程：读取 runtime_state.json 控制暂停/继续。
    paused=True  -> pause_event.set()
    paused=False -> pause_event.clear()
    """
    last = None
    while not shutdown_event.is_set():
        cur = _read_paused_state()
        if cur != last:
            if cur:
                pause_event.set()
                print("[state] paused=ON（跳过缺陷检测，仅保活接收并保持长度计数）")
            else:
                pause_event.clear()
                print("[state] paused=OFF（恢复缺陷检测）")
            last = cur
        time.sleep(0.25)


def _heartbeat_writer():
    """
    周期性把“最近一次收图时间戳”写到配置目录，供 UI 判定产线运行状态。
    之所以独立线程写文件，是为了避免在 receive_images 高频 IO 影响吞吐。
    """
    last_dump = 0.0
    while not shutdown_event.is_set():
        now = time.time()
        if now - last_dump < 0.5:
            time.sleep(0.1)
            continue
        last_dump = now
        try:
            with _hb_lock:
                per_cam = {str(k): float(v) for k, v in _hb_last_recv_ts.items()}
                last_any = max(per_cam.values()) if per_cam else 0.0
            payload = {"ts": float(last_any), "per_cam": per_cam}
            os.makedirs(os.path.dirname(_LINE_HEARTBEAT_PATH), exist_ok=True)
            tmp = _LINE_HEARTBEAT_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _LINE_HEARTBEAT_PATH)
        except Exception:
            pass


def _line_heartbeat_age_sec() -> float:
    """距离 line_heartbeat.json 中最近一次收图的时间（秒）；读失败或从未收图则视为已静止很久。"""
    try:
        with open(_LINE_HEARTBEAT_PATH, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        ts = float(d.get("ts", 0) or 0)
        if ts <= 0:
            return 1e9
        return max(0.0, time.time() - ts)
    except Exception:
        return 1e9


_line_idle_cfg_cache = None
_line_idle_cfg_mtime = 0.0


def _get_line_idle_catchup_cfg():
    """产线静止追平：是否启用、静止判定秒数、单次最多连处理帧数（带 mtime 缓存）。"""
    global _line_idle_cfg_cache, _line_idle_cfg_mtime
    path = os.path.join(_REPO_ROOT, "config", "config.yaml")
    try:
        mtime = os.path.getmtime(path)
        if _line_idle_cfg_cache is not None and mtime == _line_idle_cfg_mtime:
            return _line_idle_cfg_cache
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        en = bool(cfg.get("line_idle_catchup_enable", True))
        stale = float(cfg.get("line_idle_stale_sec", 2.0) or 2.0)
        mx = int(cfg.get("line_idle_catchup_max_frames", 40) or 40)
        mx = max(1, min(mx, 300))
        stale = max(0.3, stale)
        _line_idle_cfg_cache = (en, stale, mx)
        _line_idle_cfg_mtime = mtime
        return _line_idle_cfg_cache
    except Exception:
        return (True, 2.0, 40)


# per-camera lock 防止并发修改内存 list 结构（按相机一级锁）
# 注意：在 main 中我们会根据 num_cams 初始化 cam_locks
cam_locks = None  # 将在 main 初始化后替换为 list of threading.Lock()


def get_strip_count_and_fukuan(config0):
    """
    统一读取 1~4 条带钢配置（向后兼容）
    返回: (strip_count, fukuan_list)
    """
    raw_count = int(config0.get("strip_count", 3))
    strip_count = min(4, max(1, raw_count))

    # 优先新结构 fukuan_list
    fukuan_list = config0.get("fukuan_list")
    if isinstance(fukuan_list, list) and len(fukuan_list) >= strip_count:
        vals = [float(v) for v in fukuan_list[:strip_count]]
        if any(v <= 0 for v in vals):
            print(f"[config][warn] fukuan_list 前{strip_count}项存在非正数: {vals}")
        return strip_count, vals

    # 兼容旧结构 fukuan_1..4
    vals = [float(config0.get(f"fukuan_{i}", 0.0)) for i in range(1, 5)]
    vals = vals[:strip_count]
    if any(v <= 0 for v in vals):
        print(f"[config][warn] fukuan_1..4 前{strip_count}项存在非正数: {vals}")
    return strip_count, vals

# writer 线程：负责所有 JSON / fukuan 原子写入（单线程写文件，避免竞争）
def writer_loop():
    """
    write_queue 中任务样例（dict）:
    {"type":"save_anomaly", "info_process":..., "relpath":..., "value": ...}
    {"type":"append_fukuan", "fpath": ..., "value": ...}
    {"type":"save_history_id", "info_process":..., "value": ...}
    """

    import traceback
    batch = []
    last_flush = time.time()
    FLUSH_INTERVAL = 0.08
    MAX_BATCH = 20

    # -------------------------------
    # 核心修改：使用 shutdown_event 控制退出条件
    # -------------------------------
    while not shutdown_event.is_set() or not write_queue.empty():
        try:
            task = write_queue.get(timeout=FLUSH_INTERVAL)

            if task is None:
                # 外部发来的退出信号
                break

            batch.append(task)

            if len(batch) >= MAX_BATCH:
                _process_batch(batch)
                batch = []
                last_flush = time.time()

        except queue.Empty:
            # 正常 timeout，检查是否需要 flush
            if batch and (time.time() - last_flush) >= FLUSH_INTERVAL:
                try:
                    _process_batch(batch)
                except Exception:
                    traceback.print_exc()
                batch = []
                last_flush = time.time()
            continue

        except Exception:
            traceback.print_exc()
            continue

    # -------------------------------
    # 程序已要关闭，批量 flush 最后剩下的任务
    # -------------------------------
    if batch:
        try:
            _process_batch(batch)
        except Exception:
            traceback.print_exc()

    print("[writer] writer_loop 已安全退出（已执行最终 flush）")


def _process_batch(batch):
    """
    逐项处理 batch 中任务。写文件动作尽量原子完成（tmp -> replace）。
    """
    for task in batch:
        try:
            ttype = task.get("type")
            if ttype == "save_anomaly":
                info_process = task["info_process"]
                relpath = task["relpath"]
                value = task["value"]
                # 直接调用你原有的写方法（保证 writer 线程唯一性）
                info_process.save_anomaly_info(relpath, value)

            elif ttype == "append_fukuan":
                fpath = task["fpath"]
                val = task["value"]
                # append 写入：尽量原子（tmp->replace），但 Windows 下若 UI 正在读目标文件，
                # os.replace 可能抛 WinError 5（拒绝访问）。这里做短重试+回退，保证 writer 不“卡死”。
                try:
                    if os.path.exists(fpath):
                        with open(fpath, "r", encoding="utf-8") as fr:
                            data = json.load(fr)
                    else:
                        data = []
                except Exception:
                    data = []
                data.append(float(val))
                tmp = fpath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, ensure_ascii=False, indent=2)
                replaced = False
                last_err = None
                for _ in range(5):
                    try:
                        os.replace(tmp, fpath)
                        replaced = True
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.05)
                if not replaced:
                    # 回退：直接覆盖写入（可能让读端短暂看到半截 JSON，但 UI 端已做重试容错）
                    try:
                        with open(fpath, "w", encoding="utf-8") as fw:
                            json.dump(data, fw, ensure_ascii=False, indent=2)
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                    except Exception:
                        # 保留 tmp 以便排查
                        raise last_err

            elif ttype == "save_history_id":
                info_process = task["info_process"]
                value = task["value"]
                info_process.save_anomaly_info("history_image_id", value)

            else:
                # 未知任务类型：忽略或记录
                print("writer_loop: unknown task", ttype)
        except Exception as e:
            print("writer task exception:", e)
            # 继续下一个任务



def init_detect(conduct_id, result_path_all, Consecutive_thres_num, cam_index):
    """
    PatchCore 在线检测初始化（已移除其它模型回退分支）。
    结果目录与线下一致：result_path_all / {cam_index+1} / strip_*
    """
    with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as f:
        config0 = yaml.safe_load(f)
    _, fukuan0 = get_strip_count_and_fukuan(config0)

    with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    calibrat_cam_id = config["calibrat_cam_id"]
    steel_real_y0 = config["steel_real_y0"]
    seg_anomaly_thres = config[f"cam{cam_index + 1}_seg_anomaly_thres"]
    standard_ratio_x = config[f"cam{cam_index + 1}_standard_ratio_x"]
    patchcore_k = float(config.get(f"cam{cam_index + 1}_patchcore_k", 3.0))
    use_gradient_detection = config.get(f"cam{cam_index + 1}_use_gradient_detection", False)
    grad_threshold = float(config.get(f"cam{cam_index + 1}_grad_threshold", 1.5))
    blur_ksize = int(config.get(f"cam{cam_index + 1}_blur_ksize", 5))
    edge_crop = int(config.get(f"cam{cam_index + 1}_edge_crop", 20))
    bg_ksize = int(config.get(f"cam{cam_index + 1}_bg_ksize", 101))
    diff_threshold = float(config.get(f"cam{cam_index + 1}_diff_threshold", 1.0))
    patchcore_edge_soft_border = int(config.get(f"cam{cam_index + 1}_patchcore_edge_soft_border", 20))
    patchcore_edge_strength = float(config.get(f"cam{cam_index + 1}_patchcore_edge_strength", 1.0))
    patchcore_edge_weight_profile = config.get(
        f"cam{cam_index + 1}_patchcore_edge_weight_profile", "ease_out_cubic"
    )
    cut_ratio = config[f"cam{cam_index + 1}_cut_ratio"]
    img_size = config[f"cam{cam_index + 1}_img_size"]
    cam_y_times = config[f"cam{cam_index + 1}_times"]

    cam_folder = f"{cam_index + 1}"
    result_path = os.path.join(result_path_all, cam_folder)
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "fukuan0.json"), "w", encoding="utf-8") as f:
        json.dump(fukuan0, f, ensure_ascii=False, indent=4)

    history_id_path = os.path.join(result_path, "history_image_id.json")
    if not os.path.exists(history_id_path):
        json.dump({"last_id": 0}, open(history_id_path, "w", encoding="utf-8"), indent=4)

    for i in range(len(fukuan0)):
        strip_dir = os.path.join(result_path, f"strip_{i + 1}")
        os.makedirs(strip_dir, exist_ok=True)
        os.makedirs(os.path.join(strip_dir, "defect_images"), exist_ok=True)
        for name in ["image_anomaly_center.json", "image_anomaly_area.json", "fukuan.json"]:
            fpath = os.path.join(strip_dir, name)
            if not os.path.exists(fpath):
                json.dump([], open(fpath, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    M = None
    in_channels_detected = 1
    standard_ratio_y = standard_ratio_x * cam_y_times

    if PATCHCORE_AVAILABLE:
        try:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            _weights_root_name = config.get("patchcore_weights_root", "weights")
            patchcore_root = os.path.join(
                root_dir, "models", "patchcore_model", _weights_root_name, "image_data_patchcore_0228"
            )
            cam_name = f"CAM{cam_index + 1}"
            memory_path = os.path.join(patchcore_root, cam_name, "patchcore_memory.npz")
            if os.path.isfile(memory_path):
                print(f"[CAM{cam_index + 1}] 使用 PatchCoreDetector: {memory_path}")
                _device = _get_safe_device()
                M = PatchCoreDetector(
                    memory_path=memory_path,
                    conduct_id=conduct_id,
                    fukuan0=fukuan0,
                    cut_ratio=cut_ratio,
                    img_size=img_size,
                    seg_anomaly_thres=seg_anomaly_thres,
                    standard_ratio_x=standard_ratio_x,
                    standard_ratio_y=standard_ratio_y,
                    steel_real_y0=steel_real_y0,
                    device=_device,
                    patchcore_k=patchcore_k,
                    use_gradient_detection=use_gradient_detection,
                    grad_threshold=grad_threshold,
                    blur_ksize=blur_ksize,
                    edge_crop=edge_crop,
                    bg_ksize=bg_ksize,
                    diff_threshold=diff_threshold,
                    patchcore_edge_soft_border=patchcore_edge_soft_border,
                    patchcore_edge_strength=patchcore_edge_strength,
                    patchcore_edge_weight_profile=patchcore_edge_weight_profile,
                )
            else:
                raise FileNotFoundError(
                    f"[CAM{cam_index + 1}] 未找到 PatchCore 权重文件：{memory_path}\n"
                    f"请检查：models/patchcore_model/{_weights_root_name}/image_data_patchcore_0228/{cam_name}/patchcore_memory.npz"
                )
        except Exception as e:
            raise RuntimeError(f"[CAM{cam_index + 1}] PatchCoreDetector 初始化失败: {e}") from e

    F = get_one_image_list(cut_ratio=cut_ratio, standard_ratio_x=standard_ratio_x, fukuan0=fukuan0)
    Checker = Consecutive_anomaly_Checker(thres_num=Consecutive_thres_num, calibrate_cam=calibrat_cam_id)
    info_process = Anomaly_info_List(filepath=result_path)

    print(f"[OK] 相机 {cam_index + 1} 初始化完成（输入通道数={in_channels_detected}）。")
    return M, F, Checker, info_process, result_path, in_channels_detected



# ------------------- 检测单条带钢（与 detectoutline02.detect 对齐） -------------------
def detect(img, new_image_id, cam_id, strip_id, M, F, Checker, info_process, save_root,
           Consecutive_Check, image_anomaly_center_list, image_anomaly_area_list, fukuan_value,
           model_channels=1):
    """
    与 detectoutline02.detect 一致：单帧序号 new_image_id 对该帧内各条带共用；
    save_root 为相机根目录（数字文件夹 1~4）。
    """
    global _created_dirs

    strip_folder = os.path.join(save_root, f"strip_{strip_id + 1}")
    os.makedirs(strip_folder, exist_ok=True)
    defect_dir = os.path.join(strip_folder, "defect_images")
    os.makedirs(defect_dir, exist_ok=True)

    key = (strip_folder, defect_dir)
    if key not in _created_dirs:
        os.makedirs(defect_dir, exist_ok=True)
        _created_dirs.add(key)

    img_for_inference = img
    if model_channels == 3:
        if len(img.shape) == 2:
            img_for_inference = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    with StageTimer(cam_id, "infer"):
        test_cut, rec_cut, amap_cut = M.detect_ano(img_for_inference)

    # Debug 用「原始条带」：与 detect_ano 内部的拍平/分块/tile 无关，避免 debug 里看起来像全黑或假原图
    _dbg_orig_strip = np.asarray(img)
    if _dbg_orig_strip.ndim == 2:
        _dbg_orig_strip_bgr = cv2.cvtColor(_dbg_orig_strip, cv2.COLOR_GRAY2BGR)
    else:
        _dbg_orig_strip_bgr = _dbg_orig_strip.copy()

    # ================= Debug 可视化（speed_monitor.DEBUG_IO=True 时才写盘）=====
    if speed_monitor.DEBUG_IO:
        try:
            debug_dir = os.path.join(strip_folder, "debug_visuals")
            os.makedirs(debug_dir, exist_ok=True)
            if detect_defects_by_local_contrast is not None:
                Hm, Wm = amap_cut.shape
                locator = getattr(M, "_locator", M)
                ec = max(0, min(getattr(locator, "edge_crop", 20), Hm // 2 - 1, Wm // 2 - 1))
                valid_mask = np.zeros((Hm, Wm), dtype=np.uint8)
                valid_mask[ec : Hm - ec, ec : Wm - ec] = 255
                bg_k = getattr(locator, "bg_ksize", 101)
                diff_th = getattr(locator, "diff_threshold", 1.0)
                from models.patchcore_model.local_contrast_defect import debug_local_contrast_visualization

                debug_path = os.path.join(debug_dir, f"debug_{new_image_id}.png")
                amap_raw_dbg = getattr(M, "_last_amap_raw", None)
                debug_local_contrast_visualization(
                    heatmap=amap_cut.astype(np.float32),
                    valid_mask=valid_mask,
                    bg_ksize=bg_k,
                    diff_threshold=diff_th,
                    save_path=debug_path,
                    original_img=_dbg_orig_strip_bgr,
                    heatmap_raw=amap_raw_dbg.astype(np.float32) if amap_raw_dbg is not None else None,
                )
            else:
                Hm, Wm = amap_cut.shape
                orig_vis = cv2.resize(
                    _dbg_orig_strip_bgr, (Wm, Hm), interpolation=cv2.INTER_LINEAR
                )
                norm_base = max(amap_cut.max(), 1.0)
                amap_norm = (amap_cut / norm_base * 255).astype(np.uint8)
                amap_color = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
                max_val = amap_cut.max()
                cv2.putText(amap_color, f"Max: {max_val:.4f}", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                combined = np.hstack([orig_vis, amap_color, orig_vis])
                cv2.imwrite(os.path.join(debug_dir, f"debug_{new_image_id}.png"), combined)
        except Exception as e:
            print(f"Vis error: {e}")
    # ================= Debug 可视化结束 ==========================================

    with StageTimer(cam_id, "locate"):
        anomay_state, center_coords, area_list = M.obtain_anomaly_location(
            amap_cut, test_cut, defect_dir, new_image_id, fukuan_value,
            original_strip_for_crop=img,
        )

    lock = None
    if cam_locks is not None:
        lock = cam_locks[cam_id]

    if lock is not None:
        with lock:
            image_anomaly_center_list.append(center_coords)
            image_anomaly_area_list.append(area_list)
            rel_center = os.path.join(strip_folder, "image_anomaly_center.json")
            rel_area = os.path.join(strip_folder, "image_anomaly_area.json")
            write_queue.put({
                "type": "save_anomaly",
                "info_process": info_process,
                "relpath": rel_center,
                "value": image_anomaly_center_list
            })
            write_queue.put({
                "type": "save_anomaly",
                "info_process": info_process,
                "relpath": rel_area,
                "value": image_anomaly_area_list
            })
    else:
        info_process.save_anomaly_info(f"image_anomaly_center.json", image_anomaly_center_list)
        info_process.save_anomaly_info(f"image_anomaly_area.json", image_anomaly_area_list)

    fukuan_path = os.path.join(os.path.join(save_root, f"strip_{strip_id + 1}"), "fukuan.json")
    write_queue.put({
        "type": "append_fukuan",
        "fpath": fukuan_path,
        "value": float(fukuan_value)
    })

    if center_coords and area_list and area_list[-1] == 10000:
        save_path = os.path.join(
            defect_dir,
            f"{center_coords[0][0]}_{center_coords[0][1]}_10000_strip{strip_id + 1}_{new_image_id}.png"
        )
        Hs, Ws = img.shape[:2]
        patch = crop_square_from_image_u8(img, Ws // 2, Hs // 2, DEFECT_PATCH_SIDE)
        cv2.imwrite(save_path, patch)

    calibrate_cam_id = Checker.calibrate_cam
    cam_idx = cam_id + 1
    if cam_idx == calibrate_cam_id and Consecutive_Check == 0:
        with (cam_locks[cam_id] if cam_locks is not None else dummy_lock()):
            Checker.add_number(anomay_state)

    return new_image_id

# 图像处理函数（可选调试入口；逻辑与 detectoutline02 一致：一帧一个 new_id，各条带共用）
def process_image(image_data, history_image_id_list, index, cam_id,
                  M, F, Consecutive_Checker, anomaly_info_process,
                  anomaly_save_root, Consecutive_Check,
                  image_anomaly_center_list, image_anomaly_area_list, fukuan_list,
                  model_channels=1):
    nparr = np.frombuffer(image_data, np.uint8)
    gray_image = nparr.reshape((4096, 4096))
    img = np.stack((gray_image,) * 3, axis=-1)

    if fukuan_list and len(fukuan_list[cam_id]) > 0:
        fukuan_est = [np.mean(fukuan_list[cam_id][i]) if len(fukuan_list[cam_id][i]) > 0 else f
                      for i, f in enumerate(F.fukuan0)]
    else:
        fukuan_est = F.fukuan0

    strip_imgs, measured_widths_mm = split_multi_strips(
        img, fukuan_list_mm=fukuan_est, standard_ratio_x=F.standard_ratio_x
    )
    fukuan_list[cam_id] = measured_widths_mm

    last_id = history_image_id_list[cam_id]
    new_id = last_id + 1
    history_image_id_list[cam_id] = new_id
    write_queue.put({
        "type": "save_history_id",
        "info_process": anomaly_info_process,
        "value": {"last_id": new_id},
    })

    for i, strip in enumerate(strip_imgs):
        detect(
            strip,
            new_id,
            cam_id,
            i,
            M, F, Consecutive_Checker, anomaly_info_process,
            anomaly_save_root,
            Consecutive_Check,
            image_anomaly_center_list[cam_id][i],
            image_anomaly_area_list[cam_id][i],
            measured_widths_mm[i],
            model_channels=model_channels,
        )
        print(f"Thread {cam_id + 1} processing image {index}... (strip {i + 1})")

    print(f"Thread {cam_id+1} processing image {index}...")



# 接收图像的线程
def get_host_ip():  # 获得ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 初始化创建（ipv4和UDP协议）
        s.connect(('127.0.0.1', 80))  # 先自己socket通讯注册一个ip和端口（随便写一个ipv4的和端口）
        ip = s.getsockname()[0]  # 取得通讯的ip
    finally:
        s.close()
    return ip

def receive_images(sock, cam_id, image_queue):
    """
    与 detect_anomalies_new.py 一致：
    单次阻塞 accept → 单连接内循环收包（4 字节小端长度 + payload），与 new 的协议相同。
    额外在循环首检查 shutdown_event，并在退出时关闭 conn，便于主程序关闭监听套接字后结束线程。
    """
    index = 0
    conn = None
    try:
        conn, address = sock.accept()
        print(f"客户端{cam_id + 1}已连接:", address)
        while True:
            if shutdown_event.is_set():
                break
            length_bytes = conn.recv(4)
            if not length_bytes:
                break
            length = struct.unpack("I", length_bytes)[0]

            image_data = b""
            while len(image_data) < length:
                packet = conn.recv(length - len(image_data))
                if not packet:
                    break
                image_data += packet

            try:
                image_queue.put((image_data, index))
                # ---- 产线心跳：收到一帧就刷新（只更新内存；落盘由 _heartbeat_writer 周期写）----
                try:
                    with _hb_lock:
                        _hb_last_recv_ts[int(cam_id)] = time.time()
                except Exception:
                    pass
                # ---- 记录输入帧率 ----
                _mon_r = get_monitor()
                if _mon_r:
                    _mon_r.record_input(cam_id)
            except Exception as e:
                print(f"Receive queue put error: {e}")
            index += 1
    except OSError as e:
        # 监听套接字被关闭时 accept/recv 可能抛出
        print(f"receive_images OSError cam={cam_id}: {e}")
    except Exception as e:
        print(f"Receive error: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    print(f"receive_images 线程退出 cam={cam_id}")

# ------------------------- GPU图像处理线程（切分/检测与 detectoutline02.worker 对齐） -------------------------
def worker(image_queue, cam_id, history_image_id_list, M, F, Checker, info_process, save_root,
           Consecutive_Check, image_anomaly_center_list, image_anomaly_area_list, fukuan_list,
           model_channels=1, infer_engine=None, lite_cam_ids=frozenset()):
    """
    image_queue: 每项 (data_bytes, idx)；若 data_bytes is None => 退出
    history_image_id_list: 长度=num_cams 的列表，每相机一个递增的 last_id（与 detectoutline02 一致）
    infer_engine: InferEngine 实例，若非 None 则在专用 GPU 线程中处理推理（锁外并行）
    lite_cam_ids: 轻量模式相机集合（0-based），这些相机只测量幅宽，跳过全量缺陷检测
    """
    SPLIT_RECALC_EVERY_N = 10  # 每 10 帧重新计算一次条带切分，其余帧复用缓存坐标
    last_splits = None
    last_measured_widths = None

    def cut_by_splits(gray_img_2d, splits):
        return [gray_img_2d[:, L:R] for (L, R) in splits]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_name = f"CAM{cam_id + 1}"
    processed_count = 0
    print(f"Worker {local_name} 启动 (Model Channels: {model_channels})...")

    def process_one_image(image_data, idx):
        nonlocal processed_count, last_splits, last_measured_widths
        # ---- 队列水位监控 ----
        _mon_w = get_monitor()
        if _mon_w:
            _mon_w.record_qsize(cam_id, image_queue.qsize())

        # ---- 解码阶段计时 ----
        _t_decode = time.perf_counter()
        nparr = np.frombuffer(image_data, np.uint8)
        gray = None
        if nparr.size == 4096 * 4096:
            gray = nparr.reshape((4096, 4096))
        else:
            try:
                decoded_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if decoded_img is not None:
                    gray = cv2.resize(decoded_img, (4096, 4096)) if decoded_img.shape != (4096, 4096) else decoded_img
            except Exception:
                pass
        if _mon_w:
            _mon_w.record_stage(cam_id, "decode", time.perf_counter() - _t_decode)

        if gray is None:
            image_queue.task_done()
            return

        # 整帧（切分、帧号、detect）放在同一相机锁内，避免双 worker 与 detectoutline02 单 worker 语义一致且 ID 不重复
        strip_tasks = []  # 在 cam_lock 外初始化，防止 continue 路径后未定义
        lock = cam_locks[cam_id] if cam_locks is not None else dummy_context()
        with (lock if hasattr(lock, "__enter__") else dummy_context()):
            try:
                fukuan_est = [
                    np.mean(fukuan_list[i]) if len(fukuan_list[i]) > 0 else F.fukuan0[i]
                    for i in range(len(F.fukuan0))
                ]
            except Exception:
                fukuan_est = F.fukuan0

            need_recalc = (last_splits is None) or (processed_count % SPLIT_RECALC_EVERY_N == 0)
            if need_recalc:
                try:
                    with StageTimer(cam_id, "split"):
                        strip_imgs, measured_widths_mm, splits = split_multi_strips(
                            gray,
                            fukuan_list_mm=fukuan_est,
                            standard_ratio_x=F.standard_ratio_x,
                            cam_id=cam_id,
                            use_thumbnail=True,   # 缩略图定位加速（~3x）
                        )
                    last_splits = splits
                    last_measured_widths = measured_widths_mm
                except Exception:
                    if last_splits:
                        strip_imgs = cut_by_splits(gray, last_splits)
                        measured_widths_mm = last_measured_widths
                    else:
                        image_queue.task_done()
                        return
            else:
                strip_imgs = cut_by_splits(gray, last_splits)
                measured_widths_mm = last_measured_widths if last_measured_widths else fukuan_est

            expected_num = len(F.fukuan0)
            actual_num = len(strip_imgs)
            if actual_num != expected_num:
                print(f"[{local_name}][warn] split条数不一致 expected={expected_num}, actual={actual_num}")
            iter_num = min(actual_num, expected_num, len(image_anomaly_center_list), len(image_anomaly_area_list))
            if iter_num <= 0:
                image_queue.task_done()
                return

            # 与 detectoutline02 一致：每帧递增一次全局序号，本帧各条带共用同一 new_id
            last_id = history_image_id_list[cam_id]
            new_id = last_id + 1
            history_image_id_list[cam_id] = new_id
            write_queue.put({
                "type": "save_history_id",
                "info_process": info_process,
                "value": {"last_id": new_id},
            })

            # split_vis: 仅在 speed_monitor.DEBUG_IO=True 时写盘。
            # 为避免 OOM：对可视化图做降采样（不影响检测本身，仅影响调试文件）。
            if speed_monitor.DEBUG_IO:
                _t_dbg = time.perf_counter()
                try:
                    vis_dir = os.path.join(save_root, "split_vis")
                    os.makedirs(vis_dir, exist_ok=True)

                    H_full, W_full = gray.shape[0], gray.shape[1]
                    split_vis_max_side = 2048
                    mx = max(H_full, W_full)
                    scale = min(1.0, float(split_vis_max_side) / float(mx))

                    if scale < 1.0:
                        nw = max(1, int(round(W_full * scale)))
                        nh = max(1, int(round(H_full * scale)))
                        gray_vis = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
                        sx = nw / float(W_full)
                        sy = nh / float(H_full)
                        thick = max(1, int(round(3 * scale)))
                        font_scale = max(0.4, float(1.2 * scale))
                        text_thick = max(1, int(round(2 * scale)))
                        line_thick = max(1, int(round(1 * scale)))
                    else:
                        gray_vis = gray
                        sx = sy = 1.0
                        thick = 3
                        font_scale = 1.2
                        text_thick = 2
                        line_thick = 1

                    H_vis, W_vis = gray_vis.shape[0], gray_vis.shape[1]
                    vis = cv2.cvtColor(gray_vis, cv2.COLOR_GRAY2BGR)

                    cut_ratio = getattr(M, "cut_ratio", 3)
                    strip_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    for si, (L, R) in enumerate(last_splits):
                        color = strip_colors[si % len(strip_colors)]
                        Li, Ri = int(L * sx), int(R * sx)
                        cv2.rectangle(vis, (Li, 0), (Ri, H_vis - 1), color, thick)
                        for j in range(1, cut_ratio):
                            y = int(H_vis * j / cut_ratio)
                            cv2.line(vis, (Li, y), (Ri, y), color, line_thick)
                        cv2.putText(
                            vis,
                            f"strip{si+1}",
                            (Li + max(4, int(10 * sx)), max(16, int(40 * sy))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            text_thick,
                            cv2.LINE_AA,
                        )

                    out_path = os.path.join(vis_dir, f"frame_{new_id:06d}.png")
                    cv2.imwrite(out_path, vis)
                    del vis
                except Exception as e_vis:
                    print(f"[{local_name}] split_vis error: {e_vis}")
                _mon_dbg = get_monitor()
                if _mon_dbg:
                    _mon_dbg.record_stage(cam_id, "debug_io", time.perf_counter() - _t_dbg)

            # 收集本帧各条带的任务参数（cam_lock 内仅做 CPU 准备）
            strip_tasks = []
            for i in range(iter_num):
                strip_tasks.append((
                    strip_imgs[i],
                    new_id,
                    i,
                    measured_widths_mm[i] if i < len(measured_widths_mm) else fukuan_est[i],
                ))

        # ---- 暂停 / 轻量模式：只写幅宽（并保持 history_image_id 继续递增），跳过缺陷检测 ----
        # 暂停时不杀发送端/不关连接，只是让接收端不做缺陷推理；长度从历史继续走。
        if pause_event.is_set() or (cam_id in lite_cam_ids):
            for _, nid, sid, fw in strip_tasks:
                fukuan_path = os.path.join(save_root, f"strip_{sid + 1}", "fukuan.json")
                write_queue.put({"type": "append_fukuan", "fpath": fukuan_path, "value": float(fw)})
            processed_count += 1
            image_queue.task_done()
            _mon_p = get_monitor()
            if _mon_p:
                _mon_p.record_processed(cam_id)
            return

        # ---- 推理阶段：在 cam_lock 外通过 InferEngine 执行，允许另一个 worker 同步做 CPU 预处理 ----
        if infer_engine is not None:
            # 向 InferEngine 提交所有条带任务，立即获得 Future（非阻塞）
            futures = []
            for strip_np, nid, sid, fw in strip_tasks:
                future = infer_engine.submit(
                    detect,
                    strip_np, nid, cam_id, sid,
                    M, F, Checker, info_process, save_root, Consecutive_Check,
                    image_anomaly_center_list[sid],
                    image_anomaly_area_list[sid],
                    fw,
                    model_channels=model_channels,
                )
                futures.append(future)
            # 等待本帧所有条带推理完成
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERR] {local_name} InferEngine error idx={idx}: {e}")
                    traceback.print_exc()
        else:
            # 回退路径：直接调用（不使用 InferEngine）
            for strip_np, nid, sid, fw in strip_tasks:
                try:
                    detect(
                        strip_np, nid, cam_id, sid,
                        M, F, Checker, info_process, save_root, Consecutive_Check,
                        image_anomaly_center_list[sid],
                        image_anomaly_area_list[sid],
                        fw,
                        model_channels=model_channels,
                    )
                except Exception as e:
                    print(f"[ERR] {local_name} detect error strip{sid} idx={idx}: {e}")
                    traceback.print_exc()

        processed_count += 1

        image_queue.task_done()
        # ---- 记录本帧处理完成 ----
        _mon_p = get_monitor()
        if _mon_p:
            _mon_p.record_processed(cam_id)
    try:
        while not shutdown_event.is_set():
            try:
                item = image_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            image_data, idx = item
            if image_data is None:
                break

            process_one_image(image_data, idx)

            _exit_worker = False
            en, stale_sec, mx = _get_line_idle_catchup_cfg()
            if en and (not pause_event.is_set()) and (not shutdown_event.is_set()):
                bn = 0
                while bn < mx and (not shutdown_event.is_set()):
                    if pause_event.is_set():
                        break
                    if _line_heartbeat_age_sec() <= stale_sec:
                        break
                    try:
                        item_n = image_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item_n is None:
                        _exit_worker = True
                        break
                    idn, ixn = item_n
                    if idn is None:
                        _exit_worker = True
                        break
                    process_one_image(idn, ixn)
                    bn += 1
            if _exit_worker:
                break


    except Exception as e:
        print(f"[ERR] worker unexpected error cam={cam_id}: {e}")
        traceback.print_exc()

    print(f"[exit] worker(cam={cam_id}) 已退出")

def run_online(cwd_base_result=None):
    """
    启动函数：负责创建 sockets / threads / writer 等，并在 KeyboardInterrupt 时优雅退出。
    cwd_base_result: 如果你已有 result_path_all，可传入；否则会按日期/ID 创建。
    """
    # 读取配置（保持你原有读取行为）
    with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as f:
        config0 = yaml.safe_load(f)
    conduct_id = config0["conduct_id"]
    _, fukuan0 = get_strip_count_and_fukuan(config0)
    num_strips = len(fukuan0)

    # 结果路径
    if cwd_base_result:
        base_result_path = cwd_base_result
    else:
        base_result_path = os.path.join(os.path.join(_REPO_ROOT, "detect result"),
                                        datetime.now().strftime("%Y%m%d"),
                                        f"{conduct_id}")
    os.makedirs(base_result_path, exist_ok=True)

    # 写入 config0 快照：供报告中心/修改界面直接读取带钢卡号（无需先生成报告）
    try:
        snap0 = os.path.join(base_result_path, "config0_snapshot.yaml")
        with open(snap0, "w", encoding="utf-8") as f:
            yaml.dump(dict(config0 or {}), f, allow_unicode=True)
    except Exception:
        pass

    # 读取其它配置
    with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    Consecutive_Check = config["Consecutive_Check"]
    Consecutive_thres_num = config["Consecutive_thres_num"]
    lite_cam_ids = frozenset(int(x) for x in config.get("lite_cam_ids", []))
    if lite_cam_ids:
        print(f"[config] 轻量模式相机（0-based）: {sorted(lite_cam_ids)}，这些相机只测幅宽，跳过缺陷检测")

    # 剪裁/缺陷调试写盘：split_vis/ + 各 strip 下 debug_visuals/（大图，磁盘与吞吐压力大）
    _debug_io = bool(config.get("debug_io", False))
    speed_monitor.set_debug_io(_debug_io)
    if _debug_io:
        print("[config] debug_io=ON：将写入每相机 save_root/split_vis/ 与 strip_*/debug_visuals/")

    ports = [8885, 8886, 8887, 8888]
    num_cams = len(ports)

    # ---- 启动速度监控器（终端周期性打印：各相机接收速率、处理速率、队列、分阶段耗时）----
    _rep_sec = float(config.get("speed_report_interval_sec", 5.0))
    _speed_mon = init_monitor(num_cams=num_cams, report_interval_sec=_rep_sec)
    print(
        f"[SpeedMonitor] 已启动：每 {_speed_mon.report_interval_sec:.1f}s 打印一次 "
        f"各相机「本周期接收帧数/fps、处理帧数/fps、队列水位、累计recv/done、decode/split/infer/locate 均耗」"
        f"（DEBUG_IO={'ON' if speed_monitor.DEBUG_IO else 'OFF'}）"
    )

    # 初始化锁、队列、内存结构
    global cam_locks
    # RLock：detect() 内与 worker 可能对同一相机嵌套加锁，避免死锁
    cam_locks = [threading.RLock() for _ in range(num_cams)]
    image_queues = [queue.Queue(maxsize=200) for _ in range(num_cams)]  # 给队列一个 maxsize 避免无限制堆积

    image_anomaly_center_list = [[[] for _ in range(num_strips)] for _ in range(num_cams)]
    image_anomaly_area_list = [[[] for _ in range(num_strips)] for _ in range(num_cams)]
    fukuan_list = [[[] for _ in range(num_strips)] for _ in range(num_cams)]
    history_image_id_list = [0 for _ in range(num_cams)]
    for _ci in range(num_cams):
        _hid = os.path.join(base_result_path, str(_ci + 1), "history_image_id.json")
        try:
            with open(_hid, "r", encoding="utf-8") as _hf:
                _raw = json.load(_hf)
            if isinstance(_raw, dict):
                history_image_id_list[_ci] = int(_raw.get("last_id", 0))
            else:
                history_image_id_list[_ci] = int(_raw) if _raw is not None else 0
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            pass

    # 启动单个 writer（全局唯一）
    writer_t = threading.Thread(target=writer_loop, daemon=True)
    writer_t.start()

    # 启动运行态监控（暂停/继续）
    state_t = threading.Thread(target=_runtime_state_watcher, daemon=True)
    state_t.start()

    # 启动产线心跳写盘（运行/静止判定）
    hb_t = threading.Thread(target=_heartbeat_writer, daemon=True)
    hb_t.start()

    # 保存线程 / sockets / engines 引用以便 join/close
    recv_threads = []
    worker_threads = []
    listen_sockets = []
    infer_engines = []

    # 为每个相机做初始化并启动接收/worker
    for cam_id, port in enumerate(ports):
        # init per-camera detector (尽量只初始化一次)
        M, F, Checker, info_process, save_root, in_channels_detected = init_detect(
            conduct_id, base_result_path, Consecutive_thres_num, cam_id
        )

        # listen socket（与 detect_anomalies_new.py 一致：绑定本机 IP + 端口，无 listen 超时）
        ip_port = (get_host_ip(), port)
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(ip_port)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.listen(5)
        listen_sockets.append(listen_sock)
        print(f"[listen] 相机{cam_id + 1} 监听 {ip_port[0]}:{port} 中...")

        # start receive thread
        t_recv = threading.Thread(target=receive_images, args=(listen_sock, cam_id, image_queues[cam_id]), daemon=True)
        t_recv.start()
        recv_threads.append(t_recv)

        # 为本相机创建专用 GPU 推理线程（避免多 worker 线程抢 CUDA 上下文）
        cam_infer_engine = InferEngine(cam_id) if InferEngine is not None else None
        if cam_infer_engine is not None:
            infer_engines.append(cam_infer_engine)

        # start worker pool for this camera (2 workers as before)
        for _ in range(2):
            t_w = threading.Thread(
                target=worker,
                args=(
                    image_queues[cam_id], cam_id, history_image_id_list,
                    M, F, Checker, info_process, save_root, Consecutive_Check,
                    image_anomaly_center_list[cam_id], image_anomaly_area_list[cam_id], fukuan_list[cam_id],
                    in_channels_detected,
                ),
                kwargs={"infer_engine": cam_infer_engine, "lite_cam_ids": lite_cam_ids},
                daemon=True
            )
            t_w.start()
            worker_threads.append(t_w)

    print("\n[run] 所有相机线程已启动，系统运行中...\n")

    # 主线程阻塞等待退出信号（Ctrl+C）
    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[warn] 收到关闭信号，正在安全退出...")
        shutdown_event.set()

    # ---- 开始优雅关闭流程 ----
    print("[shutdown] 开始关闭：停止接收/通知 worker/flush writer...")

    # 1) 关闭监听 sockets（会使 accept 抛错或返回，receive_images 检测到后退出）
    for s in listen_sockets:
        try:
            s.close()
        except Exception:
            pass

    # 2) 通知所有 receive threads/worker 退出：向每个 image_queue 放入终止标记
    for q in image_queues:
        try:
            q.put((None, None))
        except Exception:
            pass

    # 3) 告诉 writer 结束（放 None）
    write_queue.put(None)

    # 4) 停止 InferEngines（放入哨兵）
    for eng in infer_engines:
        try:
            eng.stop()
        except Exception:
            pass

    # 5) 等待所有线程结束（先 worker，再 receive，再 writer）
    for t in worker_threads:
        t.join(timeout=5)

    for t in recv_threads:
        t.join(timeout=5)

    # 最后等待 writer 真正 flush 完成
    writer_t.join(timeout=5)

    _speed_mon.stop()
    print("[OK] 所有线程已退出，程序优雅关闭完成。")

if __name__ == "__main__":
    with open(os.path.join(_REPO_ROOT, 'config', 'config0.yaml'), 'r', encoding='utf-8') as f:
        config0 = yaml.safe_load(f)

    conduct_id = config0["conduct_id"]
    strip_count, fukuan0 = get_strip_count_and_fukuan(config0)
    if len(fukuan0) != strip_count:
        print(f"[config][warn] strip_count({strip_count}) 与有效幅宽数量({len(fukuan0)})不一致，按最小可用数量运行。")
    num_strips = len(fukuan0)
    print(f"[config] 检测到配置文件包含 {num_strips} 条带钢。")

    folder_id_now = find_folders_with_id(base_path="D:\\detect result", product_id=conduct_id)
    num_cams = 4
    image_anomaly_center_list = [[[] for _ in range(num_strips)] for _ in range(num_cams)]
    image_anomaly_area_list = [[[] for _ in range(num_strips)] for _ in range(num_cams)]
    fukuan_list = [[[] for _ in range(num_strips)] for _ in range(num_cams)]
    history_image_id_list = [0 for _ in range(num_cams)]

    if folder_id_now:
        result_path_all = folder_id_now[0]
        for cam_idx in range(num_cams):
            for strip_idx in range(num_strips):
                folder_path = os.path.join(result_path_all, f"{cam_idx + 1}", f"strip_{strip_idx + 1}")
                try:
                    with open(os.path.join(folder_path, "image_anomaly_center.json"), 'r', encoding='utf-8') as file:
                        image_anomaly_center_list[cam_idx][strip_idx] = fix_json(file.read())
                except FileNotFoundError:
                    image_anomaly_center_list[cam_idx][strip_idx] = []
                try:
                    with open(os.path.join(folder_path, "image_anomaly_area.json"), 'r', encoding='utf-8') as file:
                        image_anomaly_area_list[cam_idx][strip_idx] = fix_json(file.read())
                except FileNotFoundError:
                    image_anomaly_area_list[cam_idx][strip_idx] = []
                try:
                    with open(os.path.join(folder_path, "fukuan.json"), 'r', encoding='utf-8') as file:
                        fukuan_list[cam_idx][strip_idx] = fix_json(file.read())
                except FileNotFoundError:
                    fukuan_list[cam_idx][strip_idx] = []
            hid_path = os.path.join(result_path_all, f"{cam_idx + 1}", "history_image_id.json")
            try:
                with open(hid_path, "r", encoding="utf-8") as file:
                    raw = json.load(file)
                if isinstance(raw, dict):
                    history_image_id_list[cam_idx] = int(raw.get("last_id", 0))
                else:
                    history_image_id_list[cam_idx] = int(raw) if raw is not None else 0
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
                history_image_id_list[cam_idx] = 0
    else:
        start_time_str = datetime.now().strftime("%Y%m%d")
        result_path_all = os.path.join(
            os.path.join(_REPO_ROOT, "detect result"),
            start_time_str,
            f"{conduct_id}"
        )
        os.makedirs(result_path_all, exist_ok=True)

    with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    Consecutive_Check = config["Consecutive_Check"]
    Consecutive_thres_num = config["Consecutive_thres_num"]

    print("初始化已完成")
    run_online()





