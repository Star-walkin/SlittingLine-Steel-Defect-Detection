"""
速度监控模块 - speed_monitor.py
=====================================
为 detect_anomalies_online.py 提供最小侵入式的吞吐/延迟/队列水位监控。

架构数据流与背压点（反压/崩溃风险图）
──────────────────────────────────────────────────────────────────────
  ┌──────────┐  TCP 4端口   ┌────────────────────────────────────────┐
  │ C# 相机  │──────────→  │   receive_images() 接收线程             │
  │ 采集程序 │  4×4096²帧  │   recv(4字节头+payload)                 │
  │MultiCam  │             │   image_queue.put(data, idx)            │
  │Demo.exe  │             │   ★[背压点1] put()阻塞→TCP缓冲区膨胀   │
  └──────────┘             │   →C#发送线程阻塞→采集SDK回调堆积      │
                           └──────────────┬─────────────────────────┘
                                          │ queue.Queue(maxsize=200)
                                          ↓
                           ┌────────────────────────────────────────┐
                           │   worker() 处理线程（每相机×2）        │
                           │   [decode] bytes→4096×4096 numpy       │
                           │   [split ] split_multi_strips           │
                           │   [*vis  ] split_vis 写盘（DEBUG_IO）  │
                           │   [infer ] M.detect_ano()              │
                           │   [*dbg  ] debug_visuals 写盘(DEBUG_IO)│
                           │   [locate] obtain_anomaly_location()   │
                           │   ★[背压点2] cam_lock包住整帧          │
                           │   →2个worker实际串行，第二个空转        │
                           └──────────────┬─────────────────────────┘
                                          │ write_queue（无上界）
                                          ↓
                           ┌────────────────────────────────────────┐
                           │   writer_loop() 写盘线程（全局×1）     │
                           │   JSON全量重写（文件随帧数线性增大）    │
                           │   defect_images 缺陷图写盘             │
                           │   ★[背压点3] JSON体积→flush越来越慢   │
                           │   →write_queue堆积→内存持续增长        │
                           └──────────────┬─────────────────────────┘
                                          │ 文件系统（JSON + 图片）
                                          ↓
                           ┌────────────────────────────────────────┐
                           │   UI: ImageLoaderThread (2s轮询JSON)   │
                           │   UI: WaveformThread    (2s轮询JSON)   │
                           │   UI: render_timer      (33ms Mpl重绘) │
                           │   ★[风险4] Matplotlib 30FPS重绘        │
                           │   +create_button高频创建/销毁按钮       │
                           │   →与检测进程同机时抢占CPU/磁盘资源    │
                           └────────────────────────────────────────┘

  *: DEBUG_IO=False 时跳过所有大图写盘，生产环境建议保持 False。

使用方式（最小侵入集成）
──────────────────────────────────────────────────────────────────────
  # 在 run_online() 启动时：
  from speed_monitor import init_monitor, get_monitor, StageTimer
  _mon = init_monitor(num_cams=4)

  # 在 receive_images() 每收到一帧后：
  get_monitor().record_input(cam_id)

  # 在 worker() 处理每帧时：
  get_monitor().record_qsize(cam_id, image_queue.qsize())
  with StageTimer(cam_id, "decode"): ...
  with StageTimer(cam_id, "split"): ...
  get_monitor().record_processed(cam_id)

  # 在 detect() 关键阶段：
  with StageTimer(cam_id, "infer"): M.detect_ano(...)
  with StageTimer(cam_id, "locate"): M.obtain_anomaly_location(...)
"""

import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
import time
import threading
from typing import Optional, Dict

# ---------------------------------------------------------------------------
# 调试性能开关
# ---------------------------------------------------------------------------
# True  → 保存 split_vis / debug_visuals（大量磁盘写入，仅调试用）
# False → 跳过所有 debug 大图写盘（生产环境，显著提升吞吐）
# 在线检测由 config.yaml 的 debug_io 项在启动时调用 set_debug_io() 覆盖。
DEBUG_IO: bool = False


def set_debug_io(enabled: bool) -> None:
    """运行时开关：请通过本函数修改，勿依赖 `from speed_monitor import DEBUG_IO` 的旧绑定。"""
    global DEBUG_IO
    DEBUG_IO = bool(enabled)


# ---------------------------------------------------------------------------
# SpeedMonitor
# ---------------------------------------------------------------------------
class SpeedMonitor:
    """
    轻量级速度监控器。

    统计指标（按 REPORT_INTERVAL 周期汇报）：
      - input_fps   : 每相机每秒接收帧数（receive_images 侧）
      - proc_fps    : 每相机每秒处理帧数（worker 侧）
      - qsize       : 当前队列水位（最近一次记录值）
      - 阶段平均耗时: decode / split / infer / locate / write / debug_io (ms)

    判别规则：
      - proc_fps < input_fps × 0.85 → 处理落后，队列即将堆积
      - t_infer 最高 + GPU 利用率高 → 纯推理瓶颈
      - t_write / t_debug_io 最高   → 磁盘 I/O 瓶颈，关闭 DEBUG_IO
      - 2 worker 但 proc_fps 未提升 → 大锁导致 worker 串行
    """

    REPORT_INTERVAL: float = 5.0  # 默认统计周期（秒），与 init_monitor 的 report_interval_sec 一致

    _STAGES = ("decode", "split", "infer", "locate", "write", "debug_io")

    def __init__(self, num_cams: int = 4, report_interval_sec: float | None = None):
        self.num_cams = num_cams
        iv = float(self.REPORT_INTERVAL if report_interval_sec is None else report_interval_sec)
        self.report_interval_sec = max(0.5, iv)
        self._lock = threading.Lock()
        self._interval_start = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._input_cnt: Dict[int, int] = {}
        self._proc_cnt: Dict[int, int] = {}
        self._total_input = {i: 0 for i in range(self.num_cams)}
        self._total_proc = {i: 0 for i in range(self.num_cams)}
        self._qsize: Dict[int, int] = {}
        self._stage_sum: Dict[int, Dict[str, float]] = {}
        self._stage_cnt: Dict[int, Dict[str, int]] = {}
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._input_cnt = {i: 0 for i in range(self.num_cams)}
        self._proc_cnt = {i: 0 for i in range(self.num_cams)}
        self._qsize = {i: 0 for i in range(self.num_cams)}
        self._stage_sum = {i: {s: 0.0 for s in self._STAGES} for i in range(self.num_cams)}
        self._stage_cnt = {i: {s: 0 for s in self._STAGES} for i in range(self.num_cams)}

    def start(self) -> None:
        """启动后台汇报线程（daemon，不阻止主进程退出）。"""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="SpeedMonitor"
        )
        self._thread.start()

    def stop(self) -> None:
        """停止汇报线程。"""
        self._running = False

    # ---- 计数 API ----

    def record_input(self, cam_id: int) -> None:
        """每接收到并成功入队一帧时调用。"""
        with self._lock:
            if cam_id in self._input_cnt:
                self._input_cnt[cam_id] += 1
                self._total_input[cam_id] += 1

    def record_processed(self, cam_id: int) -> None:
        """每完成一帧完整处理（worker 循环末尾）时调用。"""
        with self._lock:
            if cam_id in self._proc_cnt:
                self._proc_cnt[cam_id] += 1
                self._total_proc[cam_id] += 1

    def record_qsize(self, cam_id: int, qsize: int) -> None:
        """记录当前队列水位（每帧处理前调用）。"""
        with self._lock:
            if cam_id in self._qsize:
                self._qsize[cam_id] = qsize

    def record_stage(self, cam_id: int, stage: str, elapsed_s: float) -> None:
        """累计某阶段耗时（秒）。"""
        with self._lock:
            d = self._stage_sum.get(cam_id)
            if d is not None and stage in d:
                d[stage] += elapsed_s
                self._stage_cnt[cam_id][stage] += 1

    # ---- 内部汇报 ----

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.report_interval_sec)
            self._report()

    def _report(self) -> None:
        now = time.time()
        with self._lock:
            dt = max(now - self._interval_start, 1e-9)
            sep = "=" * 72
            lines = [
                f"\n{sep}",
                f"[SpeedMonitor] 统计周期={dt:.1f}s  "
                f"DEBUG_IO={'ON（写盘模式，吞吐受限）' if DEBUG_IO else 'OFF（生产模式）'}",
            ]

            any_active = False
            for cam in range(self.num_cams):
                ifps = self._input_cnt[cam] / dt
                pfps = self._proc_cnt[cam] / dt
                qs = self._qsize[cam]
                if ifps == 0 and pfps == 0:
                    continue
                any_active = True

                ic = self._input_cnt[cam]
                pc = self._proc_cnt[cam]
                ti = self._total_input[cam]
                tp = self._total_proc[cam]
                lag = ti - tp

                # 阶段耗时摘要
                parts = []
                e2e_ms = 0.0
                for s in self._STAGES:
                    cnt = self._stage_cnt[cam][s]
                    if cnt > 0:
                        avg_ms = self._stage_sum[cam][s] / cnt * 1000
                        parts.append(f"{s}={avg_ms:.1f}ms")
                        if s in ("decode", "split", "infer", "locate"):
                            e2e_ms += avg_ms
                stage_str = "  ".join(parts) if parts else "（尚无阶段计时数据）"
                e2e_hint = f"  估算法Σ(decode+split+infer+locate)≈{e2e_ms:.1f}ms/帧" if e2e_ms > 0 else ""

                # 预警
                warns = []
                if ifps > 0.1 and pfps < ifps * 0.85:
                    warns.append(f"处理({pfps:.2f}fps)<输入({ifps:.2f}fps)")
                if qs >= 150:
                    warns.append(f"队列={qs}")
                if lag > 30:
                    warns.append(f"累计未处理≈{lag}帧(recv={ti} done={tp})")
                warn = ("  [!] " + "；".join(warns)) if warns else ""

                lines.append(
                    f"  CAM{cam + 1}: 本周期 接收={ic}帧({ifps:.2f}fps)  处理={pc}帧({pfps:.2f}fps)  "
                    f"队列={qs}  累计 recv={ti} done={tp}{warn}"
                )
                lines.append(f"         阶段: {stage_str}{e2e_hint}")

            if not any_active:
                lines.append("  （尚未收到任何帧，等待数据中...）")

            lines.append(sep)
            print("\n".join(lines), flush=True)

            # 重置本周期计数
            self._reset_counters()
            self._interval_start = now


# ---------------------------------------------------------------------------
# StageTimer：上下文管理器，用于对代码块计时
# ---------------------------------------------------------------------------
class StageTimer:
    """
    对代码块计时，自动上报给全局 SpeedMonitor 实例。

    用法：
        with StageTimer(cam_id, "infer"):
            result = M.detect_ano(img)

        with StageTimer(cam_id, "split"):
            strips, widths, splits = split_multi_strips(gray, ...)
    """

    __slots__ = ("_cam", "_stage", "_start")

    def __init__(self, cam_id: int, stage: str) -> None:
        self._cam = cam_id
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        mon = _global_monitor
        if mon is not None:
            mon.record_stage(self._cam, self._stage, time.perf_counter() - self._start)


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
_global_monitor: Optional[SpeedMonitor] = None


def init_monitor(num_cams: int = 4, report_interval_sec: float | None = None) -> SpeedMonitor:
    """
    初始化并启动全局速度监控器（在 run_online() 入口调用一次）。
    report_interval_sec: 终端汇总打印间隔（秒），速度测试可改为 1.0。
    """
    global _global_monitor
    _global_monitor = SpeedMonitor(num_cams=num_cams, report_interval_sec=report_interval_sec)
    _global_monitor.start()
    return _global_monitor


def get_monitor() -> Optional[SpeedMonitor]:
    """获取全局速度监控器实例（未初始化时返回 None，不会抛异常）。"""
    return _global_monitor
