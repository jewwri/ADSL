from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional in some environments.
    psutil = None


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class TelemetryTracker:
    run_started_utc: str
    _wall_start: float
    _last_wall_elapsed_s: float = 0.0
    _last_process_cpu_time_s: float = 0.0

    def __init__(self) -> None:
        self.run_started_utc = utc_now_iso()
        self._wall_start = perf_counter()
        self._last_wall_elapsed_s = 0.0
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._logical_cpu_count = (psutil.cpu_count(logical=True) if psutil is not None else os.cpu_count()) or 1
        self._last_process_cpu_time_s = 0.0
        if self._process is not None:
            cpu_times = self._process.cpu_times()
            self._last_process_cpu_time_s = float(getattr(cpu_times, "user", 0.0)) + float(
                getattr(cpu_times, "system", 0.0)
            )
        if psutil is not None:
            # Prime psutil's CPU percentage samplers so the next call is meaningful.
            psutil.cpu_percent(interval=None)

    def sample(self) -> dict:
        timestamp_utc = utc_now_iso()
        wall_elapsed_s = perf_counter() - self._wall_start

        process_cpu_user_s = 0.0
        process_cpu_system_s = 0.0
        process_rss_mb = float("nan")
        process_vms_mb = float("nan")
        process_thread_count = float("nan")
        system_cpu_percent = float("nan")
        system_memory_percent = float("nan")

        if self._process is not None:
            cpu_times = self._process.cpu_times()
            process_cpu_user_s = float(getattr(cpu_times, "user", 0.0))
            process_cpu_system_s = float(getattr(cpu_times, "system", 0.0))
            mem = self._process.memory_info()
            process_rss_mb = float(mem.rss) / (1024.0 * 1024.0)
            process_vms_mb = float(mem.vms) / (1024.0 * 1024.0)
            process_thread_count = float(self._process.num_threads())
            system_cpu_percent = float(psutil.cpu_percent(interval=None))
            system_memory_percent = float(psutil.virtual_memory().percent)

        process_cpu_time_s = process_cpu_user_s + process_cpu_system_s
        delta_wall_s = max(wall_elapsed_s - self._last_wall_elapsed_s, 1e-9)
        delta_cpu_s = max(process_cpu_time_s - self._last_process_cpu_time_s, 0.0)
        process_cpu_util_percent = 100.0 * delta_cpu_s / delta_wall_s
        process_cpu_util_normalized_percent = process_cpu_util_percent / max(self._logical_cpu_count, 1)

        self._last_wall_elapsed_s = wall_elapsed_s
        self._last_process_cpu_time_s = process_cpu_time_s

        gpu_memory_allocated_mb = float("nan")
        gpu_memory_reserved_mb = float("nan")
        if torch.cuda.is_available():
            gpu_memory_allocated_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            gpu_memory_reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)

        return {
            "timestamp_utc": timestamp_utc,
            "run_started_utc": self.run_started_utc,
            "wall_time_elapsed_s": wall_elapsed_s,
            "process_cpu_user_s": process_cpu_user_s,
            "process_cpu_system_s": process_cpu_system_s,
            "process_cpu_time_s": process_cpu_time_s,
            "process_cpu_util_percent": process_cpu_util_percent,
            "process_cpu_util_normalized_percent": process_cpu_util_normalized_percent,
            "process_rss_mb": process_rss_mb,
            "process_vms_mb": process_vms_mb,
            "process_thread_count": process_thread_count,
            "system_cpu_percent": system_cpu_percent,
            "system_memory_percent": system_memory_percent,
            "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
            "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
        }
