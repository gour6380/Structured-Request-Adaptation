"""Bounded-probe activity monitoring without privileged system commands."""

from __future__ import annotations

import json
import math
import plistlib
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def _cuda_gpu_utilization(device: str) -> float:
    """Query the worker's physical GPU, not nvidia-smi's unmasked index zero."""
    import torch

    from .hardware import cuda_device_uuid

    properties = torch.cuda.get_device_properties(device)
    try:
        uuid = cuda_device_uuid(properties)
    except ValueError as error:
        raise RuntimeError(
            "CUDA GPU UUID is required for unambiguous activity monitoring"
        ) from error
    if not uuid.startswith("GPU-"):
        raise RuntimeError("CUDA GPU UUID is required for unambiguous activity monitoring")
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={uuid}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    value = float(result.stdout.strip())
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("nvidia-smi returned invalid GPU utilization")
    return value


def _gpu_utilization() -> float:
    """Read system-wide GPU activity from the narrow AGX registry class."""
    if sys.platform != "darwin":
        raise RuntimeError("AGX GPU utilization is available only on macOS")
    result = subprocess.run(
        ["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator"],
        capture_output=True,
        check=True,
        timeout=2,
    )
    entries = plistlib.loads(result.stdout)
    utilization: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            stats = value.get("PerformanceStatistics", {})
            raw = stats.get("Device Utilization %") if isinstance(stats, dict) else None
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                number = float(raw)
                if not math.isfinite(number) or not 0 <= number <= 100:
                    raise ValueError("AGX reported invalid GPU utilization")
                utilization.append(number)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(entries)
    if not utilization:
        raise RuntimeError("AGX PerformanceStatistics has no Device Utilization %")
    return max(utilization)


class ActivityMonitor:
    """Sample activity into JSONL; the caller explicitly checks resource safety.

    Process CPU percent uses 100% for one logical core. GPU activity and swap
    counters are system-wide. RSS overlaps MPS unified memory and must not be
    added to it. Samples report observed maxima, not guaranteed allocation peaks.
    """

    def __init__(
        self,
        output_path: str | Path,
        device: str = "mps",
        interval: float = 1.0,
        *,
        min_available_bytes: int = 1 << 30,
        max_swap_out_bytes: int = 256 << 20,
        max_swap_used_growth_bytes: int = 256 << 20,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Sample interval must be finite and positive")
        if min(min_available_bytes, max_swap_out_bytes, max_swap_used_growth_bytes) < 0:
            raise ValueError("Safety thresholds must be nonnegative")
        self.output_path = Path(output_path)
        self.device = str(device)
        self.interval = interval
        self.min_available_bytes = min_available_bytes
        self.max_swap_out_bytes = max_swap_out_bytes
        self.max_swap_used_growth_bytes = max_swap_used_growth_bytes
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._process = None
        self._start: float | None = None
        self._end: float | None = None
        self._swap_start: int | None = None
        self._swap_latest: int | None = None
        self._swap_used_start: int | None = None
        self._swap_used_latest: int | None = None
        self._max_swap_used_growth = 0
        self._statistics: dict[str, dict[str, float | int]] = {}
        self._errors: list[str] = []
        self._samples = 0

    def _record_error(self, component: str, exc: Exception) -> str:
        message = f"{component}: {type(exc).__name__}: {exc}"
        with self._lock:
            self._errors.append(message)
        return message

    def _sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - self._start,
            "gpu_utilization_scope": (
                "physical CUDA GPU identified by UUID; all activity on that GPU"
                if self.device.startswith("cuda")
                else "system-wide; maximum across AGX entries"
            ),
            "process_cpu_percent_scale": "100% = one logical core",
            "cpu_sample_status": (
                "initial_interval_too_short; excluded from CPU aggregates"
                if self._samples == 0
                else "interval_sample"
            ),
            "swap_scope": "system-wide; not attributed to this process",
            "errors": [],
        }

        def collect(component: str, action) -> None:
            try:
                sample.update(action())
            except Exception as exc:
                sample["errors"].append(self._record_error(component, exc))

        collect("process_cpu", lambda: {"process_cpu_percent": self._process.cpu_percent()})
        collect("system_cpu", lambda: {"system_cpu_percent": psutil.cpu_percent()})
        collect("process_memory", lambda: {"rss_bytes": self._process.memory_info().rss})
        collect(
            "system_memory",
            lambda: {"system_available_bytes": psutil.virtual_memory().available},
        )

        def swap_values() -> dict[str, int]:
            swap = psutil.swap_memory()
            with self._lock:
                if self._swap_start is None:
                    self._swap_start = swap.sout
                    self._swap_used_start = swap.used
                self._swap_latest = swap.sout
                self._swap_used_latest = swap.used
                delta = max(0, swap.sout - self._swap_start)
                growth = max(0, swap.used - self._swap_used_start)
                self._max_swap_used_growth = max(self._max_swap_used_growth, growth)
            return {
                "system_swap_used_bytes": swap.used,
                "system_swap_used_growth_bytes": growth,
                "maximum_system_swap_used_growth_bytes": self._max_swap_used_growth,
                "system_swap_out_bytes": swap.sout,
                "system_swap_out_bytes_delta": delta,
            }

        collect("system_swap", swap_values)
        if self.device.startswith(("mps", "cuda")):

            def accelerator_values() -> dict[str, int]:
                import torch

                if self.device.startswith("mps"):
                    return {
                        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                    }
                return {
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device),
                }

            collect("device_memory", accelerator_values)
        if self.device.startswith("cuda"):
            collect(
                "system_gpu",
                lambda: {"system_gpu_utilization_percent": _cuda_gpu_utilization(self.device)},
            )
        elif sys.platform == "darwin":
            collect("system_gpu", lambda: {"system_gpu_utilization_percent": _gpu_utilization()})
        else:
            sample["errors"].append(
                self._record_error("system_gpu", RuntimeError("AGX GPU utilization requires macOS"))
            )
        with self._lock:
            for key, value in sample.items():
                if isinstance(value, (int, float)) and key != "elapsed_seconds":
                    if self._samples == 0 and key in (
                        "process_cpu_percent",
                        "system_cpu_percent",
                    ):
                        continue
                    stats = self._statistics.setdefault(
                        key, {"sum": 0, "samples": 0, "min": value, "max": value}
                    )
                    stats["sum"] += value
                    stats["samples"] += 1
                    stats["min"] = min(stats["min"], value)
                    stats["max"] = max(stats["max"], value)
            self._samples += 1
            self._stream.write(json.dumps(sample, allow_nan=False) + "\n")
            self._stream.flush()
        return sample

    def _poll(self) -> None:
        # psutil's system counter is thread-local; prime in this thread so its
        # first periodic reading represents a full sampling interval.
        try:
            psutil.cpu_percent()
        except Exception as exc:
            self._record_error("system_cpu_initialization", exc)
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception as exc:
                self._record_error("sample_write", exc)
                self._stop.set()

    def __enter__(self) -> ActivityMonitor:
        if self._start is not None:
            raise RuntimeError("An ActivityMonitor instance cannot be reused")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Never replace an earlier probe's evidence accidentally.
        self._stream = self.output_path.open("x", encoding="utf-8")
        self._start = time.perf_counter()
        self._process = psutil.Process()
        try:
            self._process.cpu_percent()
            psutil.cpu_percent()
            self._sample()
        except Exception:
            self._stream.close()
            raise
        self._thread = threading.Thread(target=self._poll, name="activity-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            # The only subprocess has a two-second timeout. Join before writing
            # the final sample so two writers never race with closing the file.
            self._thread.join()
        try:
            self._sample()
        except Exception as sampling_exc:
            self._record_error("final_sample", sampling_exc)
            if exc is None:
                raise
        finally:
            self._end = time.perf_counter()
            self._stream.close()

    def check_safety(self) -> None:
        """Fail in the training thread if current RAM or swap exceeds its limit."""
        if self._start is None or self._end is not None:
            raise RuntimeError("Safety checks require an active ActivityMonitor")
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("Activity monitoring stopped; inspect sampling_errors")
            baseline = self._swap_start
            used_baseline = self._swap_used_start
        try:
            available = psutil.virtual_memory().available
            swap = psutil.swap_memory()
            swap_out = swap.sout
        except Exception as exc:
            self._record_error("safety_check", exc)
            raise RuntimeError("Cannot verify RAM/swap safety") from exc
        if baseline is None or used_baseline is None:
            raise RuntimeError("No initial swap counter; cannot verify resource safety")
        with self._lock:
            self._swap_latest = swap_out
            self._swap_used_latest = swap.used
            growth = max(0, swap.used - used_baseline)
            self._max_swap_used_growth = max(self._max_swap_used_growth, growth)
        if available < self.min_available_bytes:
            raise MemoryError(
                f"System available RAM {available} bytes is below "
                f"the {self.min_available_bytes}-byte safety threshold"
            )
        delta = max(0, swap_out - baseline)
        if delta > self.max_swap_out_bytes:
            raise MemoryError(
                f"System-wide swap-out increased {delta} bytes, above "
                f"the {self.max_swap_out_bytes}-byte safety threshold"
            )
        if growth > self.max_swap_used_growth_bytes:
            raise MemoryError(
                f"System-wide swap used grew {growth} bytes from monitor entry, above "
                f"the {self.max_swap_used_growth_bytes}-byte safety threshold. "
                "This is a system pressure signal, not process attribution."
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stats = {
                key: {
                    "mean": values["sum"] / values["samples"],
                    "min": values["min"],
                    "max": values["max"],
                    "samples": values["samples"],
                }
                for key, values in self._statistics.items()
            }
            swap_delta = (
                max(0, self._swap_latest - self._swap_start)
                if self._swap_latest is not None and self._swap_start is not None
                else None
            )
            memory_keys = (
                "rss_bytes",
                "mps_allocated_bytes",
                "mps_driver_bytes",
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
            )
            return {
                "measurement": "maximum_observed_samples",
                "sample_interval_seconds": self.interval,
                "samples": self._samples,
                "elapsed_seconds": (
                    (self._end or time.perf_counter()) - self._start
                    if self._start is not None
                    else 0
                ),
                "statistics": stats,
                "maximum_observed_memory_bytes": {
                    key: stats[key]["max"] for key in memory_keys if key in stats
                },
                "minimum_system_available_bytes": stats.get("system_available_bytes", {}).get(
                    "min"
                ),
                "system_swap_out_bytes_delta": swap_delta,
                "system_swap_used_bytes_at_entry": self._swap_used_start,
                "system_swap_used_growth_bytes": (
                    max(0, self._swap_used_latest - self._swap_used_start)
                    if self._swap_used_latest is not None and self._swap_used_start is not None
                    else None
                ),
                "maximum_system_swap_used_growth_bytes": (
                    self._max_swap_used_growth if self._swap_used_start is not None else None
                ),
                "sampling_errors": list(self._errors),
                "output_path": str(self.output_path),
                "safety_thresholds": {
                    "min_available_bytes": self.min_available_bytes,
                    "max_swap_out_bytes": self.max_swap_out_bytes,
                    "max_swap_used_growth_bytes": self.max_swap_used_growth_bytes,
                },
                "qualification": (
                    "Process CPU: 100% equals one logical core. System CPU uses a "
                    "0–100% whole-machine scale. GPU activity includes other processes "
                    "on the measured device; swap is system-wide. Report RSS and "
                    "device memory separately; they overlap on MPS. Sampled memory can miss peaks. "
                    "Initial short-interval CPU samples are labeled and excluded "
                    "from CPU aggregates. Swap-used growth is monitored separately "
                    "because swap-out counters may underreport memory pressure."
                ),
            }
