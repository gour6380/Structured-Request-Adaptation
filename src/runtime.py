"""Explicit device policy, reproducibility and sampled unified-memory telemetry."""

from __future__ import annotations

import gc
import importlib.metadata
import math
import os
import platform
import random
import sys
import threading
import time
from contextlib import nullcontext
from typing import Any


def configure_hub_cache(project_root) -> dict[str, str]:
    """Explicit notebook setup before importing Hugging Face, using local caches.

    An external-drive HF_HOME is not required for this independent project.
    Restart the kernel if Hugging Face has already frozen different cache paths.
    """
    from pathlib import Path

    root = Path(project_root).resolve() / ".hf-cache"
    paths = {
        "HF_HOME": str(root),
        "HF_HUB_CACHE": str(root / "hub"),
        "HF_XET_CACHE": str(root / "xet"),
    }
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and any(
        getattr(constants, key, None) != value for key, value in paths.items()
    ):
        raise RuntimeError(
            "Restart the kernel and configure project caches before Hugging Face imports"
        )
    for key, value in paths.items():
        os.environ[key] = value
    return paths


def select_device() -> str:
    from .hardware import preferred_device

    return preferred_device()


def environment_record() -> dict[str, Any]:
    """Called only by explicit run creation/opening, never module import."""
    import torch

    if str(torch.get_default_device()) != "cpu" or torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Start a fresh kernel with default CPU/float32 tensor creation")
    device = select_device()
    if device == "mps":
        if int(platform.mac_ver()[0].split(".")[0]) < 14:
            raise RuntimeError("MPS BF16 requires macOS 14 or later")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError(
                "Disable silent MPS operation-to-CPU fallback before starting the kernel"
            )
        high_watermark = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        if high_watermark is not None:
            try:
                ratio = float(high_watermark)
            except ValueError as exc:
                raise RuntimeError("Invalid MPS memory protection ratio") from exc
            if not math.isfinite(ratio) or ratio <= 0:
                raise RuntimeError("MPS memory protection must remain enabled")
    dtype = (
        "float32"
        if device == "cpu"
        else (
            "float16"
            if device == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False)
            else "bfloat16"
        )
    )
    cuda_device = None
    if device == "cuda":
        from .hardware import cuda_device_uuid

        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda_device = {
            "index": index,
            "name": properties.name,
            "uuid": cuda_device_uuid(properties),
            "total_memory": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
        }
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": device,
        "compute_dtype": dtype,
        "use_grad_scaler": dtype == "float16",
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": cuda_device,
        "packages": dict(
            sorted(
                (d.metadata.get("Name", "unknown").lower(), d.version)
                for d in importlib.metadata.distributions()
            )
        ),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "mps_high_watermark": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        "mps_low_watermark": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "executable_name": os.path.basename(sys.executable),
    }


def torch_dtype(policy: dict[str, Any]):
    import torch

    return getattr(torch, policy["compute_dtype"])


def autocast_context(policy: dict[str, Any]):
    import torch

    if policy["device"] == "cpu" or policy["compute_dtype"] == "float32":
        return nullcontext()
    return torch.autocast(device_type=policy["device"], dtype=torch_dtype(policy))


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: str) -> None:
    import torch

    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def release_memory(device: str) -> None:
    import torch

    gc.collect()
    synchronize(device)
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()


class MemorySampler:
    """Maximum observed samples, not a true peak; RSS overlaps device allocation."""

    def __init__(self, device: str, interval: float = 0.1):
        self.device, self.interval = str(device), interval
        self._cuda_device: int | str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._maxima: dict[str, int] = {}
        self._count = 0
        self._errors: list[str] = []
        self._start = 0.0
        self._end = 0.0
        self._swap_start = 0
        self._swap_end = 0

    def sample(self) -> dict[str, int]:
        import psutil
        import torch

        values = {"rss_bytes": psutil.Process().memory_info().rss}
        if self.device.startswith("mps"):
            values.update(
                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
            )
        elif self.device.startswith("cuda"):
            values.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(self._cuda_device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(self._cuda_device),
            )
        for key, value in values.items():
            self._maxima[key] = max(value, self._maxima.get(key, 0))
        self._count += 1
        return values

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")
                return

    def __enter__(self) -> MemorySampler:
        import psutil

        if self.device.startswith("cuda"):
            import torch

            # CUDA's current device is thread-local. Bind polling to the device
            # selected by the caller before starting the sampler thread.
            self._cuda_device = (
                torch.cuda.current_device() if self.device == "cuda" else self.device
            )

        self._swap_start = psutil.swap_memory().sout
        self.sample()
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        import psutil

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval))
        self.sample()
        self._end = time.perf_counter()
        self._swap_end = psutil.swap_memory().sout

    def summary(self) -> dict[str, Any]:
        import psutil

        # Checkpoints request snapshots while the whole-fit sampler is active.
        # Preserve swap evidence at that boundary rather than waiting for exit.
        swap_end = self._swap_end if self._end else psutil.swap_memory().sout
        return {
            "metadata_version": 2,
            "metadata_scope": "single_interval",
            "metadata_complete": True,
            "intervals": 1,
            "measurement": "maximum_observed_samples",
            "sample_interval_seconds": self.interval,
            "samples": self._count,
            **self._maxima,
            "elapsed_seconds": (self._end or time.perf_counter()) - self._start,
            "system_swap_out_bytes_delta": max(0, swap_end - self._swap_start),
            "system_swap_activity_observed": swap_end > self._swap_start,
            "sampling_errors": list(self._errors),
            "sampling_errors_complete": True,
            "qualification": "RSS and device memory overlap; do not add them. Sampling may miss short-lived peaks. Swap is system-wide, not attributed to this process.",
        }
