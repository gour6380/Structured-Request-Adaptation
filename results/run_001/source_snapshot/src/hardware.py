"""Discover visible hardware and freeze a shared precision/batching plan.

Discovery runs only when explicitly requested. It does not load a model, allocate
training tensors, or silently change devices after a run has started.
"""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

LARGE_CUDA_MEMORY_BYTES = 24 * 1024**3
ARMS = ("lora", "qlora")


def cuda_device_uuid(properties: Any) -> str:
    """Return the NVIDIA mask identifier for this exact visible CUDA device.

    PyTorch's _CUuuid string is a bare UUID, whereas CUDA masks and nvidia-smi
    use GPU-prefixed identifiers. Normalize the UUID itself, never infer a
    physical GPU from an ordinal that may have been remapped by a container.
    Already qualified GPU/MIG identifiers keep their supplied identity.
    """
    value = str(getattr(properties, "uuid", "unavailable")).strip()
    if value.startswith(("GPU-", "MIG-")) and len(value) > 4:
        return value
    if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        identifier = UUID(value)
        if identifier.int:
            return f"GPU-{identifier}"
    raise ValueError(
        f"CUDA GPU UUID is missing or malformed: {value!r}. "
        "A stable device UUID is required; GPU indices cannot substitute for it."
    )


def preferred_device() -> str:
    """Select CUDA first, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def recommended_batches(device: str, minimum_gpu_memory_bytes: int | None) -> dict[str, int]:
    if device not in {"cuda", "mps", "cpu"}:
        raise ValueError("Unknown device policy")
    if device == "cuda":
        if type(minimum_gpu_memory_bytes) is not int or minimum_gpu_memory_bytes <= 0:
            raise ValueError("CUDA batching requires the selected GPUs' positive memory capacity")
        if minimum_gpu_memory_bytes >= LARGE_CUDA_MEMORY_BYTES:
            return {"microbatch_size": 4, "gradient_accumulation": 4, "evaluation_batch_size": 8}
    return {"microbatch_size": 1, "gradient_accumulation": 16, "evaluation_batch_size": 1}


def _host_record() -> dict[str, Any]:
    import psutil

    cpu_model = platform.processor().strip() or None
    system = platform.system()
    try:
        if system == "Darwin":
            cpu_model = (
                subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).strip()
                or cpu_model
            )
        elif system == "Linux":
            with Path("/proc/cpuinfo").open(encoding="utf-8") as stream:
                for line in stream:
                    key, separator, value = line.partition(":")
                    if separator and key.strip() in {"model name", "Hardware"}:
                        cpu_model = value.strip() or cpu_model
                        break
    except (OSError, subprocess.SubprocessError):
        # A restricted host may hide the model string; record its absence rather
        # than inventing a chip identity or changing the selected device.
        pass
    return {
        "system": system,
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "memory_bytes": psutil.virtual_memory().total,
    }


def _build_plan(
    device: str,
    gpus: dict[str, dict[str, Any]],
    native_bf16: dict[str, bool],
    host: dict[str, Any],
) -> dict[str, Any]:
    selected = list(range(min(2, len(gpus)))) if device == "cuda" else []
    parallel = len(selected) == 2
    assigned = {"lora": 0, "qlora": 1 if parallel else 0} if selected else dict.fromkeys(ARMS)
    masks = {
        arm: gpus[str(index)]["uuid"] if index is not None else None
        for arm, index in assigned.items()
    }
    minimum_memory = (
        min(gpus[str(index)]["total_memory"] for index in selected) if selected else None
    )
    dtype = (
        "float32"
        if device == "cpu"
        else "bfloat16"
        if device == "mps" or all(native_bf16[str(index)] for index in selected)
        else "float16"
    )
    first = gpus[str(selected[0])] if selected else None
    matched = all(
        all(gpus[str(index)][key] == first[key] for key in ("name", "total_memory", "capability"))
        for index in selected[1:]
    )
    return {
        "version": 1,
        "device": device,
        "gpus": gpus,
        "native_bf16": native_bf16,
        "selected_gpu_indices": selected,
        "parallel": parallel,
        "arm_gpu": assigned,
        "arm_masks": masks,
        "compute_dtype": dtype,
        "use_grad_scaler": dtype == "float16",
        "minimum_gpu_memory_bytes": minimum_memory,
        "batch_defaults": recommended_batches(device, minimum_memory),
        "matched_gpus": matched,
        "host": host,
    }


def validate_hardware_plan(plan: dict[str, Any]) -> None:
    """Validate a saved/discovered plan without querying or initializing hardware."""
    if not isinstance(plan, dict) or plan.get("device") not in {"cuda", "mps", "cpu"}:
        raise ValueError("Hardware plan requires device CUDA, MPS or CPU")
    device = plan["device"]
    if type(plan.get("version")) is not int or plan["version"] != 1:
        raise ValueError("Unsupported hardware plan version")
    if any(
        type(plan.get(key)) is not bool for key in ("parallel", "use_grad_scaler", "matched_gpus")
    ):
        raise ValueError("Hardware policy flags must be booleans")
    gpus = plan.get("gpus")
    native = plan.get("native_bf16")
    host = plan.get("host")
    if not isinstance(gpus, dict) or not isinstance(native, dict) or not isinstance(host, dict):
        raise ValueError("Hardware plan requires GPU, precision and host records")
    if device == "cuda":
        if not gpus or set(gpus) != {str(index) for index in range(len(gpus))}:
            raise ValueError("CUDA inventory must contain contiguous visible device indices")
        if set(native) != set(gpus) or any(type(value) is not bool for value in native.values()):
            raise ValueError("Native BF16 support must be recorded for every visible CUDA device")
        uuids = set()
        for key, gpu in gpus.items():
            if (
                not isinstance(gpu, dict)
                or type(gpu.get("index")) is not int
                or gpu["index"] != int(key)
            ):
                raise ValueError("CUDA GPU index differs from its inventory position")
            if type(gpu.get("total_memory")) is not int or gpu["total_memory"] <= 0:
                raise ValueError("CUDA GPU memory capacity must be positive")
            if not isinstance(gpu.get("name"), str) or not gpu["name"]:
                raise ValueError("CUDA GPU name is missing")
            capability = gpu.get("capability")
            if (
                not isinstance(capability, list)
                or len(capability) != 2
                or any(type(value) is not int or value < 0 for value in capability)
            ):
                raise ValueError("CUDA compute capability must contain two nonnegative integers")
            uuid = gpu.get("uuid")
            if not isinstance(uuid, str) or not uuid.startswith(("GPU-", "MIG-")):
                raise ValueError("CUDA GPU UUID is required for stable worker masking")
            if uuid in uuids:
                raise ValueError("CUDA inventory contains a duplicate GPU UUID")
            uuids.add(uuid)
    elif gpus or native:
        raise ValueError("Non-CUDA plans must not contain assigned CUDA GPUs")
    if type(host.get("memory_bytes")) is not int or host["memory_bytes"] <= 0:
        raise ValueError("Host memory capacity must be recorded")
    if not all(isinstance(host.get(key), str) and host[key] for key in ("system", "machine")):
        raise ValueError("Host system and architecture must be recorded")
    if host.get("cpu_model") is not None and not isinstance(host["cpu_model"], str):
        raise ValueError("Host CPU model must be text when available")
    expected = _build_plan(device, gpus, native, host)
    if plan != expected:
        raise ValueError(
            "Hardware assignments, precision or batching differ from the inventory policy"
        )


def detect_hardware() -> dict[str, Any]:
    """Inspect visible devices; select up to two GPUs and one common arm policy.

    CUDA masks use UUIDs, so a discovery process with an inherited device mask
    does not confuse visible indices with physical GPU identities. Other visible
    GPUs are recorded but unused. Model/quantization compatibility and memory fit
    still require the explicit preflight and bounded timing pilot.
    """
    import torch

    device = preferred_device()
    gpus: dict[str, dict[str, Any]] = {}
    native: dict[str, bool] = {}
    if device == "cuda":
        count = torch.cuda.device_count()
        if count <= 0:
            raise RuntimeError("CUDA reports availability without a visible device")
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            gpus[str(index)] = {
                "index": index,
                "name": properties.name,
                "uuid": cuda_device_uuid(properties),
                "total_memory": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            with torch.cuda.device(index):
                native[str(index)] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    plan = _build_plan(device, gpus, native, _host_record())
    validate_hardware_plan(plan)
    return plan
