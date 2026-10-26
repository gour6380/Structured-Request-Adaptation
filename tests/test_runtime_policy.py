"""Runtime policy and telemetry checks using mocked accelerator APIs."""

import threading
from types import SimpleNamespace
from uuid import UUID

import psutil
import pytest
import torch

from src import hardware, runtime


@pytest.mark.parametrize(
    "native_bf16,dtype,scaler", [(True, "bfloat16", False), (False, "float16", True)]
)
def test_cuda_runtime_uses_native_precision_and_takes_priority(
    monkeypatch, native_bf16, dtype, scaler
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name="Example GPU", uuid="GPU-physical-4", total_memory=16 * 1024**3
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 5))
    calls = []

    def bf16(*, including_emulation):
        calls.append(including_emulation)
        return native_bf16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", bf16)
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: [])
    record = runtime.environment_record()
    assert record["device"] == "cuda"
    assert record["compute_dtype"] == dtype
    assert record["use_grad_scaler"] is scaler
    assert record["cuda_device"]["uuid"] == "GPU-physical-4"
    assert calls == [False]


@pytest.mark.parametrize("as_object", [False, True])
def test_runtime_records_same_normalized_uuid_as_discovery(monkeypatch, as_object):
    text = "8932f937-d72c-4106-c12f-20bd9faed9f6"
    properties = SimpleNamespace(
        name="A100", uuid=UUID(text) if as_object else text, total_memory=80 * 1024**3
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"GPU-{text}")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: [])
    record = runtime.environment_record()
    assert record["cuda_device"]["uuid"] == hardware.cuda_device_uuid(properties) == f"GPU-{text}"


@pytest.mark.parametrize("mps,device,dtype", [(True, "mps", "bfloat16"), (False, "cpu", "float32")])
def test_runtime_mps_and_cpu_preserve_existing_precision(monkeypatch, mps, device, dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: ("26.0", (), ""))
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: [])
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    record = runtime.environment_record()
    assert record["device"] == device
    assert record["compute_dtype"] == dtype
    assert record["use_grad_scaler"] is False
    assert record["cuda_device"] is None


@pytest.mark.parametrize(
    "version,fallback,watermark,match",
    [
        ("13.0", "0", None, "macOS 14"),
        ("26.0", "1", None, "fallback"),
        ("26.0", "0", "0", "memory protection"),
        ("26.0", "0", "nan", "memory protection"),
    ],
)
def test_mps_protections_still_fail_explicitly(monkeypatch, version, fallback, watermark, match):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(runtime.platform, "mac_ver", lambda: (version, (), ""))
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", fallback)
    if watermark is None:
        monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", watermark)
    with pytest.raises(RuntimeError, match=match):
        runtime.environment_record()


@pytest.mark.parametrize("device,expected", [("cuda", 1), ("cuda:2", "cuda:2")])
def test_cuda_sampler_binds_caller_device_for_background_thread(monkeypatch, device, expected):
    main_thread = threading.get_ident()
    allocations, reservations = [], []
    sampled = threading.Event()
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(sout=0))
    monkeypatch.setattr(
        psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=300))
    )
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: 1 if threading.get_ident() == main_thread else 0
    )

    def allocated(index):
        allocations.append((index, threading.get_ident()))
        if threading.get_ident() != main_thread:
            sampled.set()
        return 100

    def reserved(index):
        reservations.append(index)
        return 200

    monkeypatch.setattr(torch.cuda, "memory_allocated", allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", reserved)
    with runtime.MemorySampler(device, interval=0.001) as sampler:
        assert sampled.wait(2), "Sampler thread did not run"
    assert len(allocations) >= 3
    assert all(index == expected for index, _ in allocations)
    assert all(index == expected for index in reservations)
    assert sampler.summary()["cuda_allocated_bytes"] == 100
    assert sampler.summary()["cuda_reserved_bytes"] == 200
