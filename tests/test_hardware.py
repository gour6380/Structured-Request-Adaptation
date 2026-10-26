"""Discovery and immutable device policies tested without accelerator execution."""

import sys
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src import hardware

HOST = {
    "system": "Linux",
    "machine": "x86_64",
    "cpu_model": "Example CPU",
    "memory_bytes": 64 * 1024**3,
}


def gpu(uuid, memory_gib=48, native_bf16=True, name="Example GPU", capability=(8, 9)):
    return {
        "uuid": uuid,
        "total_memory": memory_gib * 1024**3,
        "name": name,
        "native_bf16": native_bf16,
        "capability": capability,
    }


def install_fake_hardware(monkeypatch, gpus=(), *, mps=False, cuda_available=None):
    active = [0]
    calls = []

    @contextmanager
    def device(index):
        previous, active[0] = active[0], index
        try:
            yield
        finally:
            active[0] = previous

    def bf16(*, including_emulation):
        calls.append((active[0], including_emulation))
        return gpus[active[0]]["native_bf16"]

    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: bool(gpus) if cuda_available is None else cuda_available,
            device_count=lambda: len(gpus),
            get_device_properties=lambda index: SimpleNamespace(**gpus[index]),
            get_device_capability=lambda index: gpus[index]["capability"],
            device=device,
            is_bf16_supported=bf16,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(hardware, "_host_record", lambda: dict(HOST))
    return calls


@pytest.mark.parametrize("mps,expected", [(True, "mps"), (False, "cpu")])
def test_non_cuda_devices_are_serial_with_conservative_batches(monkeypatch, mps, expected):
    install_fake_hardware(monkeypatch, mps=mps)
    plan = hardware.detect_hardware()
    assert plan["device"] == expected
    assert plan["parallel"] is False
    assert plan["gpus"] == plan["native_bf16"] == {}
    assert plan["arm_gpu"] == plan["arm_masks"] == {"lora": None, "qlora": None}
    assert plan["selected_gpu_indices"] == []
    assert plan["compute_dtype"] == ("bfloat16" if mps else "float32")
    assert plan["use_grad_scaler"] is False
    assert plan["batch_defaults"] == {
        "microbatch_size": 1,
        "gradient_accumulation": 16,
        "evaluation_batch_size": 1,
    }
    assert plan["host"] == HOST


def test_cuda_wins_over_mps_and_single_gpu_runs_both_arms_serially(monkeypatch):
    calls = install_fake_hardware(monkeypatch, [gpu("GPU-visible-physical-7")], mps=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    plan = hardware.detect_hardware()
    assert plan["device"] == "cuda"
    assert plan["parallel"] is False
    assert plan["arm_gpu"] == {"lora": 0, "qlora": 0}
    assert plan["arm_masks"] == {
        "lora": "GPU-visible-physical-7",
        "qlora": "GPU-visible-physical-7",
    }
    assert calls == [(0, False)]
    assert plan["compute_dtype"] == "bfloat16"
    assert plan["use_grad_scaler"] is False


def test_two_cuda_gpus_are_parallel_and_unused_devices_do_not_lower_pair_policy(monkeypatch):
    devices = [gpu("GPU-b"), gpu("GPU-a"), gpu("GPU-third", 8, False)]
    calls = install_fake_hardware(monkeypatch, devices)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0,2")
    plan = hardware.detect_hardware()
    assert plan["parallel"] is True
    assert plan["selected_gpu_indices"] == [0, 1]
    assert plan["arm_gpu"] == {"lora": 0, "qlora": 1}
    assert plan["arm_masks"] == {"lora": "GPU-b", "qlora": "GPU-a"}
    assert plan["minimum_gpu_memory_bytes"] == 48 * 1024**3
    assert plan["compute_dtype"] == "bfloat16"
    assert plan["matched_gpus"] is True
    assert plan["batch_defaults"] == {
        "microbatch_size": 4,
        "gradient_accumulation": 4,
        "evaluation_batch_size": 8,
    }
    assert calls == [(0, False), (1, False), (2, False)]
    assert len(plan["gpus"]) == 3


class TorchStyleUUID:
    """Match torch._C._CUuuid's unprefixed text and 16-byte property."""

    def __init__(self, text):
        self.text = text
        self.bytes = list(bytes.fromhex(text.replace("-", "")))

    def __str__(self):
        return self.text


def test_realistic_uuid_objects_keep_visible_device_order_and_stable_worker_masks(monkeypatch):
    first = "8932f937-d72c-4106-c12f-20bd9faed9f6"
    second = "a08d91cb-e324-655c-71ba-7570956445bc"
    install_fake_hardware(monkeypatch, [gpu(TorchStyleUUID(second)), gpu(TorchStyleUUID(first))])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,2")
    plan = hardware.detect_hardware()
    assert plan["gpus"]["0"]["uuid"] == f"GPU-{second}"
    assert plan["gpus"]["1"]["uuid"] == f"GPU-{first}"
    assert plan["arm_masks"] == {"lora": f"GPU-{second}", "qlora": f"GPU-{first}"}
    assert plan["parallel"] is True
    hardware.validate_hardware_plan(plan)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("8932f937-d72c-4106-c12f-20bd9faed9f6", "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"),
        ("GPU-8932f937-d72c-4106-c12f-20bd9faed9f6", "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"),
        ("MIG-c788539f-c1ea-5a36-8b4d-9b07d024b1bb", "MIG-c788539f-c1ea-5a36-8b4d-9b07d024b1bb"),
    ],
)
def test_uuid_normalization_preserves_bits_and_existing_identity_prefix(value, expected):
    assert hardware.cuda_device_uuid(SimpleNamespace(uuid=value)) == expected


@pytest.mark.parametrize("value", [None, "", "unavailable", "0", "not-a-uuid", "8932f937"])
def test_uuid_normalization_rejects_missing_or_ambiguous_identity(value):
    with pytest.raises(ValueError, match="UUID"):
        hardware.cuda_device_uuid(SimpleNamespace(uuid=value))


def test_uuid_normalization_rejects_absent_property():
    with pytest.raises(ValueError, match="UUID"):
        hardware.cuda_device_uuid(SimpleNamespace())


def test_mixed_cuda_pair_uses_common_fp16_and_smaller_gpu_memory(monkeypatch):
    devices = [gpu("GPU-a"), gpu("GPU-b", 16, False, "Older GPU", (7, 5))]
    install_fake_hardware(monkeypatch, devices)
    plan = hardware.detect_hardware()
    assert plan["parallel"] is True
    assert plan["compute_dtype"] == "float16"
    assert plan["use_grad_scaler"] is True
    assert plan["matched_gpus"] is False
    assert plan["minimum_gpu_memory_bytes"] == 16 * 1024**3
    assert plan["batch_defaults"]["microbatch_size"] == 1
    assert plan["native_bf16"] == {"0": True, "1": False}


@pytest.mark.parametrize("memory,expected", [(24 * 1024**3, 4), (24 * 1024**3 - 1, 1)])
def test_batching_threshold_uses_actual_available_capacity(memory, expected):
    assert hardware.recommended_batches("cuda", memory)["microbatch_size"] == expected


@pytest.mark.parametrize("memory", [None, 0, -1, True, 24.5])
def test_cuda_batching_rejects_unknown_capacity(memory):
    with pytest.raises(ValueError, match="memory capacity"):
        hardware.recommended_batches("cuda", memory)


@pytest.mark.parametrize("uuids", [["unavailable"], ["GPU-same", "GPU-same"]])
def test_discovery_fails_instead_of_assigning_ambiguous_devices(monkeypatch, uuids):
    install_fake_hardware(monkeypatch, [gpu(uuid) for uuid in uuids])
    with pytest.raises(ValueError, match="UUID"):
        hardware.detect_hardware()


def test_available_cuda_with_no_devices_does_not_silently_fall_back(monkeypatch):
    install_fake_hardware(monkeypatch, mps=True, cuda_available=True)
    with pytest.raises(RuntimeError, match="without a visible device"):
        hardware.detect_hardware()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.update(compute_dtype="float32"),
        lambda plan: plan.update(parallel=False),
        lambda plan: plan.update(parallel=1),
        lambda plan: plan.update(version=True),
        lambda plan: plan["arm_masks"].update(qlora="GPU-wrong"),
        lambda plan: plan["batch_defaults"].update(gradient_accumulation=8),
        lambda plan: plan["gpus"]["0"].update(index=False),
        lambda plan: plan["native_bf16"].update({"1": 1}),
        lambda plan: plan["host"].update(memory_bytes=0),
    ],
)
def test_saved_policy_rejects_tampering_without_hardware_queries(monkeypatch, mutate):
    install_fake_hardware(monkeypatch, [gpu("GPU-a"), gpu("GPU-b")])
    plan = deepcopy(hardware.detect_hardware())
    monkeypatch.setitem(sys.modules, "torch", None)
    mutate(plan)
    with pytest.raises(ValueError):
        hardware.validate_hardware_plan(plan)


def test_apple_host_record_keeps_chip_identity(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "arm")
    calls = []

    def sysctl(command, **kwargs):
        calls.append(command)
        return "Apple M2 Pro\n"

    monkeypatch.setattr(hardware.subprocess, "check_output", sysctl)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=16 * 1024**3)),
    )
    record = hardware._host_record()
    assert record == {
        "system": "Darwin",
        "machine": "arm64",
        "cpu_model": "Apple M2 Pro",
        "memory_bytes": 16 * 1024**3,
    }
    assert calls == [["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]]


def test_restricted_host_model_records_absence_without_device_fallback(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "")

    def denied(*args, **kwargs):
        raise PermissionError("restricted")

    monkeypatch.setattr(hardware.subprocess, "check_output", denied)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=16 * 1024**3)),
    )
    assert hardware._host_record()["cpu_model"] is None
