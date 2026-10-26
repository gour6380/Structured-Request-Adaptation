"""Monitor behavior is tested with synthetic counters, never a real model."""

from __future__ import annotations

import json
import plistlib
import subprocess
from types import SimpleNamespace
from uuid import UUID

import pytest

from src import telemetry


@pytest.mark.parametrize("form", ["bare", "object", "prefixed"])
def test_cuda_utilization_uses_normalized_physical_uuid_not_remapped_ordinal(monkeypatch, form):
    import torch

    text = "8932f937-d72c-4106-c12f-20bd9faed9f6"
    value = {"bare": text, "object": UUID(text), "prefixed": f"GPU-{text}"}[form]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"GPU-{text}")
    queried_devices = []

    def properties(device):
        queried_devices.append(device)
        return SimpleNamespace(uuid=value)

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="72\n")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(telemetry.subprocess, "run", run)
    assert telemetry._cuda_gpu_utilization("cuda:0") == 72.0
    assert queried_devices == ["cuda:0"]
    assert calls[0][0] == [
        "nvidia-smi",
        f"--id=GPU-{text}",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["timeout"] == 2


def test_cuda_utilization_does_not_guess_identity_when_uuid_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid=None)
    )
    calls = []
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises((ValueError, RuntimeError), match="UUID"):
        telemetry._cuda_gpu_utilization("cuda:0")
    assert calls == []


@pytest.fixture
def counters(monkeypatch):
    state = {"available": 4 << 30, "sout": 1234, "used": 100, "gpu": 65.0}
    process = SimpleNamespace(
        cpu_percent=lambda: 170.0,
        memory_info=lambda: SimpleNamespace(rss=2048),
    )
    monkeypatch.setattr(telemetry.psutil, "Process", lambda: process)
    monkeypatch.setattr(telemetry.psutil, "cpu_percent", lambda: 40.0)
    monkeypatch.setattr(
        telemetry.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=state["available"]),
    )
    monkeypatch.setattr(
        telemetry.psutil,
        "swap_memory",
        lambda: SimpleNamespace(sout=state["sout"], used=state["used"]),
    )
    monkeypatch.setattr(telemetry.sys, "platform", "darwin")
    monkeypatch.setattr(telemetry, "_gpu_utilization", lambda: state["gpu"])
    return state


def test_incremental_samples_summary_and_explicit_safety(tmp_path, counters):
    path = tmp_path / "activity.jsonl"
    with telemetry.ActivityMonitor(path, device="cpu", interval=100) as monitor:
        assert len(path.read_text().splitlines()) == 1
        counters["sout"] += 1024
        counters["available"] = 3 << 30
        monitor.check_safety()
        monitor._sample()
        assert len(path.read_text().splitlines()) == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["process_cpu_percent"] == 170.0
    assert "system-wide" in rows[0]["gpu_utilization_scope"]
    summary = monitor.summary()
    assert summary["statistics"]["system_gpu_utilization_percent"]["mean"] == 65.0
    assert summary["maximum_observed_memory_bytes"] == {"rss_bytes": 2048}
    assert summary["minimum_system_available_bytes"] == 3 << 30
    assert summary["system_swap_out_bytes_delta"] == 1024
    assert summary["sampling_errors"] == []
    assert not monitor._thread.is_alive()


@pytest.mark.parametrize("failure", ["ram", "swap", "swap_used"])
def test_safety_uses_fresh_counters_not_last_sample(tmp_path, counters, failure):
    with telemetry.ActivityMonitor(tmp_path / "activity.jsonl", "cpu", 100) as monitor:
        if failure == "ram":
            counters["available"] = (1 << 30) - 1
        elif failure == "swap":
            counters["sout"] += (256 << 20) + 1
        else:
            counters["used"] += (256 << 20) + 1
        with pytest.raises(MemoryError):
            monitor.check_safety()


def test_configurable_safety_and_boundary(tmp_path, counters):
    with telemetry.ActivityMonitor(
        tmp_path / "activity.jsonl",
        "cpu",
        100,
        min_available_bytes=100,
        max_swap_out_bytes=25,
        max_swap_used_growth_bytes=20,
    ) as monitor:
        counters["available"] = 100
        counters["sout"] += 25
        counters["used"] += 20
        monitor.check_safety()


def test_swap_used_growth_stops_when_swap_out_counter_stays_flat(tmp_path, counters):
    path = tmp_path / "activity.jsonl"
    with telemetry.ActivityMonitor(path, "cpu", 100) as monitor:
        counters["used"] += 2 << 30
        with pytest.raises(MemoryError, match="swap used grew"):
            monitor.check_safety()
        # A subsequent decline cannot erase the observed maximum.
        counters["used"] = 100
    summary = monitor.summary()
    assert summary["system_swap_out_bytes_delta"] == 0
    assert summary["system_swap_used_growth_bytes"] == 0
    assert summary["maximum_system_swap_used_growth_bytes"] == 2 << 30
    last = json.loads(path.read_text().splitlines()[-1])
    assert last["maximum_system_swap_used_growth_bytes"] == 2 << 30
    assert "not attributed" in last["swap_scope"]


def test_swap_used_baseline_is_entry_not_absolute_swap(tmp_path, counters):
    counters["used"] = 3 << 30
    with telemetry.ActivityMonitor(tmp_path / "activity.jsonl", "cpu", 100) as monitor:
        monitor.check_safety()
        counters["used"] += 128 << 20
        monitor._sample()
        monitor.check_safety()
    summary = monitor.summary()
    assert summary["system_swap_used_bytes_at_entry"] == 3 << 30
    assert summary["system_swap_used_growth_bytes"] == 128 << 20
    assert summary["statistics"]["system_swap_used_growth_bytes"]["max"] == 128 << 20


def test_initial_cpu_samples_are_marked_and_excluded(tmp_path, counters):
    path = tmp_path / "activity.jsonl"
    with telemetry.ActivityMonitor(path, "cpu", 100) as monitor:
        initial = json.loads(path.read_text().splitlines()[0])
        assert "excluded" in initial["cpu_sample_status"]
        assert "process_cpu_percent" not in monitor.summary()["statistics"]
    assert monitor.summary()["statistics"]["process_cpu_percent"]["samples"] == 1


def test_negative_swap_used_threshold_rejected(tmp_path):
    with pytest.raises(ValueError):
        telemetry.ActivityMonitor(tmp_path / "activity.jsonl", max_swap_used_growth_bytes=-1)


def test_sample_component_error_is_recorded_and_other_metrics_survive(
    tmp_path, counters, monkeypatch
):
    def broken_gpu():
        raise subprocess.TimeoutExpired("ioreg", 2)

    monkeypatch.setattr(telemetry, "_gpu_utilization", broken_gpu)
    path = tmp_path / "activity.jsonl"
    with telemetry.ActivityMonitor(path, "cpu", 100) as monitor:
        monitor.check_safety()
    first = json.loads(path.read_text().splitlines()[0])
    assert first["rss_bytes"] == 2048
    assert "system_gpu_utilization_percent" not in first
    assert "TimeoutExpired" in first["errors"][0]
    assert monitor.summary()["sampling_errors"]


def test_missing_initial_swap_fails_closed(tmp_path, counters, monkeypatch):
    def broken_swap():
        raise OSError("counter unavailable")

    monkeypatch.setattr(telemetry.psutil, "swap_memory", broken_swap)
    with telemetry.ActivityMonitor(tmp_path / "activity.jsonl", "cpu", 100) as monitor:
        with pytest.raises(RuntimeError, match="Cannot verify"):
            monitor.check_safety()
    assert monitor.summary()["system_swap_out_bytes_delta"] is None


def test_mps_memory_samples_use_no_model(tmp_path, counters, monkeypatch):
    import torch

    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 300)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 700)
    with telemetry.ActivityMonitor(tmp_path / "activity.jsonl", "mps", 100) as monitor:
        monitor.check_safety()
    memory = monitor.summary()["maximum_observed_memory_bytes"]
    assert memory == {"rss_bytes": 2048, "mps_allocated_bytes": 300, "mps_driver_bytes": 700}


def test_existing_output_is_never_overwritten(tmp_path, counters):
    path = tmp_path / "activity.jsonl"
    path.write_text("existing evidence\n")
    with pytest.raises(FileExistsError), telemetry.ActivityMonitor(path, "cpu"):
        pass
    assert path.read_text() == "existing evidence\n"


def test_ioreg_is_targeted_bounded_and_system_wide(monkeypatch):
    calls = []
    payload = [
        {"IORegistryEntryChildren": [{"PerformanceStatistics": {"Device Utilization %": 72}}]},
        {"PerformanceStatistics": {"Device Utilization %": 40}},
    ]

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=plistlib.dumps(payload))

    monkeypatch.setattr(telemetry.sys, "platform", "darwin")
    monkeypatch.setattr(telemetry.subprocess, "run", run)
    assert telemetry._gpu_utilization() == 72.0
    assert calls[0][0] == ["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator"]
    assert calls[0][1]["timeout"] == 2
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize(
    "payload", [[], [{"PerformanceStatistics": {"Device Utilization %": 101}}]]
)
def test_missing_or_invalid_gpu_counter_rejected(monkeypatch, payload):
    monkeypatch.setattr(telemetry.sys, "platform", "darwin")
    monkeypatch.setattr(
        telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=plistlib.dumps(payload)),
    )
    with pytest.raises((RuntimeError, ValueError)):
        telemetry._gpu_utilization()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_rejected(tmp_path, interval):
    with pytest.raises(ValueError):
        telemetry.ActivityMonitor(tmp_path / "activity.jsonl", interval=interval)
