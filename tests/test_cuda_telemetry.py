"""GPU activity uses UUIDs so CUDA_VISIBLE_DEVICES does not swap arm readings."""

from types import SimpleNamespace

import pytest
import torch

from src import telemetry


def test_cuda_activity_targets_physical_uuid(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-arm-one")
    )
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="86\n")

    monkeypatch.setattr(telemetry.subprocess, "run", query)
    assert telemetry._cuda_gpu_utilization("cuda:0") == 86.0
    assert "--id=GPU-arm-one" in calls[0][0]
    assert calls[0][1]["timeout"] == 2


@pytest.mark.parametrize("value", ["N/A", "nan", "101", "-1", "10\n20"])
def test_cuda_activity_rejects_ambiguous_or_invalid_readings(monkeypatch, value):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-arm-one")
    )
    monkeypatch.setattr(
        telemetry.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=value)
    )
    with pytest.raises(ValueError):
        telemetry._cuda_gpu_utilization("cuda:0")


def test_missing_uuid_cannot_silently_sample_the_other_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace())
    with pytest.raises(RuntimeError, match="UUID"):
        telemetry._cuda_gpu_utilization("cuda:0")
