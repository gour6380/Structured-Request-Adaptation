"""Telemetry aggregation checks require no model or accelerator execution."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src import runtime
from src.train import _merge_memory


def interval(*, samples=10, seconds=1.0, memory=100, swap=0, errors=()):
    return {
        "metadata_version": 2,
        "metadata_scope": "single_interval",
        "metadata_complete": True,
        "intervals": 1,
        "samples": samples,
        "elapsed_seconds": seconds,
        "cuda_allocated_bytes": memory,
        "cuda_reserved_bytes": memory * 2,
        "rss_bytes": memory * 3,
        "system_swap_out_bytes_delta": swap,
        "system_swap_activity_observed": swap > 0,
        "sampling_errors": list(errors),
        "sampling_errors_complete": True,
    }


def test_disjoint_intervals_sum_coverage_keep_maxima_errors_and_inputs():
    first = interval(samples=100, seconds=10, memory=800, swap=4, errors=["first error"])
    last = interval(samples=2, seconds=0.002, memory=200, swap=8, errors=["last error"])
    originals = deepcopy((first, last))
    result = _merge_memory(first, last)
    assert result["samples"] == 102
    assert result["elapsed_seconds"] == pytest.approx(10.002)
    assert result["intervals"] == 2
    assert result["cuda_allocated_bytes"] == 800
    assert result["cuda_reserved_bytes"] == 1600
    assert result["rss_bytes"] == 2400
    assert result["system_swap_out_bytes_delta"] == 12
    assert result["system_swap_activity_observed"] is True
    assert result["sampling_errors"] == ["first error", "last error"]
    assert result["sampling_errors_complete"] is True
    assert result["metadata_complete"] is True
    assert (first, last) == originals


def test_resumed_session_repeated_snapshots_use_fixed_committed_prefix():
    committed = _merge_memory(interval(samples=20), interval(samples=30))
    first_snapshot = _merge_memory(committed, interval(samples=5))
    later_snapshot = _merge_memory(committed, interval(samples=9))
    final_snapshot = _merge_memory(committed, interval(samples=11))
    assert first_snapshot["samples"] == 55
    assert later_snapshot["samples"] == 59
    assert final_snapshot["samples"] == 61
    assert final_snapshot["intervals"] == 3


@pytest.mark.parametrize("observed_swap", [False, True])
def test_legacy_metadata_remains_unknown_but_maxima_and_positive_evidence_survive(observed_swap):
    legacy = {
        "samples": 2,
        "elapsed_seconds": 0.002,
        "cuda_allocated_bytes": 900,
        "system_swap_activity_observed": observed_swap,
        "system_swap_out_bytes_delta": 3,
        "sampling_errors": ["last saved error"],
    }
    original = deepcopy(legacy)
    result = _merge_memory(legacy, interval(memory=400))
    for key in ("samples", "elapsed_seconds", "intervals", "system_swap_out_bytes_delta"):
        assert result[key] is None
    assert result["cuda_allocated_bytes"] == 900
    assert result["metadata_complete"] is False
    assert result["sampling_errors_complete"] is False
    assert result["sampling_errors"] == ["last saved error"]
    assert result["system_swap_activity_observed"] is (True if observed_swap else None)
    assert _merge_memory(result, interval())["samples"] is None
    assert legacy == original


def test_empty_aggregate_is_identity_and_does_not_invent_observations():
    assert _merge_memory({}, {}) == {}
    result = _merge_memory({}, interval())
    assert result["samples"] == 10
    assert result["intervals"] == 1
    assert result["metadata_complete"] is True


def test_live_sampler_summary_records_swap_before_exit_and_freezes_after_exit(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(sout=130))
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: 20.0)
    sampler = runtime.MemorySampler("cpu")
    sampler._swap_start = 100
    sampler._start = 10.0
    sampler._count = 10
    live = sampler.summary()
    assert live["system_swap_out_bytes_delta"] == 30
    assert live["system_swap_activity_observed"] is True
    assert live["elapsed_seconds"] == 10
    assert live["metadata_scope"] == "single_interval"
    assert live["metadata_complete"] is True
    sampler._end = 19.0
    sampler._swap_end = 120
    closed = sampler.summary()
    assert closed["system_swap_out_bytes_delta"] == 20
    assert closed["elapsed_seconds"] == 9
