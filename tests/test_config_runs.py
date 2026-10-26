"""Authored handoff checks. Deliberately not executed during implementation."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.config import ExperimentConfig
from src.hardware import _build_plan
from src.io import atomic_json, read_json
from src.runs import create_run, open_run


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text('"""Test fixture."""\n')
    (tmp_path / "requirements.txt").write_text("torch==2.14.0\n")
    monkeypatch.setattr(
        "src.runs.environment_record", lambda: {"device": "cpu", "compute_dtype": "float32"}
    )
    return tmp_path


def test_config_round_trip():
    config = ExperimentConfig(model_revision="a" * 40)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "main"},
        {"epochs": 0},
        {"microbatch_size": 2},
        {"evaluation_batch_size": 0},
        {"evaluation_batch_size": True},
        {"evaluation_batch_size": 1.5},
        {"learning_rate": float("nan")},
        {"max_new_tokens": 2048},
        {"seed": True},
        {"seed": 2027.0},
        {"seed": 2**32},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(model_revision="a" * 40), **change).validate()


def hardware_plan(device="cuda", memory_gib=48):
    gpu = {
        "index": 0,
        "name": "Example CUDA GPU",
        "uuid": "GPU-example",
        "total_memory": memory_gib * 1024**3,
        "capability": [8, 9],
    }
    return _build_plan(
        device,
        {"0": gpu} if device == "cuda" else {},
        {"0": True} if device == "cuda" else {},
        {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_model": "Example CPU",
            "memory_bytes": 64 * 1024**3,
        },
    )


def test_hardware_batches_preserve_full_budget_and_existing_defaults():
    local = ExperimentConfig(model_revision="a" * 40)
    development = ExperimentConfig.development(model_revision="a" * 40)
    large_gpu = ExperimentConfig.for_hardware(hardware_plan(), model_revision="a" * 40)
    assert (local.microbatch_size, local.gradient_accumulation, local.evaluation_batch_size) == (
        1,
        16,
        1,
    )
    assert (development.microbatch_size, development.gradient_accumulation) == (1, 16)
    assert development.evaluation_batch_size == 1
    assert (large_gpu.preset, large_gpu.epochs) == ("full", 5)
    assert (
        large_gpu.microbatch_size,
        large_gpu.gradient_accumulation,
        large_gpu.evaluation_batch_size,
    ) == (
        4,
        4,
        8,
    )
    assert large_gpu.effective_batch_size == local.effective_batch_size == 16
    assert large_gpu.train_example_limit is None and large_gpu.validation_example_limit is None
    assert ExperimentConfig.from_dict(large_gpu.to_dict()) == large_gpu
    assert ExperimentConfig.for_hardware(hardware_plan()).model_revision == ""


@pytest.mark.parametrize("device,memory", [("cuda", 16), ("mps", 0), ("cpu", 0)])
def test_small_device_factory_preserves_effective_batch_with_low_memory_defaults(device, memory):
    config = ExperimentConfig.for_hardware(hardware_plan(device, memory))
    assert (config.microbatch_size, config.gradient_accumulation, config.evaluation_batch_size) == (
        1,
        16,
        1,
    )


def test_hardware_factory_rejects_changed_plan():
    plan = hardware_plan()
    plan["batch_defaults"]["microbatch_size"] = 8
    with pytest.raises(ValueError, match="policy"):
        ExperimentConfig.for_hardware(plan)


@pytest.mark.parametrize("device,memory,microbatch", [("cuda", 48, 4), ("mps", 0, 1)])
def test_hardware_factory_development_preserves_small_workflow_defaults(device, memory, microbatch):
    config = ExperimentConfig.for_hardware(
        hardware_plan(device, memory), preset="development", model_revision="a" * 40
    )
    assert (config.preset, config.epochs) == ("development", 1)
    assert (config.train_example_limit, config.validation_example_limit) == (128, 32)
    assert (config.benchmark_requests, config.benchmark_warmup, config.benchmark_repetitions) == (
        16,
        2,
        1,
    )
    assert config.microbatch_size == microbatch
    assert config.effective_batch_size == 16
    assert config.evaluation_batch_size == (8 if device == "cuda" else 1)


@pytest.mark.parametrize("microbatch,accumulation", [(2, 8), (4, 4), (8, 2), (16, 1)])
def test_full_batch_implementations_share_effective_budget(microbatch, accumulation):
    config = ExperimentConfig.for_hardware(
        hardware_plan(), microbatch_size=microbatch, gradient_accumulation=accumulation
    )
    assert config.effective_batch_size == 16


def test_full_batch_budget_change_requires_a_different_protocol():
    with pytest.raises(ValueError, match="effective batch"):
        ExperimentConfig.for_hardware(hardware_plan(), microbatch_size=8, gradient_accumulation=4)


def test_allocation_never_reuses_deleted_gap(project):
    config = ExperimentConfig(model_revision="a" * 40)
    first = create_run(project, config)
    (project / "runs" / "run_003").mkdir()
    next_run = create_run(project, config)
    assert first.path.name == "run_001"
    assert next_run.path.name == "run_004"


def test_existing_run_and_source_identity(project):
    run = create_run(project, ExperimentConfig(model_revision="a" * 40))
    assert open_run(project, run.path.name).config == run.config
    (project / "src" / "train.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="Source"):
        open_run(project, run.path.name)


def test_registered_artifact_changes_are_detected(project):
    run = create_run(project, ExperimentConfig(model_revision="a" * 40))
    path = run.path / "preparation" / "fixture.json"
    atomic_json(path, {"a": 1})
    run.register_artifacts([path])
    atomic_json(path, {"a": 2})
    with pytest.raises(ValueError):
        run.verify_artifacts()
    with pytest.raises(ValueError):
        run.register_artifacts([path])
    run.register_artifacts([path], replace=True)
    run.verify_artifacts()


def test_stage_records_interrupt_and_can_resume(project):
    run = create_run(project, ExperimentConfig(model_revision="a" * 40))
    with pytest.raises(KeyboardInterrupt):
        with run.stage("fixture"):
            raise KeyboardInterrupt
    assert read_json(run.path / "stages" / "fixture.json")["status"] == "interrupted"
    with run.stage("fixture"):
        pass
    assert run.require_stage("fixture")["status"] == "complete"


def test_freeze_blocks_training(project):
    run = create_run(project, ExperimentConfig(model_revision="a" * 40))
    atomic_json(run.path / "evaluations" / "frozen_protocol.json", {})
    with pytest.raises(RuntimeError, match="frozen"):
        run.assert_training_open()


def test_paths_must_stay_inside_run(project):
    run = create_run(project, ExperimentConfig(model_revision="a" * 40))
    with pytest.raises(ValueError):
        run.register_artifacts([Path(project) / "requirements.txt"])
