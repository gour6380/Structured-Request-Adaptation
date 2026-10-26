"""Offline execution identity, stage gates and cross-process writer locks."""

from __future__ import annotations

import copy
import multiprocessing
from contextlib import contextmanager
from pathlib import Path

import pytest

from src import execution
from src.config import ExperimentConfig
from src.hardware import _build_plan
from src.io import atomic_json, read_json
from src.runs import RUN_FOLDERS


def hardware(device="cpu", count=0):
    gpus = {
        str(index): {
            "index": index,
            "name": "Fixture GPU",
            "uuid": f"GPU-{index}",
            "total_memory": 48 * 1024**3,
            "capability": [8, 9],
        }
        for index in range(count)
    }
    return _build_plan(
        device,
        gpus,
        {index: True for index in gpus},
        {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_model": "fixture",
            "memory_bytes": 64 * 1024**3,
        },
    )


def environment(plan, index=0):
    return {
        "device": plan["device"],
        "compute_dtype": plan["compute_dtype"],
        "use_grad_scaler": plan["use_grad_scaler"],
        "cuda_device": plan["gpus"].get(str(index)),
        "platform": "host-a",
        "packages": {"torch": "fixture"},
    }


@pytest.fixture
def simulated_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src/train.py").write_text("# source fixture\n")
    (root / "requirements.txt").write_text("# dependency fixture\n")
    plan = hardware("cuda", 2)
    state = {"environment": environment(plan)}
    monkeypatch.setattr(execution, "_environment", lambda plan: copy.deepcopy(state["environment"]))
    run = execution._create_run(
        root, ExperimentConfig.for_hardware(plan, model_revision="a" * 40), plan
    )
    return run, plan, state


def test_run_records_complete_plan_and_source_identity(simulated_run):
    run, plan, _ = simulated_run
    run.verify_identity()
    assert run.active_binding()["hardware"] == plan
    assert (run.path / "source_snapshot/requirements.txt").is_file()
    assert (run.path / "source_snapshot/src/train.py").is_file()
    assert "backend_binding_hash" in run.identity


@pytest.mark.parametrize("relative", ["src/train.py", "requirements.txt"])
def test_changed_source_or_dependencies_cannot_resume(simulated_run, relative):
    run, _, _ = simulated_run
    (run.project_root / relative).write_text("# changed\n")
    with pytest.raises(ValueError, match="identity changed"):
        run.verify_identity()


def test_masked_second_gpu_matches_physical_uuid_without_weakening_environment():
    plan = hardware("cuda", 2)
    recorded = environment(plan)
    current = environment(plan, 1)
    current["cuda_device"] = {**current["cuda_device"], "index": 0}
    execution.validate_worker_environment(current, recorded, {"hardware": plan}, 1)
    current["packages"] = {"torch": "different"}
    with pytest.raises(ValueError, match="Environment/backend"):
        execution.validate_worker_environment(current, recorded, {"hardware": plan}, 1)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_identity_needs_no_gpu_index(device):
    plan = hardware(device)
    record = environment(plan)
    execution.validate_worker_environment(record, record, {"hardware": plan}, None)
    changed = {**record, "compute_dtype": "float16"}
    with pytest.raises(ValueError):
        execution.validate_worker_environment(changed, record, {"hardware": plan}, None)


def test_unavailable_frozen_backend_never_falls_back(monkeypatch):
    monkeypatch.setattr("src.runtime.environment_record", lambda: environment(hardware()))
    with pytest.raises(RuntimeError, match="no automatic fallback"):
        execution._environment(hardware("cuda", 1))


def test_paired_precision_can_use_fp16_on_bf16_capable_second_gpu(monkeypatch):
    plan = hardware("cuda", 2)
    plan = _build_plan("cuda", plan["gpus"], {"0": False, "1": True}, plan["host"])
    native = environment(hardware("cuda", 2), 1)
    monkeypatch.setattr("src.runtime.environment_record", lambda: native)
    record = execution._environment(plan)
    assert record["compute_dtype"] == "float16" and record["use_grad_scaler"] is True


def test_new_gpu_session_requires_explicit_registration_and_retains_original_identity(
    simulated_run,
):
    run, plan, state = simulated_run
    original = (run.path / "identity.json").read_bytes()
    replacement = copy.deepcopy(plan)
    for gpu in replacement["gpus"].values():
        gpu["uuid"] += "-replacement"
    replacement["arm_masks"] = {
        arm: replacement["gpus"][str(index)]["uuid"]
        for arm, index in replacement["arm_gpu"].items()
    }
    replacement["host"]["cpu_model"] = "replacement-host-cpu"
    state["environment"] = environment(replacement)
    state["environment"]["platform"] = "host-b"
    with pytest.raises(ValueError, match="Device identity"):
        run.verify_identity()
    resumed = execution._resume_session(run.project_root, run.path.name, replacement)
    resumed.verify_identity()
    assert (run.path / "identity.json").read_bytes() == original
    assert resumed.active_binding()["hardware"] == replacement


def test_resume_rejects_topology_precision_and_non_cuda_chip_changes():
    plan = hardware("cuda", 2)
    with pytest.raises(ValueError):
        execution._compatible_hardware(plan, hardware("cuda", 1))
    replacement = copy.deepcopy(plan)
    replacement["compute_dtype"] = "float16"
    with pytest.raises(ValueError):
        execution._compatible_hardware(plan, replacement)
    cpu = hardware()
    replacement = copy.deepcopy(cpu)
    replacement["host"]["cpu_model"] = "another CPU"
    with pytest.raises(ValueError):
        execution._compatible_hardware(cpu, replacement)


def _context(root, path, role):
    return execution.ExecutionRunContext(
        Path(root),
        Path(path),
        ExperimentConfig(model_revision="a" * 40),
        {"device": "cpu", "compute_dtype": "float32"},
        role=role,
    )


@pytest.fixture
def lock_project(tmp_path):
    run_path = tmp_path / "runs/run_001"
    for folder in RUN_FOLDERS:
        (run_path / folder).mkdir(parents=True)
    atomic_json(run_path / "manifest.json", {"artifacts": {}})
    return tmp_path, run_path


def _lock_child(root, path, role, sender, release):
    try:
        with _context(root, path, role).writer_lock():
            sender.send(True)
            release.wait(15)
    except BaseException as error:
        sender.send(repr(error))
    finally:
        sender.close()


@contextmanager
def held_lock(root, path, role):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(False)
    release = context.Event()
    process = context.Process(
        target=_lock_child, args=(str(root), str(path), role, sender, release)
    )
    process.start()
    sender.close()
    try:
        assert receiver.poll(15) and receiver.recv() is True
        yield
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        receiver.close()
    assert process.exitcode == 0


@pytest.mark.parametrize("parallel", [True, False])
def test_backend_writer_lock_matches_parallel_or_serial_plan(lock_project, parallel):
    root, path = lock_project
    atomic_json(path / "backend_binding.json", {"hardware": {"parallel": parallel}})
    with held_lock(root, path, "lora"):
        if parallel:
            with _context(root, path, "qlora").writer_lock():
                pass
        else:
            with pytest.raises(RuntimeError):
                with _context(root, path, "qlora").writer_lock():
                    pytest.fail("Serial backend allowed overlapping arms")
        for role in ("lora", "coordinator"):
            with pytest.raises(RuntimeError):
                with _context(root, path, role).writer_lock():
                    pytest.fail("Conflicting backend writer entered")
    with _context(root, path, "coordinator").writer_lock():
        pass


def test_paired_preflight_gate_rejects_missing_second_arm_and_mismatched_initialization(
    simulated_run,
):
    run, _, _ = simulated_run
    first = run.path / "preparation/preflight_lora.json"
    atomic_json(first, {"passed": True, "initial_adapter_sha256": "first"})
    atomic_json(run.path / "stages/preflight_lora.json", {"status": "complete"})
    run.register_artifacts([first])
    with pytest.raises(RuntimeError, match="preflight_qlora"):
        execution._finish_preflight(run)
    assert not (run.path / "preparation/preflight.json").exists()
    second = run.path / "preparation/preflight_qlora.json"
    atomic_json(second, {"passed": True, "initial_adapter_sha256": "different"})
    atomic_json(run.path / "stages/preflight_qlora.json", {"status": "complete"})
    run.register_artifacts([second])
    with pytest.raises(AssertionError, match="same initial adapter"):
        execution._finish_preflight(run)
    assert read_json(run.path / "stages/preflight.json")["status"] == "failed"


def test_arm_stage_gate_rejects_cross_arm_writes(lock_project):
    root, path = lock_project
    run = _context(root, path, "lora")
    run.assert_stage_allowed("arm_lora_pilot")
    run.assert_stage_allowed("evaluate_lora_test")
    with pytest.raises(RuntimeError):
        run.assert_stage_allowed("train_qlora")
