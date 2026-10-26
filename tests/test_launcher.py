"""Real CPU subprocess tests of device masking, queues and failure cleanup."""

from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

from src.hardware import _build_plan
from src.launcher import ExperimentLauncher


def plan_for(device="cpu", count=0):
    gpus = {
        str(index): {
            "index": index,
            "name": "Fixture GPU",
            "uuid": f"GPU-fixture-{index}",
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


@pytest.fixture
def worker_project(tmp_path):
    package = tmp_path / "fake_backend"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent("""
        import os, sys, time, json
        from pathlib import Path
        def execute(request):
            root = Path(request['project_root'])
            action = request['action']
            if action == 'inspect_hardware':
                return json.loads((root / 'hardware.json').read_text())
            arm = request.get('arm', 'coordinator')
            if action == 'run_arm':
                mode = (root / 'mode').read_text()
                if mode == 'failure' and arm == 'qlora':
                    deadline = time.monotonic() + 10
                    while not (root / 'started_lora').exists():
                        if time.monotonic() > deadline: raise TimeoutError('missing peer')
                        time.sleep(0.01)
                    raise RuntimeError('intentional worker failure')
                if mode == 'failure_first' and arm == 'lora':
                    time.sleep(0.2)
                    raise RuntimeError('intentional first failure')
                (root / ('started_' + arm)).write_text(str(time.monotonic()))
                if mode in {'blocked', 'failure'}:
                    while True: time.sleep(0.01)
                time.sleep(0.2)
                (root / ('ended_' + arm)).write_text(str(time.monotonic()))
            return {'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid(),
                    'python': sys.executable,
                    'torch_loaded': 'torch' in sys.modules, 'action': action, 'arm': arm}
    """)
    )
    (tmp_path / "mode").write_text("success")
    (tmp_path / "hardware.json").write_text(json.dumps(plan_for()))
    return tmp_path


def launcher_for(root, plan):
    (root / "hardware.json").write_text(json.dumps(plan))
    launcher = ExperimentLauncher(root, python_executable=sys.executable)
    launcher.backend_module = "fake_backend"
    real_wait = launcher.wait
    launcher.wait = lambda workers, **kwargs: real_wait(workers, poll_seconds=0.01)
    assert launcher.inspect_hardware() == plan
    return launcher


def test_default_launcher_runs_with_current_interpreter_without_project_environment(worker_project):
    assert not (worker_project / ".venv").exists()
    launcher = ExperimentLauncher(worker_project)
    launcher.backend_module = "fake_backend"
    real_wait = launcher.wait
    launcher.wait = lambda workers, **kwargs: real_wait(workers, poll_seconds=0.01)
    try:
        assert launcher.inspect_hardware() == plan_for()
        worker = launcher.start_arm("run_001", "lora", "pilot")
        result = launcher.wait([worker])["lora"]
        assert result["python"] == sys.executable
        assert result["pid"] != os.getpid()
        assert not (worker_project / ".venv").exists()
    finally:
        launcher.cancel_all()


@pytest.mark.parametrize("device,count", [("cpu", 0), ("mps", 0), ("cuda", 1)])
def test_serial_arms_queue_and_never_overlap(worker_project, device, count):
    launcher = launcher_for(worker_project, plan_for(device, count))
    try:
        first = launcher.start_arm("run_001", "lora", "pilot")
        second = launcher.start_arm("run_001", "qlora", "pilot")
        assert second.process is None
        assert second.status()["status"] == "queued"
        assert (
            json.loads((second.result_path.parent / "process.json").read_text())["status"]
            == "queued"
        )
        result = launcher.wait([first, second])
        assert float((worker_project / "started_qlora").read_text()) >= float(
            (worker_project / "ended_lora").read_text()
        )
        assert result["lora"]["pid"] != result["qlora"]["pid"]
        expected_mask = "GPU-fixture-0" if device == "cuda" else ""
        assert all(
            value["visible"] == expected_mask and not value["torch_loaded"]
            for value in result.values()
        )
    finally:
        launcher.cancel_all()


def test_waiting_only_queued_arm_also_waits_for_its_dependency(worker_project):
    launcher = launcher_for(worker_project, plan_for())
    try:
        first = launcher.start_arm("run_001", "lora")
        second = launcher.start_arm("run_001", "qlora")
        assert launcher.wait([second])["qlora"]["arm"] == "qlora"
        assert first.status()["status"] == "complete"
    finally:
        launcher.cancel_all()


def test_parallel_cuda_masks_are_set_before_import_and_failure_cleans_peer(worker_project):
    launcher = launcher_for(worker_project, plan_for("cuda", 2))
    try:
        (worker_project / "mode").write_text("failure")
        workers = [launcher.start_arm("run_001", arm) for arm in ("lora", "qlora")]
        assert all(worker.process is not None for worker in workers)
        with pytest.raises(RuntimeError, match="intentional worker failure"):
            launcher.wait(workers)
        assert all(worker.process.poll() is not None for worker in workers)
        (worker_project / "mode").write_text("success")
        result = launcher.wait([launcher.start_arm("run_001", arm) for arm in ("lora", "qlora")])
        assert [result[arm]["visible"] for arm in ("lora", "qlora")] == [
            "GPU-fixture-0",
            "GPU-fixture-1",
        ]
        assert all(not value["torch_loaded"] for value in result.values())
    finally:
        launcher.cancel_all()


def test_failed_first_arm_cancels_queued_arm_without_launching_it(worker_project):
    launcher = launcher_for(worker_project, plan_for())
    try:
        (worker_project / "mode").write_text("failure_first")
        first = launcher.start_arm("run_001", "lora")
        second = launcher.start_arm("run_001", "qlora")
        with pytest.raises(RuntimeError, match="intentional first failure"):
            launcher.wait([first, second])
        assert second.process is None and second.status()["status"] == "cancelled"
        assert not (worker_project / "started_qlora").exists()
    finally:
        launcher.cancel_all()


def test_cancel_queued_job_never_starts_it_and_releases_running_peer(worker_project):
    launcher = launcher_for(worker_project, plan_for())
    try:
        (worker_project / "mode").write_text("blocked")
        first = launcher.start_arm("run_001", "lora")
        second = launcher.start_arm("run_001", "qlora")
        launcher.cancel_all()
        assert first.process.poll() is not None
        assert second.process is None and second.cancelled
        assert not (worker_project / "started_qlora").exists()
    finally:
        launcher.cancel_all()


def test_other_controller_cannot_overlap_serial_device_or_coordinator(worker_project):
    first = launcher_for(worker_project, plan_for())
    second = launcher_for(worker_project, plan_for())
    try:
        (worker_project / "mode").write_text("blocked")
        first.start_arm("run_001", "lora")
        with pytest.raises(RuntimeError, match="busy"):
            second.start_arm("run_001", "qlora")
        with pytest.raises(RuntimeError, match="busy"):
            second.run_action("run_summary", run_name="run_001")
    finally:
        first.cancel_all()
        second.cancel_all()


def test_pending_duplicate_and_coordinator_actions_fail_without_deadlock(worker_project):
    launcher = launcher_for(worker_project, plan_for())
    try:
        (worker_project / "mode").write_text("blocked")
        launcher.start_arm("run_001", "lora")
        launcher.start_arm("run_001", "qlora")
        with pytest.raises(RuntimeError, match="pending"):
            launcher.start_arm("run_001", "qlora")
        with pytest.raises(RuntimeError, match="pending"):
            launcher.run_action("freeze_evaluation", run_name="run_001")
    finally:
        launcher.cancel_all()


def test_coordinator_preserves_inherited_visibility_and_prediction_masks_selected_gpu(
    worker_project, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture-0,GPU-fixture-1")
    launcher = launcher_for(worker_project, plan_for("cuda", 2))
    try:
        created = launcher.run_action("create_run", config={})
        assert created["visible"] == "GPU-fixture-0,GPU-fixture-1"
        assert (
            launcher.run_action("predict", run_name="run_001", arm="qlora", text="fixture")[
                "visible"
            ]
            == "GPU-fixture-1"
        )
    finally:
        launcher.cancel_all()


def test_process_record_failure_reaps_newly_spawned_child(worker_project, monkeypatch):
    import src.launcher as module

    launcher = launcher_for(worker_project, plan_for())
    original = module._atomic_json

    def failing(path, value):
        if path.name == "process.json":
            raise OSError("intentional status failure")
        return original(path, value)

    monkeypatch.setattr(module, "_atomic_json", failing)
    try:
        with pytest.raises(OSError, match="status failure"):
            launcher.start_arm("run_001", "lora")
        assert launcher._workers[-1].process.poll() is not None
    finally:
        monkeypatch.setattr(module, "_atomic_json", original)
        launcher.cancel_all()


def test_status_failure_during_cleanup_never_leaves_running_peer(worker_project, monkeypatch):
    import src.launcher as module

    launcher = launcher_for(worker_project, plan_for("cuda", 2))
    (worker_project / "mode").write_text("failure")
    first = launcher.start_arm("run_001", "lora")
    failed = launcher.start_arm("run_001", "qlora")
    original = module._atomic_json

    def fail_status(path, value):
        if path == failed.result_path.parent / "process.json":
            raise OSError("intentional cleanup bookkeeping failure")
        return original(path, value)

    monkeypatch.setattr(module, "_atomic_json", fail_status)
    try:
        with pytest.raises(RuntimeError, match="intentional worker failure"):
            launcher.wait([failed, first])
        assert failed.process.poll() is not None
        assert first.process.poll() is not None
    finally:
        monkeypatch.setattr(module, "_atomic_json", original)
        launcher.cancel_all()
