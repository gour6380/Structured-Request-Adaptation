"""Atomicity, retention, hash verification, and RNG resume using tiny models."""

import random

import numpy as np
import pytest
import torch

from src.checkpoints import capture_rng, resolve_checkpoint, restore_rng, save_checkpoint
from src.io import atomic_json, read_json


class DummyRun:
    def __init__(self, root):
        self.path = root
        self.registered = []
        atomic_json(root / "manifest.json", {"artifacts": {}})

    def register_artifacts(self, paths, replace=False):
        from src.io import sha256_file

        self.registered.extend(paths)
        manifest = read_json(self.path / "manifest.json")
        for path in paths:
            manifest["artifacts"][str(path.relative_to(self.path))] = {"sha256": sha256_file(path)}
        atomic_json(self.path / "manifest.json", manifest)

    def verify_artifacts(self, prefixes=None):
        from src.io import sha256_file

        for name, record in read_json(self.path / "manifest.json")["artifacts"].items():
            if prefixes is None or name in prefixes:
                assert sha256_file(self.path / name) == record["sha256"]


class TinyAdapter:
    def save_pretrained(self, path, **kwargs):
        path.mkdir(parents=True)
        atomic_json(path / "adapter_config.json", {"r": 1})
        (path / "adapter_model.safetensors").write_bytes(b"test-only-adapter")


def setup_objects(tmp_path):
    run = DummyRun(tmp_path)
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    return run, TinyAdapter(), optimizer, scheduler


def test_rng_resume_restores_python_numpy_and_torch():
    state = capture_rng()
    expected = (random.random(), float(np.random.rand()), torch.rand(5))
    restore_rng(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(5))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])


def test_checkpoint_detects_corrupt_adapter(tmp_path):
    run, model, optimizer, scheduler = setup_objects(tmp_path)
    path = save_checkpoint(
        run,
        "lora",
        model,
        optimizer,
        scheduler,
        None,
        {"global_step": 1, "epoch": 0},
        name="step_00000001",
    )
    assert resolve_checkpoint(run, "lora", "last") == path
    (path / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        resolve_checkpoint(run, "lora", "last")


def test_retention_keeps_last_two_and_best_and_final(tmp_path):
    run, model, optimizer, scheduler = setup_objects(tmp_path)
    for step in range(1, 6):
        save_checkpoint(
            run,
            "qlora",
            model,
            optimizer,
            scheduler,
            None,
            {"global_step": step, "epoch": step},
            name=f"step_{step}",
            best=step == 1,
        )
    save_checkpoint(
        run,
        "qlora",
        model,
        optimizer,
        scheduler,
        None,
        {"global_step": 5, "epoch": 5},
        name="final",
        final=True,
    )
    root = tmp_path / "trials" / "qlora" / "checkpoints"
    index = read_json(root / "index.json")
    assert index["best"] == "step_1" and index["final"] == "final"
    assert set(index["entries"]) == {"step_1", "step_5", "final"}
    assert not (root / "step_2").exists()


def test_failed_save_never_publishes_checkpoint(tmp_path):
    run, _, optimizer, scheduler = setup_objects(tmp_path)

    class FailedAdapter:
        def save_pretrained(self, path, **kwargs):
            raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(
            run,
            "lora",
            FailedAdapter(),
            optimizer,
            scheduler,
            None,
            {"global_step": 1, "epoch": 0},
            name="broken",
        )
    assert not (tmp_path / "trials" / "lora" / "checkpoints" / "broken").exists()
    assert not (tmp_path / "trials" / "lora" / "checkpoints" / "index.json").exists()


def test_unregistered_index_is_ignored_after_commit_interruption(tmp_path):
    from src.checkpoints import read_index

    run, model, optimizer, scheduler = setup_objects(tmp_path)
    real_register = run.register_artifacts

    def fail_register(*args, **kwargs):
        raise OSError("manifest commit interrupted")

    run.register_artifacts = fail_register
    with pytest.raises(OSError, match="commit interrupted"):
        save_checkpoint(
            run,
            "lora",
            model,
            optimizer,
            scheduler,
            None,
            {"global_step": 1, "epoch": 0},
            name="step_1",
        )
    assert read_index(run, "lora")["last"] is None
    run.register_artifacts = real_register
    path = save_checkpoint(
        run,
        "lora",
        model,
        optimizer,
        scheduler,
        None,
        {"global_step": 1, "epoch": 0},
        name="step_1",
    )
    assert resolve_checkpoint(run, "lora", "last") == path
    assert len(list(path.parent.glob(".orphan-step_1-*"))) == 1


def test_convenience_index_corruption_does_not_change_committed_pointer(tmp_path):
    run, model, optimizer, scheduler = setup_objects(tmp_path)
    path = save_checkpoint(
        run,
        "lora",
        model,
        optimizer,
        scheduler,
        None,
        {"global_step": 1, "epoch": 0},
        name="step_1",
    )
    atomic_json(path.parent / "index.json", {"last": "untrusted"})
    assert resolve_checkpoint(run, "lora", "last") == path


def test_optimizer_resume_matches_uninterrupted_next_update(tmp_path):
    from safetensors.torch import load_file, save_file

    from src.checkpoints import restore_training_state

    run = DummyRun(tmp_path)
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.7]))

    class ParameterAdapter:
        def save_pretrained(self, path, **kwargs):
            path.mkdir(parents=True)
            save_file({"weight": parameter.detach()}, path / "adapter_model.safetensors")

    optimizer = torch.optim.AdamW([parameter], lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))

    def update(p, opt, schedule):
        opt.zero_grad(set_to_none=True)
        (p.square().sum()).backward()
        opt.step()
        schedule.step()

    update(parameter, optimizer, scheduler)
    checkpoint = save_checkpoint(
        run,
        "lora",
        ParameterAdapter(),
        optimizer,
        scheduler,
        None,
        {"global_step": 1, "epoch": 0},
        name="step_1",
    )
    update(parameter, optimizer, scheduler)
    expected = parameter.detach().clone()
    resumed = torch.nn.Parameter(
        load_file(checkpoint / "adapter" / "adapter_model.safetensors")["weight"]
    )
    resumed_optimizer = torch.optim.AdamW([resumed], lr=0.02)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lambda step: 1 / (step + 1)
    )
    restore_training_state(checkpoint, resumed_optimizer, resumed_scheduler, None, "cpu")
    update(resumed, resumed_optimizer, resumed_scheduler)
    torch.testing.assert_close(resumed, expected, rtol=0, atol=0)
