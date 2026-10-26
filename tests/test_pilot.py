"""Exercise the bounded pilot with a tiny CPU fixture, never a pretrained model."""

from __future__ import annotations

import json
import math
import random
import weakref
from functools import wraps
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import pilot
from src.config import ExperimentConfig
from src.io import atomic_json, read_json, sha256_file


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    @wraps(torch.backends.mps.is_available)
    def no_mps():
        return False

    monkeypatch.setattr(torch.backends.mps, "is_available", no_mps)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.embedding.weight.requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(use_cache=False)

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def gradient_checkpointing_disable(self):
        pass

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
        hidden = self.embedding(input_ids)
        if logits_to_keep is not None:
            hidden = hidden.index_select(1, logits_to_keep)
        return SimpleNamespace(logits=self.adapter(hidden))

    def generate(self, input_ids, **kwargs):
        continuation = torch.tensor([[3, 7]] * len(input_ids))
        return torch.cat([input_ids, continuation], dim=1)


class ActivityFixture:
    def __init__(self, output_path, **kwargs):
        self.path = output_path

    def __enter__(self):
        # Use the real monitor's exclusive-create behavior to catch retry bugs.
        with self.path.open("x") as stream:
            stream.write('{"test_cpu_fixture": true}\n')
        return self

    def __exit__(self, *args):
        pass

    def check_safety(self):
        pass

    def summary(self):
        return {"test_cpu_fixture": True}


@pytest.mark.parametrize(
    "training_count,validation_count,microbatch,evaluation_batch,warmups,measured,use_test_scaler",
    [
        (360, 40, 4, 8, 2, 20, False),
        (128, 7, 1, 1, 2, 6, False),
        (20, 2, 1, 1, 1, 1, False),
        (3, 1, 1, 1, 0, 1, False),
        (20, 2, 1, 1, 1, 1, True),
    ],
)
def test_pilot_is_bounded_retryable_and_cannot_change_primary_training(
    tmp_path,
    monkeypatch,
    training_count,
    validation_count,
    microbatch,
    evaluation_batch,
    warmups,
    measured,
    use_test_scaler,
):
    config = ExperimentConfig.development(
        model_revision="a" * 40,
        train_example_limit=training_count,
        validation_example_limit=validation_count,
        microbatch_size=microbatch,
        gradient_accumulation=16 // microbatch,
        evaluation_batch_size=evaluation_batch,
        benchmark_requests=min(16, validation_count),
        benchmark_warmup=min(2, validation_count),
    )
    run = SimpleNamespace(
        path=tmp_path,
        config=config,
        environment={"device": "cpu", "compute_dtype": "float32", "use_grad_scaler": False},
        verify_identity=lambda: None,
        require_stage=lambda name: None,
        assert_training_open=lambda: None,
        verify_artifacts=lambda **kwargs: None,
    )
    atomic_json(tmp_path / "manifest.json", {"artifacts": {}})

    def register(paths):
        manifest = read_json(tmp_path / "manifest.json")
        for path in paths:
            manifest["artifacts"][str(path.relative_to(tmp_path))] = {"sha256": sha256_file(path)}
        atomic_json(tmp_path / "manifest.json", manifest)

    run.register_artifacts = register
    directory = tmp_path / "trials/lora"
    (directory / "pilot").mkdir(parents=True)
    (directory / "pilot/activity.jsonl").write_text("earlier interrupted pilot\n")
    primary = directory / "training_summary.json"
    primary.write_text("primary training must remain unchanged\n")
    initial_path = tmp_path / "initial_adapter/adapter_model.safetensors"
    initial_path.parent.mkdir()
    initial_path.write_bytes(b"initial adapter fixture must remain unchanged")
    training = [
        {
            "id": str(index),
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
            "target_tokens": 2,
            "prompt_ids": [1, 2],
            "target": {"intent": "alarm_set", "slots": []},
        }
        for index in range(training_count)
    ]
    monkeypatch.setattr(
        "src.data.load_examples",
        lambda run, split, **kwargs: training if split == "train" else training[:validation_count],
    )
    monkeypatch.setattr(
        "src.data.load_ontology", lambda run: {"intents": ["alarm_set"], "slots": []}
    )
    monkeypatch.setattr("src.telemetry.ActivityMonitor", ActivityFixture)

    class MemoryFixture:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def summary(self):
            return {"test_cpu_fixture": True}

    monkeypatch.setattr("src.runtime.MemorySampler", MemoryFixture)
    calls = []
    model_references = []
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=7,
        decode=lambda *args, **kwargs: '{"intent":"alarm_set","slots":[]}',
    )

    def load(run, arm, *, checkpoint, training):
        calls.append((arm, checkpoint, training))
        model = TinyModel()
        model_references.append(weakref.ref(model))
        return model, tokenizer

    monkeypatch.setattr("src.models.load_system", load)
    scaler_events = []
    if use_test_scaler:
        # Exercise the real scaled optimization branch on CPU. The separate
        # policy tests establish that only CUDA FP16 selects it in production.
        inner_scaler = torch.amp.GradScaler("cpu", init_scale=128.0)

        class RecordingScaler:
            def scale(self, loss):
                scaler_events.append("scale")
                return inner_scaler.scale(loss)

            def unscale_(self, optimizer):
                scaler_events.append("unscale")
                inner_scaler.unscale_(optimizer)

            def step(self, optimizer):
                assert scaler_events[-1] == "clip"
                scaler_events.append("step")
                return inner_scaler.step(optimizer)

            def update(self):
                scaler_events.append("update")
                inner_scaler.update()

        original_clip = torch.nn.utils.clip_grad_norm_

        def clip(*args, **kwargs):
            assert scaler_events[-1] == "unscale"
            scaler_events.append("clip")
            return original_clip(*args, **kwargs)

        monkeypatch.setattr(pilot, "_gradient_scaler", lambda _: RecordingScaler())
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    original_threads = torch.get_num_threads()
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        torch.set_num_threads(1)
        result = pilot.run_pilot(run, "lora")
    finally:
        torch.set_num_threads(original_threads)
    assert calls == [("lora", "initial", True)]
    assert all(reference() is None for reference in model_references)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    assert result["measured_updates"] == measured
    assert len(result["updates"]) == warmups + measured
    assert result["warmup_updates"] == sum(row["warmup"] for row in result["updates"]) == warmups
    if use_test_scaler:
        for name in ("unscale", "clip", "step", "update"):
            assert scaler_events.count(name) == warmups + measured
        assert scaler_events.count("scale") == training_count
    assert all(row["target_tokens"] == 2 * row["examples"] for row in result["updates"])
    expected_examples = min(training_count, 16 * (warmups + measured))
    assert sum(row["examples"] for row in result["updates"]) == expected_examples
    sampled_validation = min(validation_count, pilot.VALIDATION_EXAMPLES)
    assert result["validation_sample"]["examples"] == sampled_validation
    assert len(result["validation_sample"]["generation_batches"]) == math.ceil(
        sampled_validation / evaluation_batch
    )
    if not warmups:
        assert "cold-start" in result["warmup_qualification"]
    assert result["primary_training_changed"] is False
    assert primary.read_text() == "primary training must remain unchanged\n"
    assert initial_path.read_bytes() == b"initial adapter fixture must remain unchanged"
    assert not (directory / "checkpoints").exists()
    preserved = list((tmp_path / ".recovery").glob("partial-pilot-*/activity.jsonl"))
    assert len(preserved) == 1 and preserved[0].read_text() == "earlier interrupted pilot\n"
    committed = (directory / "pilot/summary.json").read_bytes()
    assert pilot.run_pilot(run, "lora") == json.loads(committed)
    assert len(calls) == 1
    assert (directory / "pilot/summary.json").read_bytes() == committed


@pytest.mark.parametrize(
    "device,dtype", [("cpu", "float32"), ("mps", "bfloat16"), ("cuda", "bfloat16")]
)
def test_non_fp16_policy_never_creates_a_gradient_scaler(monkeypatch, device, dtype):
    monkeypatch.setattr(
        torch.amp, "GradScaler", lambda *_args, **_kwargs: pytest.fail("Unexpected CUDA scaler")
    )
    assert (
        pilot._gradient_scaler({"device": device, "compute_dtype": dtype, "use_grad_scaler": False})
        is None
    )


def test_cuda_fp16_policy_selects_scaler_without_touching_cuda(monkeypatch):
    calls = []
    marker = object()

    def scaler(device):
        calls.append(device)
        return marker

    monkeypatch.setattr(torch.amp, "GradScaler", scaler)
    assert (
        pilot._gradient_scaler(
            {"device": "cuda:0", "compute_dtype": "float16", "use_grad_scaler": True}
        )
        is marker
    )
    assert calls == ["cuda"]


@pytest.mark.parametrize(
    "device,dtype,enabled",
    [
        ("cuda", "float16", False),
        ("cuda", "bfloat16", True),
        ("cpu", "float32", True),
        ("mps", "bfloat16", True),
    ],
)
def test_precision_policy_cannot_silently_change(device, dtype, enabled):
    with pytest.raises(ValueError, match="exactly for CUDA FP16"):
        pilot._gradient_scaler(
            {"device": device, "compute_dtype": dtype, "use_grad_scaler": enabled}
        )


def test_failed_pilot_restores_incoming_random_state(monkeypatch):
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()

    def fail(*_args):
        torch.random.default_generator.manual_seed(41)
        random.seed(42)
        np.random.seed(43)
        raise RuntimeError("synthetic pilot failure")

    monkeypatch.setattr(pilot, "_run_pilot", fail)
    with pytest.raises(RuntimeError, match="synthetic pilot failure"):
        pilot.run_pilot(None, "lora")
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
