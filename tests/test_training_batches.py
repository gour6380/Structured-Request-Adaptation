"""Batched optimizer boundaries and interruption recovery with a tiny CPU table."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from src.config import ExperimentConfig
from src.io import atomic_json, read_jsonl
from src.train import _train_adapter, accumulation_windows, epoch_order, microbatch_windows


def test_batching_keeps_original_update_boundaries_and_final_partial_window():
    order = epoch_order(35, 2027, 0)
    windows = list(accumulation_windows(order, 0, 16))
    batches = [list(microbatch_windows(window, 4)) for _, window in windows]
    assert [[len(batch) for batch in group] for group in batches] == [[4] * 4, [4] * 4, [3]]
    assert [index for group in batches for batch in group for index in batch] == order
    assert list(accumulation_windows(order, 16, 16)) == windows[1:]
    assert list(accumulation_windows(order, len(order), 16)) == []
    with pytest.raises(ValueError, match="boundary"):
        list(accumulation_windows(order, 4, 16))


def test_batched_training_and_resumed_partial_window_match_uninterrupted_updates(
    tmp_path, monkeypatch
):
    """Exercise the real training loop; replace only model I/O and external telemetry."""
    rows = []
    for index in range(35):
        tokens = [index + 1, 1, *range(2, 4 + index % 4)]
        labels = [-100, -100, *tokens[2:]]
        rows.append(
            {
                "id": str(index),
                "input_ids": tokens,
                "labels": labels,
                "target_tokens": len(tokens) - 2,
            }
        )
    initial = torch.randn(64, 64, generator=torch.Generator().manual_seed(77))
    models, latest, snapshots, saves = {}, {}, {}, {}
    interrupted_root = tmp_path / "interrupted"
    interrupted_once = False

    class TinyModel(torch.nn.Module):
        def __init__(self, interrupt=False):
            super().__init__()
            self.table = torch.nn.Parameter(initial.clone())
            self.config = SimpleNamespace(use_cache=False)
            self.calls = []
            self.interrupt = interrupt

        def gradient_checkpointing_enable(self, **kwargs):
            pass

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=None):
            if self.interrupt and len(self.calls) == 5:
                raise KeyboardInterrupt("test interruption inside the second optimizer window")
            self.calls.append(input_ids[:, 0].tolist())
            assert torch.all(attention_mask[:, 0] == 1)
            logits = self.table[input_ids]
            if logits_to_keep is not None:
                logits = logits[:, logits_to_keep]
            return SimpleNamespace(logits=logits)

    class Memory:
        def __init__(self, *args, **kwargs):
            self.count = 0

        def __enter__(self):
            self.count += 1
            return self

        def __exit__(self, *args):
            self.count += 1

        def sample(self):
            self.count += 1
            return {}

        def summary(self):
            return {
                "metadata_version": 2,
                "metadata_complete": True,
                "metadata_scope": "single_interval",
                "intervals": 1,
                "samples": self.count,
                "elapsed_seconds": self.count / 10,
                "system_swap_out_bytes_delta": 0,
                "system_swap_activity_observed": False,
                "sampling_errors": [],
            }

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    def load(run, arm, checkpoint, training):
        nonlocal interrupted_once
        interrupt = run.path == interrupted_root and not interrupted_once
        interrupted_once |= interrupt
        model = TinyModel(interrupt)
        if checkpoint == "last":
            model.load_state_dict(snapshots[latest[run.path]]["model"])
        models[run.path] = model
        return model, SimpleNamespace(pad_token_id=0)

    def save(run, arm, model, optimizer, scheduler, scaler, state, *, name, **kwargs):
        destination = run.path / "trials" / arm / "checkpoints" / name
        destination.mkdir(parents=True, exist_ok=True)
        snapshots[destination] = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "state": copy.deepcopy(state),
        }
        latest[run.path] = destination
        saves.setdefault(run.path, []).append(copy.deepcopy(state))
        return destination

    def restore(checkpoint, optimizer, scheduler, scaler, device):
        saved = snapshots[checkpoint]
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        return copy.deepcopy(saved["state"])

    monkeypatch.setattr("src.data.load_examples", lambda *args, **kwargs: rows)
    monkeypatch.setattr("src.models.load_system", load)
    monkeypatch.setattr("src.models.describe_model", lambda *args: {"model": "tiny CPU test"})
    monkeypatch.setattr("src.models._registered_files", lambda *args: False)
    monkeypatch.setattr("src.runtime.MemorySampler", Memory)
    monkeypatch.setattr("src.runtime.synchronize", lambda *args: None)
    monkeypatch.setattr("src.runtime.release_memory", lambda *args: None)
    monkeypatch.setattr("src.runtime.seed_everything", lambda *args: None)
    monkeypatch.setattr("torch.utils.tensorboard.SummaryWriter", Writer)
    monkeypatch.setattr(
        "src.evaluate.evaluate_loaded",
        lambda *args: {
            "metrics": {"joint_exact_match": 0.5, "answer_token_nll": 1.0},
            "predictions": [],
        },
    )
    monkeypatch.setattr("src.train.save_checkpoint", save)
    monkeypatch.setattr("src.train.restore_training_state", restore)
    monkeypatch.setattr("src.train.read_index", lambda run, arm: {"last": latest.get(run.path)})
    monkeypatch.setattr("src.train.resolve_checkpoint", lambda run, arm, kind: latest[run.path])

    def run_context(root):
        atomic_json(
            root / "preparation" / "preflight.json",
            {"passed": True, "initial_adapter_sha256": "test"},
        )
        return SimpleNamespace(
            path=root,
            config=ExperimentConfig.development(
                model_revision="a" * 40,
                train_example_limit=35,
                epochs=2,
                microbatch_size=4,
                gradient_accumulation=4,
                checkpoint_interval=1,
            ),
            environment={"device": "cpu", "compute_dtype": "float32", "use_grad_scaler": False},
            writer_lock=nullcontext,
            stage=lambda name: nullcontext(),
            verify_identity=lambda: None,
            assert_training_open=lambda: None,
            require_stage=lambda name: None,
            verify_artifacts=lambda **kwargs: None,
            register_artifacts=lambda *args, **kwargs: None,
        )

    reference_run = run_context(tmp_path / "reference")
    reference = _train_adapter(reference_run, "lora")
    # Each epoch has three one-update samplers (three samples apiece) and
    # an empty sampler after its final checkpoint (enter/exit only).
    assert reference["training_memory"]["samples"] == 2 * (3 * 3 + 2)
    assert reference["training_memory"]["intervals"] == 8
    # The whole-fit sampler has one enter and one exit. It must not be
    # re-added to the earlier snapshot already committed in state["memory"].
    assert reference["fit_memory"]["samples"] == 2
    assert reference["fit_memory"]["intervals"] == 1
    interrupted = run_context(interrupted_root)
    with pytest.raises(KeyboardInterrupt, match="second optimizer window"):
        _train_adapter(interrupted, "lora")
    assert saves[interrupted_root][-1]["global_step"] == 1
    assert saves[interrupted_root][-1]["next_example"] == 16
    resumed = _train_adapter(interrupted, "lora")
    assert resumed["training_memory"]["samples"] == reference["training_memory"]["samples"]
    assert resumed["training_memory"]["intervals"] == reference["training_memory"]["intervals"]
    # One committed sample from the interrupted fit, then enter/exit in
    # the resumed segment; uncommitted interrupted work is not invented.
    assert resumed["fit_memory"]["samples"] == 3
    assert resumed["fit_memory"]["intervals"] == 2
    assert resumed["optimizer_updates"] == reference["optimizer_updates"] == 6
    assert resumed["examples_seen"] == reference["examples_seen"] == 70
    assert resumed["target_tokens_seen"] == sum(row["target_tokens"] for row in rows) * 2
    assert resumed["effective_batch_size"] == 16
    assert [len(batch) for batch in models[reference_run.path].calls] == ([4] * 8 + [3]) * 2
    resumed_ids = [token - 1 for batch in models[interrupted_root].calls for token in batch]
    assert resumed_ids == epoch_order(35, 2027, 0)[16:] + epoch_order(35, 2027, 1)
    torch.testing.assert_close(
        models[interrupted_root].table, models[reference_run.path].table, rtol=0, atol=0
    )
    history = read_jsonl(interrupted_root / "trials" / "lora" / "metrics" / "train.jsonl")
    assert [record["examples"] for record in history] == [16, 16, 3] * 2
    assert [record["global_step"] for record in history] == list(range(1, 7))
