"""Run explicitly after implementation review; never part of environment setup."""

import pytest
import torch

from src.io import atomic_json, canonical_hash, read_jsonl, sha256_file, write_jsonl
from src.train import (
    accumulation_windows,
    epoch_order,
    is_better,
    reconcile_logs,
    replay_tensorboard_boundary,
    summed_answer_loss,
    supervised_count,
)


def test_token_weighted_accumulation_matches_concatenated_gradient():
    torch.manual_seed(9)
    parameter = torch.nn.Parameter(torch.randn(3, 7))
    examples = [
        (torch.tensor([[1, 2, 3]]), torch.tensor([[-100, 2, 3]])),
        (torch.tensor([[2, 1, 0]]), torch.tensor([[-100, -100, 4]])),
    ]
    # Differing answer lengths expose the incorrect mean-of-microbatch-means rule.
    sums = [summed_answer_loss(parameter.unsqueeze(0), labels) for _, labels in examples]
    total_tokens = sum(count for _, count in sums)
    reference = torch.autograd.grad(sum(loss for loss, _ in sums) / total_tokens, parameter)[0]
    parameter.grad = None
    for _, labels in examples:
        loss, _ = summed_answer_loss(parameter.unsqueeze(0), labels)
        (loss / total_tokens).backward()
    torch.testing.assert_close(parameter.grad, reference)


def test_masked_prompt_and_padding_have_no_loss_gradient():
    logits = torch.randn(1, 5, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, -100]])
    numerator, tokens = summed_answer_loss(logits, labels)
    numerator.backward()
    assert tokens == 2
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 3:]) == 0
    assert torch.count_nonzero(logits.grad[:, 1:3]) > 0


def test_empty_supervision_is_rejected():
    with pytest.raises(ValueError, match="no supervised"):
        summed_answer_loss(torch.zeros(1, 3, 5), torch.full((1, 3), -100))


def test_final_partial_accumulation_keeps_every_example():
    order = epoch_order(35, 2027, 0)
    windows = list(accumulation_windows(order, 0, 16))
    assert [len(indices) for _, indices in windows] == [16, 16, 3]
    assert [index for _, indices in windows for index in indices] == order
    resumed = list(accumulation_windows(order, 16, 16))
    assert resumed == windows[1:]


def test_epoch_order_does_not_consume_global_rng():
    import random

    random.seed(23)
    before = random.getstate()
    assert epoch_order(31, 2027, 0) == epoch_order(31, 2027, 0)
    assert epoch_order(31, 2027, 1) != epoch_order(31, 2027, 0)
    assert random.getstate() == before


def test_best_checkpoint_earliest_exact_tie():
    best = {"joint_exact_match": 0.8, "answer_token_nll": 0.4}
    assert not is_better(dict(best), best)
    assert is_better({"joint_exact_match": 0.8, "answer_token_nll": 0.3}, best)
    assert is_better({"joint_exact_match": 0.81, "answer_token_nll": 0.7}, best)


def test_log_recovery_truncates_only_uncommitted_suffix(tmp_path):
    first = {"global_step": 1, "loss": 0.2}
    write_jsonl(tmp_path / "train.jsonl", [first, {"global_step": 2, "loss": 0.1}])
    result = reconcile_logs(
        tmp_path, {"journal_length": 1, "journal_hash": canonical_hash([first])}
    )
    assert result == [first] == read_jsonl(tmp_path / "train.jsonl")
    write_jsonl(tmp_path / "train.jsonl", [{"global_step": 1, "loss": 99}])
    with pytest.raises(ValueError, match="modified"):
        reconcile_logs(tmp_path, {"journal_length": 1, "journal_hash": canonical_hash([first])})


def test_supervision_count_disagreement_rejected():
    with pytest.raises(ValueError, match="accounting"):
        supervised_count({"id": "a", "input_ids": [1, 2], "labels": [-100, 2], "target_tokens": 2})


def test_recovery_quarantines_uncommitted_epoch_results(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    predictions = metrics / "epoch_01_predictions.jsonl"
    write_jsonl(predictions, [{"prediction": "committed"}])
    epoch = {"epoch": 1, "metrics": {}, "predictions_sha256": sha256_file(predictions)}
    atomic_json(metrics / "epoch_01.json", epoch)
    atomic_json(metrics / "epoch_02.json", {"epoch": 2, "uncommitted": True})
    write_jsonl(metrics / "epoch_02_predictions.jsonl", [{"prediction": "uncommitted"}])
    state = {"journal_length": 0, "journal_hash": canonical_hash([]), "epochs": [epoch]}
    reconcile_logs(metrics, state)
    assert (metrics / "epoch_01.json").exists()
    assert not (metrics / "epoch_02.json").exists()
    assert len(list((tmp_path / ".recovery").iterdir())) == 2
    predictions.write_text("tampered")
    with pytest.raises(ValueError, match="predictions"):
        reconcile_logs(metrics, state)


def test_tensorboard_resume_replays_only_committed_boundary():
    class Writer:
        def __init__(self):
            self.events = []

        def add_scalar(self, tag, value, step):
            self.events.append((tag, value, step))

    writer = Writer()
    record = {
        "global_step": 50,
        "loss": 0.4,
        "learning_rate": 0.01,
        "gradient_norm": 1.0,
        "target_tokens_seen": 500,
        "target_tokens": 10,
        "elapsed_seconds": 2.0,
        "memory": {"rss_bytes": 100},
    }
    state = {
        "global_step": 50,
        "epochs": [{"global_step": 50, "metrics": {"joint_exact_match": 0.7}}],
    }
    replay_tensorboard_boundary(writer, [{**record, "global_step": 49}, record], state)
    assert writer.events
    assert {step for _, _, step in writer.events} == {50}
    assert ("validation/joint_exact_match", 0.7, 50) in writer.events


def test_completed_training_cache_does_not_reopen_frozen_training(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import src.models as models
    import src.train as train

    root = tmp_path / "trials" / "lora"
    root.mkdir(parents=True)
    summary = {"completed": True, "optimizer_updates": 3600}
    atomic_json(root / "training_summary.json", summary)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A committed cache must not attempt training or change a complete stage"
        )

    run = SimpleNamespace(
        path=tmp_path,
        writer_lock=nullcontext,
        verify_identity=lambda: None,
        verify_artifacts=lambda **kwargs: None,
        require_stage=lambda name: None,
        assert_training_open=forbidden,
        stage=forbidden,
    )
    monkeypatch.setattr(models, "_registered_files", lambda *args: True)
    monkeypatch.setattr(train, "resolve_checkpoint", lambda *args: root / "checkpoints" / "final")
    assert train.train_adapter(run, "lora") == summary
