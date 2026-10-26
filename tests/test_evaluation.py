"""Authored tests; not executed during project preparation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.evaluate import (
    _decode_continuation,
    _percentile,
    _require_frozen,
    _stored_outputs_complete,
    answer_token_nll,
    evaluate_model,
)


def test_answer_nll_uses_next_token_and_masks_prompt() -> None:
    import torch

    logits = torch.tensor([[[7.0, -2.0], [-1.0, 3.0], [2.0, -1.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, -100, 1, 0]])
    total, count = answer_token_nll(logits, labels)
    expected = -torch.log_softmax(logits[0, 1], -1)[1] - torch.log_softmax(logits[0, 2], -1)[0]
    assert count == 2
    assert total == pytest.approx(expected.item())


def test_answer_nll_chunks_do_not_include_final_logit() -> None:
    import torch

    logits = torch.zeros((1, 131, 3))
    logits[:, -1, :] = float("nan")  # The unused final prediction must not be scored.
    labels = torch.ones((1, 131), dtype=torch.long)
    labels[:, :3] = -100
    total, count = answer_token_nll(logits, labels)
    assert count == 128
    assert total == pytest.approx(128 * torch.log(torch.tensor(3.0)).item())


def test_answer_nll_rejects_unscorable_example() -> None:
    import torch

    with pytest.raises(ValueError, match="no supervised"):
        answer_token_nll(torch.zeros((1, 3, 2)), torch.full((1, 3), -100))


class TinyTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(
            {1: "{", 2: "}", 7: "<unexpected>", 9: "<eos>", 0: "<pad>"}[value] for value in ids
        )


def test_decode_keeps_unexpected_special_and_raw_terminal_tokens() -> None:
    result = _decode_continuation(TinyTokenizer(), [1, 7, 2, 9, 0])
    assert result["prediction_text"] == "{<unexpected>}"
    assert result["raw_continuation"] == "{<unexpected>}<eos><pad>"
    assert result["terminal_token_ids"] == [9, 0]


def test_internal_eos_is_not_repaired() -> None:
    assert _decode_continuation(TinyTokenizer(), [1, 9, 2])["prediction_text"] == "{<eos>}"


def test_p95_is_interpolated_and_empty_is_rejected() -> None:
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        _percentile([], 0.95)


def test_test_evaluation_cannot_load_a_model_before_freeze(tmp_path, monkeypatch) -> None:
    from src import evaluate

    calls = []
    run = SimpleNamespace(
        path=tmp_path,
        config=SimpleNamespace(preset="full"),
        verify_identity=lambda: None,
        require_stage=lambda name: calls.append(name),
    )
    monkeypatch.setattr(evaluate, "_loaded", lambda *_: pytest.fail("Model loaded before freeze"))
    with pytest.raises(RuntimeError, match="freeze_evaluation"):
        evaluate_model(run, "base")
    assert "tokenization" in calls


def test_frozen_protocol_cannot_accept_changed_artifact(tmp_path) -> None:
    from src.io import atomic_json, canonical_hash

    config = SimpleNamespace(to_dict=lambda: {"seed": 2027})
    atomic_json(
        tmp_path / "evaluations" / "frozen_protocol.json",
        {
            "protocol_hash": canonical_hash(config.to_dict()),
            "artifacts": {"missing.json": "a" * 64},
        },
    )
    atomic_json(tmp_path / "manifest.json", {"artifacts": {"evaluations/frozen_protocol.json": {}}})
    atomic_json(tmp_path / "stages" / "freeze_evaluation.json", {"status": "complete"})
    run = SimpleNamespace(
        path=tmp_path,
        config=config,
        verify_identity=lambda: None,
        require_stage=lambda _: None,
        verify_artifacts=lambda **_: None,
        writer_lock=lambda: nullcontext(),
    )
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        _require_frozen(run)


def test_existing_validation_is_loaded_without_inference(tmp_path, monkeypatch) -> None:
    from src import evaluate
    from src.io import atomic_json

    directory = tmp_path / "evaluations" / "base" / "validation"
    atomic_json(directory / "metrics.json", {"joint_exact_match": 0.5})
    (directory / "predictions.jsonl").write_text("", encoding="utf-8")
    atomic_json(
        tmp_path / "manifest.json",
        {
            "artifacts": {
                "evaluations/base/validation/metrics.json": {},
                "evaluations/base/validation/predictions.jsonl": {},
            }
        },
    )
    atomic_json(tmp_path / "stages" / "evaluate_base_validation.json", {"status": "complete"})
    run = SimpleNamespace(
        path=tmp_path,
        verify_identity=lambda: None,
        require_stage=lambda _: None,
        verify_artifacts=lambda **_: None,
        writer_lock=lambda: nullcontext(),
    )
    monkeypatch.setattr(
        evaluate, "_loaded", lambda *_: pytest.fail("Cached result triggered inference")
    )
    assert evaluate_model(run, "base", "validation")["metrics"]["joint_exact_match"] == 0.5


def test_committed_outputs_reconcile_interrupted_stage(tmp_path) -> None:
    from src.io import atomic_json

    events = []
    atomic_json(tmp_path / "manifest.json", {"artifacts": {"metrics.json": {}}})
    atomic_json(tmp_path / "stages" / "evaluation.json", {"status": "running"})
    run = SimpleNamespace(
        path=tmp_path,
        verify_artifacts=lambda **_: events.append("verified"),
        stage=lambda name: events.append(name) or nullcontext(),
    )
    assert _stored_outputs_complete(run, [tmp_path / "metrics.json"], "evaluation")
    assert events == ["verified", "evaluation"]


def test_unregistered_partial_output_is_not_a_cached_result(tmp_path) -> None:
    from src.io import atomic_json

    atomic_json(tmp_path / "manifest.json", {"artifacts": {}})
    atomic_json(tmp_path / "metrics.json", {"incomplete": True})
    run = SimpleNamespace(path=tmp_path)
    assert not _stored_outputs_complete(run, [tmp_path / "metrics.json"], "evaluation")


def test_development_test_gate_precedes_model_or_artifact_access():
    from types import SimpleNamespace

    from src.evaluate import evaluate_loaded, evaluate_model

    run = SimpleNamespace(config=SimpleNamespace(preset="development"))
    with pytest.raises(RuntimeError, match="disabled for development"):
        evaluate_model(run, "lora", "test")
    with pytest.raises(RuntimeError, match="disabled for development"):
        evaluate_loaded(run, None, None, "test")


@pytest.mark.parametrize("requests,warmups,repetitions", [(16, 2, 1), (7, 3, 2)])
def test_development_benchmark_executes_configured_budget_and_replays_without_model(
    tmp_path, monkeypatch, requests, warmups, repetitions
):
    from contextlib import contextmanager

    from src import data, evaluate, runtime
    from src.config import ExperimentConfig
    from src.io import atomic_json, read_json, read_jsonl, sha256_file

    class Run:
        path = tmp_path
        config = ExperimentConfig.development(
            model_revision="a" * 40,
            evaluation_batch_size=8,  # Evaluation batching must not alter latency benchmarks.
            benchmark_requests=requests,
            benchmark_warmup=warmups,
            benchmark_repetitions=repetitions,
        )
        environment = {"device": "cpu"}

        def writer_lock(self):
            return nullcontext()

        @contextmanager
        def stage(self, name):
            yield
            atomic_json(self.path / "stages" / f"{name}.json", {"status": "complete"})

        def register_artifacts(self, paths):
            manifest = read_json(self.path / "manifest.json")
            for path in paths:
                manifest["artifacts"][str(path.relative_to(self.path))] = {
                    "sha256": sha256_file(path)
                }
            atomic_json(self.path / "manifest.json", manifest)

        def verify_artifacts(self, prefixes):
            manifest = read_json(self.path / "manifest.json")
            for name in prefixes:
                assert sha256_file(self.path / name) == manifest["artifacts"][name]["sha256"]

    class Memory:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def summary(self):
            return {"measurement": "synthetic-no-device"}

    run = Run()
    atomic_json(tmp_path / "manifest.json", {"artifacts": {}})
    rows = [
        {
            "id": str(i),
            "utterance": f"request {i}",
            "target": {"intent": "alarm_set", "slots": []},
            "novel_input": True,
            "prompt_ids": [i],
        }
        for i in range(32)
    ]
    loaded = []
    generated = []

    @contextmanager
    def load(_run, system, phases):
        loaded.append(system)
        phases["loading"] = {"measurement": "mocked-model"}
        yield object(), object()

    def generate(_run, _model, _tokenizer, prompt):
        generated.append(prompt[0])
        return {
            "prediction_text": '{"intent":"alarm_set","slots":[]}',
            "generated_tokens": 10,
            "generation_seconds": 0.0,
        }

    monkeypatch.setattr(evaluate, "_require_frozen", lambda _: {})
    monkeypatch.setattr(evaluate, "_loaded", load)
    monkeypatch.setattr(evaluate, "_generate", generate)
    monkeypatch.setattr(data, "load_examples", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(data, "load_ontology", lambda _: {"intents": ["alarm_set"], "slots": []})
    monkeypatch.setattr(
        data,
        "chat_token_ids",
        lambda _tokenizer, messages, **_kwargs: [int(messages[-1]["content"].split()[-1])],
    )
    monkeypatch.setattr(runtime, "MemorySampler", Memory)
    monkeypatch.setattr(runtime, "seed_everything", lambda _: None)
    monkeypatch.setattr(runtime, "synchronize", lambda _: None)
    summary = evaluate.benchmark_model(run, "lora")
    assert loaded == ["lora"]
    assert len(generated) == warmups + requests * repetitions
    assert summary["requests_per_repetition"] == requests
    assert summary["warmup_requests"] == warmups
    assert summary["repetitions"] == repetitions
    assert summary["measured_requests"] == requests * repetitions
    assert summary["generated_tokens"] == 10 * requests * repetitions
    assert summary["development_only"] and summary["preset"] == "development"
    assert len(summary["metrics_by_repetition"]) == repetitions
    assert all(m["joint_exact_match"] == 1.0 for m in summary["metrics_by_repetition"])
    records = read_jsonl(tmp_path / "benchmarks" / "lora" / "requests.jsonl")
    selection = read_json(tmp_path / "benchmarks" / "requests.json")
    selected_ids = selection["example_ids"]
    assert len(selected_ids) == requests
    assert generated[:warmups] == [int(value) for value in selected_ids[:warmups]]
    for repetition in range(1, repetitions + 1):
        assert [row["id"] for row in records if row["repetition"] == repetition] == selected_ids
    assert evaluate.benchmark_model(run, "lora") == summary
    assert loaded == ["lora"]
    assert len(generated) == warmups + requests * repetitions


@pytest.mark.parametrize("requests,warmups,repetitions", [(99, 5, 3), (100, 2, 3), (100, 5, 1)])
def test_full_benchmark_budget_remains_fixed(monkeypatch, requests, warmups, repetitions):
    from src import data, evaluate
    from src.config import ExperimentConfig

    run = SimpleNamespace(
        config=ExperimentConfig(
            benchmark_requests=requests,
            benchmark_warmup=warmups,
            benchmark_repetitions=repetitions,
        )
    )
    monkeypatch.setattr(evaluate, "_require_frozen", lambda _: {})
    monkeypatch.setattr(data, "load_examples", lambda *_: pytest.fail("Invalid budget read data"))
    with pytest.raises(ValueError, match="fixed at 100"):
        evaluate.benchmark_model(run, "base")
