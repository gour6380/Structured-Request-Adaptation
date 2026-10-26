"""CPU-only parity and accounting for batched validation/test generation."""

from functools import wraps
from types import SimpleNamespace

import pytest
import torch

from src.evaluate import _generate, _generate_batch, evaluate_loaded


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    @wraps(torch.backends.mps.is_available)
    def no_mps():
        return False

    monkeypatch.setattr(torch.backends.mps, "is_available", no_mps)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 31

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return " ".join(map(str, ids))


def run_config(batch_size=2, **overrides):
    values = {
        "preset": "full",
        "expected_split_counts": (0, 5, 5),
        "evaluation_batch_size": batch_size,
        "max_new_tokens": 6,
        "max_sequence_length": 32,
        "seed": 2027,
        "answer_only_logits": True,
        "telemetry_interval_seconds": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**values), environment={"device": "cpu"})


class ScriptedGenerator:
    def __init__(self, continuations):
        self.continuations = continuations
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        assert kwargs["use_cache"] is True
        return torch.cat(
            (kwargs["input_ids"], torch.tensor(self.continuations, dtype=torch.long)), dim=1
        )


def test_generation_left_pads_and_counts_only_emitted_tokens(monkeypatch):
    from src import evaluate

    ticks = iter([10.0, 14.0])
    monkeypatch.setattr(evaluate.time, "perf_counter", lambda: next(ticks))
    model = ScriptedGenerator([[1, 1, 2, 31, 0, 0], [4, 4, 5, 4, 4, 5]])
    result = _generate_batch(run_config(), model, Tokenizer(), [[8, 9], [10, 11, 12, 13]])
    call = model.calls[0]
    assert call["input_ids"].tolist() == [[0, 0, 8, 9], [10, 11, 12, 13]]
    assert call["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert result["generation_seconds"] == 4.0
    assert result["padded_prompt_tokens"] == 4 and result["batch_size"] == 2
    first, second = result["predictions"]
    assert first["generated_token_ids"] == [1, 1, 2, 31]
    assert first["prediction_text"] == "1 1 2"
    assert first["raw_continuation"] == "1 1 2 31"
    assert first["generated_tokens"] == 4 and first["post_eos_padding_tokens"] == 2
    assert first["prompt_tokens"] == 2 and not first["hit_generation_limit"]
    assert second["generated_token_ids"] == [4, 4, 5, 4, 4, 5]
    assert second["generated_tokens"] == 6 and second["hit_generation_limit"]
    assert all("generation_seconds" not in prediction for prediction in result["predictions"])


def test_shared_eos_and_pad_id_does_not_inflate_counts():
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = ScriptedGenerator([[1, 31, 31, 31], [2, 2, 2, 31]])
    result = _generate_batch(run_config(max_new_tokens=4), model, tokenizer, [[8], [9, 10]])[
        "predictions"
    ]
    assert model.calls[0]["attention_mask"].tolist() == [[0, 1], [1, 1]]
    assert [row["generated_tokens"] for row in result] == [2, 4]
    assert [row["post_eos_padding_tokens"] for row in result] == [2, 0]
    assert all(not row["hit_generation_limit"] for row in result)


def test_generation_does_not_hide_nonpadding_after_eos():
    with pytest.raises(RuntimeError, match="non-padding tokens after EOS"):
        _generate_batch(run_config(), ScriptedGenerator([[1, 31, 2]]), Tokenizer(), [[8]])


@pytest.mark.parametrize("prompts", [[], [[]], [[1] * 27], [[8], [1] * 27]])
def test_invalid_batch_is_rejected_before_generation(prompts):
    model = ScriptedGenerator([])
    with pytest.raises(ValueError, match="no silent truncation"):
        _generate_batch(run_config(), model, Tokenizer(), prompts)
    assert not model.calls


def test_single_request_path_ignores_evaluation_batch_size():
    model = ScriptedGenerator([[1, 31]])
    result = _generate(run_config(batch_size=8), model, Tokenizer(), [8, 9])
    assert model.calls[0]["input_ids"].shape[0] == 1
    assert result["generated_tokens"] == 2 and result["generation_seconds"] >= 0


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=True,
            attention_dropout=0.0,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=31,
        )
    ).cpu()


def example_rows():
    rows = []
    for index, prompt in enumerate(([1, 2], [3, 4, 5, 6], [7], [8, 9, 10], [11, 12])):
        answer = [20 + index] * (index % 3 + 1) + [31]
        rows.append(
            {
                "id": f"request-{index}",
                "utterance": f"request {index}",
                "target": {"intent": "alarm_set", "slots": []},
                "novel_input": index % 2 == 0,
                "prompt_ids": list(prompt),
                "input_ids": list(prompt) + answer,
                "labels": [-100] * len(prompt) + answer,
            }
        )
    return rows


class Memory:
    def __init__(self, *_args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def summary(self):
        return {"measurement": "CPU test"}


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("answer_only_logits", [False, True])
def test_tiny_qwen_evaluation_batches_match_individual_requests(
    monkeypatch, batch_size, answer_only_logits
):
    from src import data, runtime

    rows = example_rows()
    monkeypatch.setattr(data, "load_examples", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(data, "load_ontology", lambda _: {"intents": ["alarm_set"], "slots": []})
    monkeypatch.setattr(runtime, "MemorySampler", Memory)
    model = tiny_model().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = torch.get_rng_state().clone()
    singleton = evaluate_loaded(
        run_config(1, answer_only_logits=answer_only_logits), model, Tokenizer(), "validation"
    )
    assert model.training and model.is_gradient_checkpointing and not model.config.use_cache
    assert torch.equal(torch.get_rng_state(), rng)
    batched = evaluate_loaded(
        run_config(batch_size, answer_only_logits=answer_only_logits),
        model,
        Tokenizer(),
        "validation",
    )
    assert model.training and model.is_gradient_checkpointing and not model.config.use_cache
    assert torch.equal(torch.get_rng_state(), rng)
    assert [row["id"] for row in batched["predictions"]] == [row["id"] for row in rows]
    for single, batch in zip(singleton["predictions"], batched["predictions"], strict=True):
        for key in ("generated_token_ids", "generated_tokens", "prediction_text", "answer_tokens"):
            assert batch[key] == single[key]
        assert batch["answer_nll_sum"] == pytest.approx(single["answer_nll_sum"], rel=2e-6)
        assert "generation_seconds" not in batch
    for result in (singleton, batched):
        metrics = result["metrics"]
        assert metrics["answer_tokens"] == sum(
            row["answer_tokens"] for row in result["predictions"]
        )
        assert metrics["example_count"] == 5
        assert metrics["schema_valid_rate"] == 0.0  # Malformed outputs remain in the denominator.
        novel = [row for row in result["predictions"] if row["novel_input"]]
        subset = metrics["novel_input_subset"]
        assert subset["example_count"] == 3
        assert subset["answer_tokens"] == sum(row["answer_tokens"] for row in novel)
        assert subset["answer_nll_sum"] == pytest.approx(
            sum(row["answer_nll_sum"] for row in novel)
        )
    assert batched["metrics"]["answer_token_nll"] == pytest.approx(
        singleton["metrics"]["answer_token_nll"], rel=2e-6
    )
    batches = batched["metrics"]["generation_batches"]
    assert [batch["batch_size"] for batch in batches] == ([2, 2, 1] if batch_size == 2 else [5])
    assert [item for batch in batches for item in batch["example_ids"]] == [
        row["id"] for row in rows
    ]
    assert batched["metrics"]["generation_seconds"] == sum(b["generation_seconds"] for b in batches)


def test_evaluation_restores_model_and_rng_when_batch_generation_fails(monkeypatch):
    from src import data, evaluate, runtime

    monkeypatch.setattr(data, "load_examples", lambda *_args, **_kwargs: example_rows())
    monkeypatch.setattr(data, "load_ontology", lambda _: {"intents": ["alarm_set"], "slots": []})
    monkeypatch.setattr(runtime, "MemorySampler", Memory)

    def fail(*_args):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(evaluate, "_generate_batch", fail)
    model = tiny_model().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        evaluate_loaded(run_config(), model, Tokenizer(), "validation")
    assert model.training and model.is_gradient_checkpointing and not model.config.use_cache
    assert torch.equal(torch.get_rng_state(), rng)
