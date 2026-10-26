"""Loss-preserving vocabulary projection on tiny local Qwen2 models."""

import copy
from dataclasses import replace

import pytest
import torch

from src.config import TARGET_MODULES, ExperimentConfig
from src.losses import (
    collate_answer_rows,
    forward_answer_loss,
    forward_answer_nll,
    forward_answer_nll_per_example,
)


def _model(adapter: bool, checkpointing: bool):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(27)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=True,
            attention_dropout=0.0,
            use_cache=False,
        )
    )
    if adapter:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=list(TARGET_MODULES),
                # Nonzero A and B test gradients through both adapter matrices.
                init_lora_weights=False,
            ),
        )
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize(
    "input_rows,label_rows,attention_rows",
    [
        ([[1, 2, 3, 3, 4, 0]], [[-100, -100, 3, 3, 4, -100]], [[1, 1, 1, 1, 1, 0]]),
        (
            [[1, 2, 3, 4, 5, 0, 0], [1, 6, 7, 8, 9, 10, 11]],
            [[-100, -100, 3, 4, 5, -100, -100], [-100, -100, -100, -100, 9, 10, 11]],
            [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        ),
        # A non-contiguous answer mask also catches accidental second shifting.
        ([[1, 2, 3, 2, 5, 2, 7]], [[-100, 2, -100, 2, -100, 2, -100]], [[1] * 7]),
    ],
)
def test_selected_qwen_logits_preserve_loss_and_every_trainable_gradient(
    adapter, checkpointing, input_rows, label_rows, attention_rows
):
    dense = _model(adapter, checkpointing)
    selected = copy.deepcopy(dense)
    batch = {"input_ids": torch.tensor(input_rows), "attention_mask": torch.tensor(attention_rows)}
    labels = torch.tensor(label_rows)
    projected_lengths = []
    hook = selected.get_output_embeddings().register_forward_pre_hook(
        lambda _module, args: projected_lengths.append(args[0].shape[1])
    )
    dense_loss, dense_count = forward_answer_loss(dense, batch, labels, answer_only_logits=False)
    selected_loss, selected_count = forward_answer_loss(
        selected, batch, labels, answer_only_logits=True
    )
    dense_loss.backward()
    selected_loss.backward()
    hook.remove()
    expected_positions = int((labels[:, 1:] != -100).any(dim=0).sum())
    assert projected_lengths == [expected_positions]
    assert expected_positions < labels.shape[1]
    assert selected_count == dense_count == int((labels[:, 1:] != -100).sum())
    torch.testing.assert_close(selected_loss, dense_loss, rtol=1e-6, atol=1e-6)
    dense_parameters = dict(dense.named_parameters())
    checked = 0
    for name, parameter in selected.named_parameters():
        reference = dense_parameters[name]
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert reference.grad is not None, name
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=3e-5, atol=3e-6)
            checked += 1
        else:
            assert parameter.grad is None and reference.grad is None, name
    assert checked > 0


def test_selected_nll_matches_training_loss_and_dense_nll():
    model = _model(adapter=True, checkpointing=False).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
    }
    labels = torch.tensor([[-100, -100, 3, 4, -100], [-100, -100, -100, 8, 9]])
    with torch.inference_mode():
        dense_nll, dense_count = forward_answer_nll(model, batch, labels)
        selected_nll, selected_count = forward_answer_nll(
            model, batch, labels, answer_only_logits=True
        )
        loss, loss_count = forward_answer_loss(model, batch, labels, answer_only_logits=True)
    assert selected_count == dense_count == loss_count == 4
    assert selected_nll == pytest.approx(dense_nll, rel=1e-6)
    assert selected_nll == pytest.approx(loss.item(), rel=1e-6)


def test_empty_supervision_rejected_before_selected_forward():
    def model(**kwargs):
        raise AssertionError("Empty selected targets must not launch a model forward")

    with pytest.raises(ValueError, match="no supervised"):
        forward_answer_loss(
            model,
            {"input_ids": torch.ones(1, 3, dtype=torch.long)},
            torch.full((1, 3), -100),
            answer_only_logits=True,
        )


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_projection_policy_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match="answer_only_logits"):
        replace(ExperimentConfig(), answer_only_logits=value).validate(require_revision=False)


def test_selected_projection_flag_survives_config_round_trip():
    config = ExperimentConfig(model_revision="a" * 40, answer_only_logits=True)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_answer_collation_uses_lengths_when_pad_and_eos_are_identical():
    rows = [
        {"input_ids": [1, 2, 0], "labels": [-100, 2, 0]},
        {"input_ids": [1, 4, 5, 6, 0], "labels": [-100, -100, 5, 6, 0]},
    ]
    original = copy.deepcopy(rows)
    batch, labels = collate_answer_rows(rows, "cpu", pad_token_id=0)
    assert rows == original
    assert batch["input_ids"].tolist() == [[1, 2, 0, 0, 0], [1, 4, 5, 6, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert labels.tolist() == [[-100, 2, 0, -100, -100], [-100, -100, 5, 6, 0]]
    assert labels.dtype == torch.long


@pytest.mark.parametrize(
    "rows,pad_token_id",
    [
        ([], 0),
        ([{"input_ids": [1], "labels": [-100]}], 0),
        ([{"input_ids": [1, 2], "labels": [-100]}], 0),
        ([{"input_ids": [1, 2], "labels": [-100, 2]}], None),
        ([{"input_ids": [1, 2], "labels": [-100, 2]}], True),
    ],
)
def test_answer_collation_rejects_invalid_batches(rows, pad_token_id):
    with pytest.raises(ValueError):
        collate_answer_rows(rows, "cpu", pad_token_id)


def _variable_length_rows():
    return [
        {
            "input_ids": [1, 2, *range(3, 6 + index % 4)],
            "labels": [-100] * (2 + index % 2) + list(range(3 + index % 2, 6 + index % 4)),
        }
        for index in range(7)
    ]


@pytest.mark.parametrize("selected", [False, True])
def test_padded_microbatches_and_partial_batch_match_serial_qwen_gradients(selected):
    serial = _model(adapter=True, checkpointing=False)
    batched = copy.deepcopy(serial)
    rows = _variable_length_rows()
    total_tokens = sum(sum(value != -100 for value in row["labels"][1:]) for row in rows)
    for row in rows:
        batch, labels = collate_answer_rows([row], "cpu", 0)
        loss, _ = forward_answer_loss(serial, batch, labels, answer_only_logits=selected)
        (loss / total_tokens).backward()
    # Seven rows produce one full batch of four and a retained batch of three.
    for start in range(0, len(rows), 4):
        batch, labels = collate_answer_rows(rows[start : start + 4], "cpu", 0)
        loss, _ = forward_answer_loss(batched, batch, labels, answer_only_logits=selected)
        (loss / total_tokens).backward()
    reference = dict(serial.named_parameters())
    for name, parameter in batched.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("selected", [False, True])
def test_per_example_nll_matches_unpadded_reference_after_one_batched_forward(selected):
    model = _model(adapter=True, checkpointing=False).eval()
    rows = _variable_length_rows()
    forwards = []
    hook = model.register_forward_hook(lambda *_: forwards.append(True))
    batch, labels = collate_answer_rows(rows, "cpu", 0)
    with torch.inference_mode():
        measured = forward_answer_nll_per_example(model, batch, labels, answer_only_logits=selected)
        assert len(forwards) == 1
        references = []
        for row in rows:
            row_batch, row_labels = collate_answer_rows([row], "cpu", 0)
            references.append(
                forward_answer_nll(model, row_batch, row_labels, answer_only_logits=selected)
            )
    hook.remove()
    assert len(measured) == len(rows)
    for actual, expected in zip(measured, references, strict=True):
        assert actual[1] == expected[1]
        assert actual[0] == pytest.approx(expected[0], rel=1e-6)
