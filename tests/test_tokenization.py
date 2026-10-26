"""Chat-boundary contracts use a tiny fake tokenizer, never model downloads."""

import json

import pytest

from src.data import build_system_prompt, prompt_messages, tokenize_example


class CharacterTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        if add_generation_prompt:
            assert messages[-1]["role"] == "user"
            return [2, 3, 4]
        assert messages[-1]["role"] == "assistant"
        return [2, 3, 4] + [ord(char) + 100 for char in messages[-1]["content"]] + [1, 110]

    def decode(self, ids, **kwargs):
        assert not kwargs.get("skip_special_tokens")
        return "".join(chr(identifier - 100) for identifier in ids)


def example():
    target = {"intent": "alarm_set", "slots": []}
    return {
        "id": "1",
        "utterance": "set alarm",
        "target": target,
        "target_text": json.dumps(target, separators=(",", ":")),
        "novel_input": False,
    }


def test_only_answer_and_eos_are_supervised_and_prompt_is_exact_prefix():
    row = tokenize_example(example(), CharacterTokenizer(), "system", 2048, 256)
    assert row["input_ids"][:3] == row["prompt_ids"]
    assert row["labels"][:3] == [-100, -100, -100]
    assert row["labels"][-2:] == [1, -100]
    assert len(row["input_ids"]) == len(row["labels"])
    assert row["target_tokens"] == len(row["target_text"]) + 1
    assert row["target_tokens"] == sum(label != -100 for label in row["labels"][1:])


@pytest.mark.parametrize("sequence_length,new_tokens", [(30, 20), (260, 258), (2048, 5)])
def test_overlong_sequences_are_rejected_without_truncation(sequence_length, new_tokens):
    with pytest.raises(ValueError, match="Token budget"):
        tokenize_example(example(), CharacterTokenizer(), "system", sequence_length, new_tokens)


def test_prompt_generation_and_training_use_same_frozen_ontology():
    ontology = {"intents": ["alarm_set", "general_greet"], "slots": ["time"]}
    messages = prompt_messages("hello", ontology["intents"], ontology["slots"])
    assert messages[0]["content"] == build_system_prompt(ontology)
    assert messages[1] == {"role": "user", "content": "hello"}
    assert "alarm_set, general_greet" in messages[0]["content"]
    assert "Allowed slot types: time" in messages[0]["content"]


def test_prefix_mismatch_cannot_train_on_wrong_mask():
    class BadPrefix(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"]:
                ids[0] = 99
            return ids

    with pytest.raises(ValueError, match="prefix mismatch"):
        tokenize_example(example(), BadPrefix(), "system", 2048, 256)


def test_non_whitespace_template_tail_is_rejected():
    class BadTail(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"]:
                ids.append(ord("x") + 100)
            return ids

    with pytest.raises(ValueError, match="suffix"):
        tokenize_example(example(), BadTail(), "system", 2048, 256)


def test_transformers_batch_encoding_return_preserves_token_ids():
    from transformers import BatchEncoding

    class ModernTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})

    expected = tokenize_example(example(), CharacterTokenizer(), "system", 2048, 256)
    actual = tokenize_example(example(), ModernTokenizer(), "system", 2048, 256)
    assert actual == expected
