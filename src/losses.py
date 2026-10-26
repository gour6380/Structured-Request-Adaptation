"""Causal answer losses with optional projection of supervised positions only.

Qwen2's ``logits_to_keep`` selects hidden states before its large vocabulary
projection. The transformer still sees the complete prompt. Selection changes
neither the causal shift nor which answer tokens contribute to the objective.
"""

from __future__ import annotations

import math


def collate_answer_rows(rows: list[dict], device: str, pad_token_id: int):
    """Right-pad prepared rows; prompt and padding labels stay ignored.

    Padding is identified by sequence length, never by token identity, because
    the tokenizer can use the same token for EOS and PAD. The input rows are not
    modified. This collator is shared by training and answer-token NLL evaluation.
    """
    import torch

    if not rows:
        raise ValueError("Cannot collate an empty answer batch")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("The tokenizer must provide a nonnegative integer pad_token_id")
    if any(
        len(row["input_ids"]) != len(row["labels"]) or len(row["input_ids"]) < 2 for row in rows
    ):
        raise ValueError("Answer rows require aligned input/label sequences of at least two tokens")
    width = max(len(row["input_ids"]) for row in rows)
    inputs, labels, masks = [], [], []
    for row in rows:
        size = len(row["input_ids"])
        padding = width - size
        inputs.append(list(row["input_ids"]) + [pad_token_id] * padding)
        labels.append(list(row["labels"]) + [-100] * padding)
        masks.append([1] * size + [0] * padding)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
    }, torch.tensor(labels, dtype=torch.long, device=device)


def _validate_aligned(logits, targets) -> None:
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected logits [batch, sequence, vocab] and matching targets")


def summed_token_loss(logits, targets):
    """FP32 summed cross entropy for already aligned source logits and targets."""
    import torch.nn.functional as functional

    _validate_aligned(logits, targets)
    mask = targets != -100
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("A microbatch has no supervised answer tokens")
    loss = functional.cross_entropy(logits[mask].float(), targets[mask], reduction="sum")
    return loss, count


def token_nll(logits, targets) -> tuple[float, int]:
    """Measure aligned NLL in bounded FP32 chunks; call under inference_mode."""
    import torch.nn.functional as functional

    _validate_aligned(logits, targets)
    count = int((targets != -100).sum().item())
    if not count:
        raise ValueError("Example has no supervised answer tokens after causal shift")
    total = 0.0
    for start in range(0, targets.shape[1], 64):
        end = min(start + 64, targets.shape[1])
        target_chunk = targets[:, start:end].reshape(-1)
        supervised = target_chunk != -100
        if not bool(supervised.any().item()):
            continue
        scores = logits[:, start:end, :].reshape(-1, logits.shape[-1])[supervised]
        value = functional.cross_entropy(scores.float(), target_chunk[supervised], reduction="sum")
        total += float(value.item())
    if not math.isfinite(total):
        raise RuntimeError("Non-finite answer-token NLL")
    return total, count


def _forward_answer_logits(model, batch, labels, *, answer_only_logits: bool):
    """Return logits and targets after exactly one shift, with batch-safe selection.

    For a padded batch, select the union of source positions needed by any row.
    Each row retains its own -100 mask, so a position useful to one example does
    not accidentally supervise another example's prompt or padding.
    """
    import torch

    if type(answer_only_logits) is not bool:
        raise ValueError("answer_only_logits must be a boolean")
    if labels.ndim != 2 or batch["input_ids"].shape != labels.shape:
        raise ValueError("Expected matching [batch, sequence] input_ids and labels")
    targets = labels[:, 1:]
    if answer_only_logits:
        positions = torch.nonzero((targets != -100).any(dim=0), as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("A microbatch has no supervised answer tokens")
        output = model(**batch, use_cache=False, logits_to_keep=positions)
        return output.logits, targets.index_select(1, positions)
    output = model(**batch, use_cache=False)
    return output.logits[:, :-1, :], targets


def forward_answer_loss(model, batch, labels, *, answer_only_logits: bool = False):
    """Return differentiable summed answer loss and its supervised-token count."""
    logits, targets = _forward_answer_logits(
        model, batch, labels, answer_only_logits=answer_only_logits
    )
    return summed_token_loss(logits, targets)


def forward_answer_nll(
    model, batch, labels, *, answer_only_logits: bool = False
) -> tuple[float, int]:
    """Return measured summed NLL and token count; call under inference_mode."""
    logits, targets = _forward_answer_logits(
        model, batch, labels, answer_only_logits=answer_only_logits
    )
    return token_nll(logits, targets)


def forward_answer_nll_per_example(
    model, batch, labels, *, answer_only_logits: bool = False
) -> list[tuple[float, int]]:
    """One batched forward with separate NLL/token counts for each original row."""
    logits, targets = _forward_answer_logits(
        model, batch, labels, answer_only_logits=answer_only_logits
    )
    return [
        token_nll(logits[index : index + 1], targets[index : index + 1])
        for index in range(labels.shape[0])
    ]
