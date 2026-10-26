"""Strict JSON parsing and occurrence-aware MASSIVE generation measurements."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def parse_json(text: str) -> Any:
    """Parse the complete output without extracting, repairing or retrying it."""
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _structure_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"intent", "slots"}
        and isinstance(value["intent"], str)
        and isinstance(value["slots"], list)
        and all(
            isinstance(slot, dict)
            and set(slot) == {"type", "value"}
            and isinstance(slot["type"], str)
            and isinstance(slot["value"], str)
            and bool(slot["value"])
            for slot in value["slots"]
        )
    )


def slot_counter(target: dict[str, Any]) -> Counter[tuple[str, str]]:
    """Keep exact string values and multiplicity; ignore ordering only."""
    return Counter((slot["type"], slot["value"]) for slot in target["slots"])


def validate_prediction(
    text: str, intents: list[str] | tuple[str, ...], slot_types: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Validate an unseen request's output without needing a reference target."""
    prediction: Any = None
    reasons: list[str] = []
    try:
        prediction = parse_json(text)
        json_valid = True
    except (ValueError, TypeError, RecursionError):
        json_valid = False
        reasons.append("invalid_json")
    structure_valid = json_valid and _structure_valid(prediction)
    if json_valid and not structure_valid:
        reasons.append("invalid_structure")
    if structure_valid and prediction["intent"] not in intents:
        reasons.append("unknown_intent")
    if structure_valid and any(slot["type"] not in slot_types for slot in prediction["slots"]):
        reasons.append("unknown_slot_type")
    return {
        "json_valid": bool(json_valid),
        "structure_valid": bool(structure_valid),
        "schema_valid": bool(structure_valid and not reasons),
        "parsed_prediction": prediction if isinstance(prediction, dict) else None,
        "failure_reasons": reasons,
    }


def score_prediction(
    text: str,
    target: dict[str, Any],
    intents: list[str] | tuple[str, ...],
    slot_types: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Score a raw continuation; malformed records receive no field predictions.

    Well-shaped records retain field-level diagnostics even if a label is unknown.
    Unknown slot labels are ordinary nonmatching predictions (false positives).
    This never changes the output or makes an invalid whole record correct.
    """
    intent_set, slot_set = set(intents), set(slot_types)
    if (
        not _structure_valid(target)
        or target["intent"] not in intent_set
        or any(slot["type"] not in slot_set for slot in target["slots"])
    ):
        raise ValueError("Ground truth is outside the frozen output schema")
    validation = validate_prediction(text, intents, slot_types)
    prediction = validation["parsed_prediction"]
    json_valid = validation["json_valid"]
    structure_valid = validation["structure_valid"]
    reasons = validation["failure_reasons"]
    predicted_intent = prediction["intent"] if structure_valid else None
    schema_valid = validation["schema_valid"]
    gold_slots = slot_counter(target)
    predicted_slots = slot_counter(prediction) if structure_valid else Counter()
    tp = sum((gold_slots & predicted_slots).values())
    fp, fn = sum(predicted_slots.values()) - tp, sum(gold_slots.values()) - tp
    remaining_gold = Counter()
    remaining_prediction = Counter()
    for (slot_type, _), occurrences in (gold_slots - predicted_slots).items():
        remaining_gold[slot_type] += occurrences
    for (slot_type, _), occurrences in (predicted_slots - gold_slots).items():
        remaining_prediction[slot_type] += occurrences
    # After exact matches are removed, each same-type residual pair is one wrong
    # value. Different types cannot be paired; unknown types remain extra slots.
    wrong_values = sum((remaining_gold & remaining_prediction).values())
    missing_slots, extra_slots = fn - wrong_values, fp - wrong_values
    intent_correct = structure_valid and predicted_intent == target["intent"]
    if structure_valid and not intent_correct:
        reasons.append("wrong_intent")
    for reason, occurrences in (
        ("missing_slots", missing_slots),
        ("extra_slots", extra_slots),
        ("wrong_values", wrong_values),
    ):
        if occurrences:
            reasons.append(reason)
    return {
        "json_valid": bool(json_valid),
        "structure_valid": bool(structure_valid),
        "schema_valid": bool(schema_valid),
        "joint_exact_match": bool(
            schema_valid and intent_correct and predicted_slots == gold_slots
        ),
        "intent_correct": bool(intent_correct),
        "predicted_intent": predicted_intent,
        "slot_tp": tp,
        "slot_fp": fp,
        "slot_fn": fn,
        "missing_slots": missing_slots,
        "extra_slots": extra_slots,
        "wrong_values": wrong_values,
        "failure_reasons": reasons,
        "parsed_prediction": prediction if isinstance(prediction, dict) else None,
    }


def joint_exact_match(
    text: str, target: dict[str, Any], intents: list[str], slot_types: list[str]
) -> bool:
    return score_prediction(text, target, intents, slot_types)["joint_exact_match"]


def _aggregate(
    records: list[dict[str, Any]], intents: list[str], slot_types: list[str]
) -> dict[str, Any]:
    count = len(records)
    scores = [
        score_prediction(
            record.get("prediction", record.get("prediction_text", "")),
            record["target"],
            intents,
            slot_types,
        )
        for record in records
    ]
    counts = {
        name: sum(int(score[name]) for score in scores)
        for name in (
            "json_valid",
            "structure_valid",
            "schema_valid",
            "joint_exact_match",
            "intent_correct",
            "slot_tp",
            "slot_fp",
            "slot_fn",
            "missing_slots",
            "extra_slots",
            "wrong_values",
        )
    }
    tp, fp, fn = counts["slot_tp"], counts["slot_fp"], counts["slot_fn"]
    confusion: Counter[tuple[str, str]] = Counter()
    class_f1: dict[str, float] = {}
    for record, score in zip(records, scores, strict=True):
        predicted = score["predicted_intent"]
        if predicted not in intents:
            predicted = "__invalid__"
        confusion[(record["target"]["intent"], predicted)] += 1
    for label in intents:
        class_tp = confusion[(label, label)]
        class_fp = sum(
            n for (gold, predicted), n in confusion.items() if predicted == label and gold != label
        )
        class_fn = sum(
            n for (gold, predicted), n in confusion.items() if gold == label and predicted != label
        )
        denominator = 2 * class_tp + class_fp + class_fn
        class_f1[label] = 2 * class_tp / denominator if denominator else 0.0
    reasons: Counter[str] = Counter(
        reason for score in scores for reason in score["failure_reasons"]
    )
    return {
        "example_count": count,
        "joint_exact_match": counts["joint_exact_match"] / count if count else None,
        "exact_match": counts["joint_exact_match"] / count if count else None,
        "json_valid_rate": counts["json_valid"] / count if count else None,
        "structure_valid_rate": counts["structure_valid"] / count if count else None,
        "schema_valid_rate": counts["schema_valid"] / count if count else None,
        "intent_accuracy": counts["intent_correct"] / count if count else None,
        "intent_macro_f1": sum(class_f1.values()) / len(intents) if count and intents else None,
        "slot_precision": tp / (tp + fp) if tp + fp else None,
        "slot_recall": tp / (tp + fn) if tp + fn else None,
        "slot_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "counts": counts,
        "failure_counts": dict(sorted(reasons.items())),
        "intent_per_class_f1": class_f1,
        "confusion_matrix": [
            {"target": gold, "prediction": predicted, "count": n}
            for (gold, predicted), n in sorted(confusion.items())
        ],
        "undefined_reason": "No examples in this slice" if not count else None,
        "slot_scoring": "Exact case-sensitive value/type multisets; malformed structures predict no slots",
    }


def evaluate_predictions(
    records: list[dict[str, Any]], intents: list[str], slot_types: list[str]
) -> dict[str, Any]:
    """Aggregate complete official records and their predeclared novel-input slice."""
    if len(set(intents)) != len(intents) or len(set(slot_types)) != len(slot_types):
        raise ValueError("Ontology entries must be unique")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction IDs must be unique")
    result = _aggregate(records, intents, slot_types)
    novel = [record for record in records if record.get("novel_input", False)]
    result["novel_input_subset"] = _aggregate(novel, intents, slot_types)
    return result
