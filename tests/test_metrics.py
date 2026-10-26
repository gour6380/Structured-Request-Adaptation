"""Authored contracts for strict generation scoring; not executed at handoff."""

import json

import pytest

from src.metrics import evaluate_predictions, score_prediction, validate_prediction

INTENTS = ["alarm_set", "general_greet"]
SLOTS = ["date", "time"]


def target(slots=None, intent="alarm_set"):
    return {"intent": intent, "slots": slots or []}


def score(text, gold=None):
    return score_prediction(text, gold or target(), INTENTS, SLOTS)


def test_json_and_slot_order_are_irrelevant_but_multiplicity_is_not():
    gold = target([{"type": "time", "value": "seven"}, {"type": "date", "value": "today"}])
    reversed_slots = '{"slots":[{"value":"today","type":"date"},{"value":"seven","type":"time"}],"intent":"alarm_set"}'
    assert score(reversed_slots, gold)["joint_exact_match"]
    repeated = target([{"type": "time", "value": "seven"}] * 2)
    fewer = score(json.dumps(target([{"type": "time", "value": "seven"}])), repeated)
    assert (fewer["slot_tp"], fewer["slot_fp"], fewer["slot_fn"]) == (1, 0, 1)
    excess = score(json.dumps(repeated), target([{"type": "time", "value": "seven"}]))
    assert (excess["slot_tp"], excess["slot_fp"], excess["slot_fn"]) == (1, 1, 0)


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"intent":"alarm_set","slots":[]}\n```',
        'Here is your JSON: {"intent":"alarm_set","slots":[]}',
        '{"intent":"alarm_set","slots":[]} trailing',
        '{"intent":"alarm_set","slots":[]} {"intent":"general_greet","slots":[]}',
        '{"intent":"alarm_set","intent":"general_greet","slots":[]}',
        '{"intent":"alarm_set","slots":NaN}',
        "",
        '{"intent":',
    ],
)
def test_output_is_never_extracted_or_repaired(text):
    result = score(text)
    assert not result["json_valid"]
    assert not result["joint_exact_match"]


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        {"intent": "alarm_set"},
        {"intent": "alarm_set", "slots": [], "explanation": "hello"},
        {"intent": "alarm_set", "slots": [{"type": "time", "value": 7}]},
        {"intent": "alarm_set", "slots": [{"type": "time", "value": ""}]},
        {"intent": "alarm_set", "slots": [{"type": "time", "value": "seven", "start": 0}]},
    ],
)
def test_valid_json_can_have_invalid_structure(value):
    result = score(json.dumps(value))
    assert result["json_valid"]
    assert not result["structure_valid"]
    assert not result["schema_valid"]
    assert not result["intent_correct"]


def test_unknown_slot_is_false_positive_and_missed_reference_is_false_negative():
    gold = target([{"type": "time", "value": "seven"}])
    result = score('{"intent":"alarm_set","slots":[{"type":"clock","value":"seven"}]}', gold)
    assert result["structure_valid"] and not result["schema_valid"]
    assert result["intent_correct"]
    assert (result["slot_tp"], result["slot_fp"], result["slot_fn"]) == (0, 1, 1)
    assert "unknown_slot_type" in result["failure_reasons"]


def test_values_are_not_normalized_and_decoded_json_escapes_are_equivalent():
    gold = target([{"type": "date", "value": "Today"}])
    assert not score(json.dumps(target([{"type": "date", "value": "today"}])), gold)[
        "joint_exact_match"
    ]
    assert not score(json.dumps(target([{"type": "date", "value": " Today"}])), gold)[
        "joint_exact_match"
    ]
    assert score('{"intent":"alarm_set","slots":[{"type":"date","value":"\\u0054oday"}]}', gold)[
        "joint_exact_match"
    ]


def test_invalid_no_slot_output_never_passes_primary_metric():
    records = [{"id": "1", "target": target(), "prediction": "invalid", "novel_input": True}]
    result = evaluate_predictions(records, INTENTS, SLOTS)
    assert result["joint_exact_match"] == 0
    assert result["intent_accuracy"] == 0
    assert result["slot_f1"] is None
    assert result["novel_input_subset"]["example_count"] == 1


def test_fixed_label_macro_f1_and_novel_slice():
    records = [
        {
            "id": "1",
            "target": target(),
            "prediction_text": json.dumps(target()),
            "novel_input": True,
        },
        {
            "id": "2",
            "target": target(intent="general_greet"),
            "prediction": json.dumps(target(intent="unknown")),
            "novel_input": False,
        },
    ]
    result = evaluate_predictions(records, INTENTS, SLOTS)
    assert result["joint_exact_match"] == 0.5
    assert result["intent_macro_f1"] == 0.5
    assert result["novel_input_subset"]["joint_exact_match"] == 1
    assert {"target": "general_greet", "prediction": "__invalid__", "count": 1} in result[
        "confusion_matrix"
    ]


def test_empty_slice_is_not_presented_as_perfect():
    result = evaluate_predictions([], INTENTS, SLOTS)
    assert result["joint_exact_match"] is None
    assert result["slot_f1"] is None
    assert result["undefined_reason"]


def test_duplicate_ids_and_invalid_reference_are_rejected():
    row = {"id": "1", "target": target(), "prediction": json.dumps(target())}
    with pytest.raises(ValueError, match="unique"):
        evaluate_predictions([row, row], INTENTS, SLOTS)
    with pytest.raises(ValueError, match="Ground truth"):
        score_prediction("{}", target(intent="unknown"), INTENTS, SLOTS)


def test_validation_api_requires_no_reference():
    result = validate_prediction('{"intent":"alarm_set","slots":[]}', INTENTS, SLOTS)
    assert result["schema_valid"]
    assert result["failure_reasons"] == []


def test_failure_taxonomy_pairs_remaining_slots_by_type_after_exact_matching():
    gold = target(
        [
            {"type": "time", "value": "seven"},
            {"type": "time", "value": "seven"},
            {"type": "time", "value": "eight"},
            {"type": "date", "value": "today"},
        ]
    )
    predicted = target(
        [
            {"type": "time", "value": "seven"},
            {"type": "time", "value": "nine"},
            {"type": "date", "value": "today"},
            {"type": "unknown", "value": "unexpected"},
        ]
    )
    result = score(json.dumps(predicted), gold)
    assert (result["slot_tp"], result["slot_fp"], result["slot_fn"]) == (2, 2, 2)
    assert (result["wrong_values"], result["missing_slots"], result["extra_slots"]) == (1, 1, 1)
    assert {"wrong_values", "missing_slots", "extra_slots", "unknown_slot_type"} <= set(
        result["failure_reasons"]
    )


def test_failure_aggregate_distinguishes_occurrences_from_affected_records():
    gold = target([{"type": "time", "value": "seven"}] * 2)
    predictions = [
        target([{"type": "time", "value": "eight"}] * 2),
        target(),
        target([{"type": "time", "value": "seven"}] * 3),
    ]
    records = [
        {"id": str(index), "target": gold, "prediction": json.dumps(prediction)}
        for index, prediction in enumerate(predictions)
    ]
    result = evaluate_predictions(records, INTENTS, SLOTS)
    assert result["counts"]["wrong_values"] == 2
    assert result["counts"]["missing_slots"] == 2
    assert result["counts"]["extra_slots"] == 1
    assert result["counts"]["slot_fp"] == 3
    assert result["counts"]["slot_fn"] == 4
    assert result["failure_counts"]["wrong_values"] == 1
    assert result["failure_counts"]["missing_slots"] == 1
    assert result["failure_counts"]["extra_slots"] == 1
