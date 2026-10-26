"""Authored dataset and archive contracts; no remote data is required."""

import tarfile
from contextlib import contextmanager

import pytest

from src.data import (
    _resume_completed_stage,
    _validate_member,
    build_prepared_rows,
    normalized_utterance,
    parse_annotated_utterance,
)
from src.io import atomic_json, atomic_text, read_json, sha256_file


def record(identifier, partition, utterance, annotated=None, intent="iot_on"):
    return {
        "id": identifier,
        "partition": partition,
        "locale": "en-US",
        "utt": utterance,
        "annot_utt": annotated if annotated is not None else utterance,
        "intent": intent,
    }


def examples():
    return [
        record("0", "train", "set lamp", "set [device : lamp]"),
        record("1", "train", "hello", intent="general_greet"),
        record("2", "dev", "SET lamp", "SET [device : lamp]"),
        record("3", "test", "set radio", "set [device : radio]"),
    ]


def test_repeated_slots_and_exact_values_are_preserved():
    utterance = 'ask Zoë and Zoë about "tomorrow"'
    annotated = 'ask [person : Zoë] and [person : Zoë] about [date : "tomorrow"]'
    assert parse_annotated_utterance(utterance, annotated) == [
        {"type": "person", "value": "Zoë"},
        {"type": "person", "value": "Zoë"},
        {"type": "date", "value": '"tomorrow"'},
    ]
    assert parse_annotated_utterance("hello", "hello") == []
    assert parse_annotated_utterance("set  seven ", "set [time :  seven ]") == [
        {"type": "time", "value": " seven "}
    ]


@pytest.mark.parametrize(
    "utterance,annotated",
    [
        ("set seven", "set [time: seven]"),
        ("set seven", "set [time : seven"),
        ("set seven", "set [time : [date : seven]]"),
        ("set Seven", "set [time : seven]"),
    ],
)
def test_annotation_errors_are_not_silently_repaired(utterance, annotated):
    with pytest.raises(ValueError):
        parse_annotated_utterance(utterance, annotated)


def test_official_partitions_and_all_rows_are_preserved():
    splits, ontology, audit = build_prepared_rows(examples(), (2, 1, 1))
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    assert ontology == {"intents": ["general_greet", "iot_on"], "slots": ["device"]}
    assert splits["validation"][0]["utterance"] == "SET lamp"
    assert not splits["validation"][0]["novel_input"]
    assert splits["test"][0]["novel_input"]
    assert audit["normalized"]["cross_split_group_count"] == 1
    assert audit["exact"]["cross_split_group_count"] == 0


def test_test_novelty_excludes_both_training_and_validation_utterances():
    rows = examples()
    rows[2] = record("2", "dev", "set radio", "set [device : radio]")
    splits, _, audit = build_prepared_rows(rows, (2, 1, 1))
    assert splits["validation"][0]["novel_input"]
    assert not splits["test"][0]["novel_input"]
    assert audit["exact"]["cross_split_group_count"] == 1


def test_annotation_conflicts_are_retained_and_flagged():
    rows = examples()
    rows[2] = record("2", "dev", "hello", intent="iot_on")
    splits, _, audit = build_prepared_rows(rows, (2, 1, 1))
    assert len(splits["validation"]) == 1
    assert audit["normalized"]["conflicting_group_count"] == 1


def test_invalid_counts_ids_and_unseen_labels_fail():
    with pytest.raises(ValueError, match="split counts"):
        build_prepared_rows(examples(), (3, 1, 1))
    rows = examples()
    rows[3]["id"] = "0"
    with pytest.raises(ValueError, match="Repeated"):
        build_prepared_rows(rows, (2, 1, 1))
    rows = examples()
    rows[3]["intent"] = "new_test_only_label"
    with pytest.raises(ValueError, match="Held-out label"):
        build_prepared_rows(rows, (2, 1, 1))


def test_audit_normalization_does_not_require_model_input_normalization():
    assert normalized_utterance("  Ａlarm\tSET ") == "alarm set"


@pytest.mark.parametrize("name", ["../outside", "/absolute", "1.1/../../outside", "1.1\\outside"])
def test_archive_traversal_is_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        _validate_member(tarfile.TarInfo(name))


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE]
)
def test_archive_links_and_devices_are_rejected(kind):
    member = tarfile.TarInfo("1.1/data/en-US.jsonl")
    member.type = kind
    with pytest.raises(ValueError, match="member type"):
        _validate_member(member)


def test_expected_regular_archive_entry_is_accepted():
    member = tarfile.TarInfo("./1.1/data/en-US.jsonl")
    member.size = 100
    _validate_member(member)


class RecoveryRun:
    """No model imports or device inspection are needed to exercise stage commits."""

    def __init__(self, path):
        self.path = path
        self.recovered = []

    def verify_artifacts(self, prefixes):
        artifacts = read_json(self.path / "manifest.json")["artifacts"]
        for relative in prefixes:
            assert sha256_file(self.path / relative) == artifacts[relative]["sha256"]

    @contextmanager
    def stage(self, name):
        yield
        self.recovered.append(name)
        atomic_json(self.path / "stages" / f"{name}.json", {"status": "complete"})


def test_committed_outputs_recover_interrupted_stage_without_rewriting_them(tmp_path):
    output = tmp_path / "preparation" / "train.jsonl"
    summary = tmp_path / "preparation" / "data.json"
    atomic_text(output, "fixed content\n")
    atomic_json(summary, {"files": {output.name: sha256_file(output)}})
    artifacts = {
        str(path.relative_to(tmp_path)): {"sha256": sha256_file(path)} for path in (output, summary)
    }
    atomic_json(tmp_path / "manifest.json", {"artifacts": artifacts})
    atomic_json(tmp_path / "stages" / "data.json", {"status": "interrupted"})
    run = RecoveryRun(tmp_path)
    digest = sha256_file(output)
    assert _resume_completed_stage(run, "data", summary, [output])["files"] == {output.name: digest}
    assert run.recovered == ["data"]
    assert sha256_file(output) == digest
    _resume_completed_stage(run, "data", summary, [output])
    assert run.recovered == ["data"]


def test_unregistered_summary_does_not_claim_stage_completion(tmp_path):
    summary = tmp_path / "preparation" / "data.json"
    atomic_json(summary, {"files": {}})
    atomic_json(tmp_path / "manifest.json", {"artifacts": {}})
    run = RecoveryRun(tmp_path)
    assert _resume_completed_stage(run, "data", summary, []) is None
    assert run.recovered == []


def test_registered_summary_cannot_hide_uncommitted_outputs(tmp_path):
    summary = tmp_path / "preparation" / "data.json"
    output = tmp_path / "preparation" / "train.jsonl"
    atomic_text(output, "uncommitted")
    atomic_json(summary, {"files": {output.name: sha256_file(output)}})
    atomic_json(
        tmp_path / "manifest.json",
        {
            "artifacts": {
                str(summary.relative_to(tmp_path)): {"sha256": sha256_file(summary)},
            }
        },
    )
    with pytest.raises(ValueError, match="not committed"):
        _resume_completed_stage(RecoveryRun(tmp_path), "data", summary, [output])
