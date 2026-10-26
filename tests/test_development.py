"""Development subsets remain separate from the intact scientific partitions."""

import copy
import random
from contextlib import contextmanager
from dataclasses import replace

import pytest

from src.config import SPLITS, ExperimentConfig
from src.data import development_selection, load_examples, prepare_data
from src.io import atomic_json, canonical_hash, read_json, read_jsonl, sha256_file, write_jsonl


def _splits():
    return {
        split: [{"id": f"{split}-{i:05d}", "utterance": f"request {i}"} for i in range(count)]
        for split, count in zip(SPLITS, (11514, 2033, 2974), strict=True)
    }


def test_full_and_development_defaults_are_distinct_and_round_trip():
    full = ExperimentConfig(model_revision="a" * 40)
    development = ExperimentConfig.development(model_revision="a" * 40)
    full.validate()
    assert (full.preset, full.epochs, full.train_example_limit, full.validation_example_limit) == (
        "full",
        5,
        None,
        None,
    )
    assert (development.preset, development.epochs) == ("development", 1)
    assert (development.train_example_limit, development.validation_example_limit) == (128, 32)
    assert (
        development.benchmark_requests,
        development.benchmark_warmup,
        development.benchmark_repetitions,
    ) == (16, 2, 1)
    assert full.answer_only_logits and development.answer_only_logits
    assert ExperimentConfig.from_dict(development.to_dict()) == development
    assert ExperimentConfig.development().model_revision == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"preset": "tiny"},
        {"train_example_limit": None},
        {"train_example_limit": 0},
        {"train_example_limit": True},
        {"train_example_limit": 128.0},
        {"train_example_limit": 11515},
        {"validation_example_limit": 2034},
        {"validation_example_limit": "32"},
        {"validation_example_limit": 8},
        {"benchmark_warmup": 17},
    ],
)
def test_development_rejects_invalid_sampling_budgets(changes):
    with pytest.raises(ValueError):
        replace(ExperimentConfig.development(), **changes).validate(require_revision=False)


@pytest.mark.parametrize("name", ["train_example_limit", "validation_example_limit"])
def test_full_protocol_forbids_hidden_subsets(name):
    with pytest.raises(ValueError, match="all official"):
        replace(ExperimentConfig(), **{name: 1}).validate(require_revision=False)


def test_selection_is_paired_deterministic_ordered_and_does_not_use_labels_or_test():
    config = ExperimentConfig.development(model_revision="a" * 40)
    splits = _splits()
    del splits["test"]
    original = copy.deepcopy(splits)
    random.seed(17)
    before = random.getstate()
    first = development_selection(config, splits)
    assert random.getstate() == before
    assert first == development_selection(config, splits)
    assert splits == original
    for split, limit in (("train", 128), ("validation", 32)):
        ids = first["splits"][split]["ids"]
        assert len(ids) == len(set(ids)) == limit
        assert ids == sorted(ids)
        for row in splits[split]:
            row["target"] = "This annotation is never inspected for sampling"
    assert first == development_selection(config, splits)
    second = development_selection(replace(config, seed=2028), splits)
    assert first["splits"]["train"]["ids"] != second["splits"]["train"]["ids"]


class SelectionRun:
    def __init__(self, path, config):
        self.path, self.config = path, config
        atomic_json(path / "manifest.json", {"artifacts": {}})

    def verify_identity(self):
        pass

    def require_stage(self, name):
        assert name in ("data", "dataset", "tokenization")

    @contextmanager
    def writer_lock(self):
        yield

    @contextmanager
    def stage(self, name):
        yield

    def register_artifacts(self, paths):
        manifest = read_json(self.path / "manifest.json")
        for path in paths:
            manifest["artifacts"][str(path.relative_to(self.path))] = {"sha256": sha256_file(path)}
        atomic_json(self.path / "manifest.json", manifest)

    def verify_artifacts(self, prefixes):
        for name, value in read_json(self.path / "manifest.json")["artifacts"].items():
            if name in prefixes and sha256_file(self.path / name) != value["sha256"]:
                raise ValueError("Registered artifact changed")


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    """Exercise the actual preparation commit, without archive/model downloads."""
    run = SelectionRun(tmp_path, ExperimentConfig.development(model_revision="a" * 40))
    splits = _splits()
    for rows in splits.values():
        for row in rows:
            row["novel_input"] = True
    source = tmp_path / "shared_dataset"
    source.mkdir()
    write_jsonl(source / "en-US.jsonl", [{"raw": "Conversion is independently covered"}])
    monkeypatch.setattr("src.data.verify_shared_dataset", lambda _: source)
    monkeypatch.setattr(
        "src.data.build_prepared_rows",
        lambda *_: (splits, {"intents": list(range(60)), "slots": list(range(55))}, {}),
    )
    prepare_data(run)
    paths = []
    for split, rows in splits.items():
        path = tmp_path / "preparation" / f"tokenized_{split}.jsonl"
        write_jsonl(path, [{**row, "input_ids": [1, 2], "labels": [-100, 2]} for row in rows])
        paths.append(path)
    token_summary = tmp_path / "preparation" / "tokenization.json"
    atomic_json(token_summary, {"files": {p.name: sha256_file(p) for p in paths}})
    run.register_artifacts([*paths, token_summary])
    return run


def test_loading_development_is_paired_and_preserves_complete_disk_partitions(prepared_run):
    run = prepared_run
    directory = run.path / "preparation"
    before = {p.name: sha256_file(p) for p in directory.iterdir()}
    summary = read_json(directory / "data.json")
    assert summary["split_counts"] == dict(zip(SPLITS, (11514, 2033, 2974), strict=True))
    selection = read_json(directory / "development_selection.json")
    assert selection["config_sha256"] == canonical_hash(run.config.to_dict())
    assert "development_selection.json" in summary["files"]
    for split, limit in (("train", 128), ("validation", 32)):
        lora_rows = load_examples(run, split)
        qlora_rows = load_examples(run, split)
        prepared_rows = load_examples(run, split, tokenized=False)
        assert len(lora_rows) == limit
        assert lora_rows == qlora_rows
        assert [r["id"] for r in lora_rows] == [r["id"] for r in prepared_rows]
        assert [r["id"] for r in lora_rows] == selection["splits"][split]["ids"]
    assert len(read_jsonl(directory / "test.jsonl")) == 2974
    assert len(read_jsonl(directory / "tokenized_train.jsonl")) == 11514
    assert before == {p.name: sha256_file(p) for p in directory.iterdir()}


@pytest.mark.parametrize("tokenized", [False, True])
def test_development_cannot_load_test_examples(prepared_run, tokenized):
    with pytest.raises(RuntimeError, match="held-out test"):
        load_examples(prepared_run, "test", tokenized=tokenized)


def test_registered_but_wrong_selection_is_rejected(prepared_run):
    path = prepared_run.path / "preparation" / "development_selection.json"
    selection = read_json(path)
    selection["splits"]["train"]["ids"][0] = "train-00000"
    atomic_json(path, selection)
    prepared_run.register_artifacts([path])
    with pytest.raises(ValueError, match="selection disagrees"):
        load_examples(prepared_run, "train")


def test_selection_cannot_be_reused_with_a_changed_configuration(prepared_run):
    prepared_run.config = replace(prepared_run.config, seed=2028)
    with pytest.raises(ValueError, match="selection disagrees"):
        load_examples(prepared_run, "validation")


def test_tokenized_subset_cannot_replace_the_complete_official_split(prepared_run):
    path = prepared_run.path / "preparation" / "tokenized_train.jsonl"
    write_jsonl(path, read_jsonl(path)[:128])
    prepared_run.register_artifacts([path])
    with pytest.raises(ValueError, match="complete official split"):
        load_examples(prepared_run, "train")


def test_full_load_returns_all_rows_and_does_not_apply_development_selection(prepared_run):
    prepared_run.config = ExperimentConfig(model_revision="a" * 40)
    assert len(load_examples(prepared_run, "train")) == 11514
    assert len(load_examples(prepared_run, "validation")) == 2033
    assert len(load_examples(prepared_run, "test")) == 2974
