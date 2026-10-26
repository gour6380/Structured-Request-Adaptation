"""Rendering/report tests use stored synthetic rows, never trained models."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.config import ARMS, SYSTEMS, ExperimentConfig
from src.io import atomic_json, read_json, write_jsonl
from src.plots import (
    _comparison_summary,
    _disagreements,
    _training_summary_for_report,
    collect_results,
    plot_disagreements,
    plot_losses,
    plot_memory_runtime,
    plot_task_metrics,
    plot_validation,
    write_report,
)


def fake_run(path):
    return SimpleNamespace(
        path=path,
        verify_identity=lambda: None,
        verify_artifacts=lambda: None,
        writer_lock=lambda: nullcontext(),
        register_artifacts=lambda *_, **__: None,
        config=ExperimentConfig(model_revision="a" * 40),
        environment={"device": "mps", "compute_dtype": "bfloat16"},
    )


def test_empty_run_is_explicitly_incomplete(tmp_path) -> None:
    result = collect_results(fake_run(tmp_path))
    assert result["complete"] is False
    assert len(result["missing"]) == 10
    assert result["test"] == {}
    assert result["benchmarks"] == {}


def test_metric_file_without_complete_stage_is_not_result(tmp_path) -> None:
    atomic_json(
        tmp_path / "evaluations" / "lora" / "test" / "metrics.json", {"joint_exact_match": 1.0}
    )
    atomic_json(
        tmp_path / "stages" / "evaluate_lora_test.json",
        {"status": "failed", "error": "interrupted"},
    )
    result = collect_results(fake_run(tmp_path))
    assert "lora" not in result["test"]
    assert "evaluate_lora_test" in result["failed_stages"]


def test_report_states_missing_evidence_without_fabricating_values(tmp_path) -> None:
    path = write_report(fake_run(tmp_path))
    text = path.read_text()
    assert "INCOMPLETE" in text
    assert "not measured" in text
    assert "No completed test predictions" in text
    assert "No completed timing or memory measurements" in text
    assert "no overall winner claim" in text


def test_report_labels_full_test_failure_counts_separately_from_novel_subset(tmp_path):
    atomic_json(
        tmp_path / "evaluations" / "lora" / "test" / "metrics.json",
        {
            "example_count": 10,
            "failure_counts": {"missing_slots": 3},
            "novel_input_subset": {
                "example_count": 8,
                "joint_exact_match": 0.5,
                "failure_counts": {"missing_slots": 1},
            },
        },
    )
    atomic_json(tmp_path / "stages" / "evaluate_lora_test.json", {"status": "complete"})
    write_jsonl(tmp_path / "evaluations" / "lora" / "test" / "predictions.jsonl", [])
    report = write_report(fake_run(tmp_path)).read_text()
    assert "| lora | 8 | 0.5000 |" in report
    assert "lora: full-test n=10; failure categories={'missing_slots': 3}" in report
    assert "categories can overlap" in report
    assert "novel n=8" not in report


def test_legacy_training_coverage_is_unavailable_without_mutating_evidence():
    from copy import deepcopy

    saved = {
        "memory": {
            "cuda_allocated_bytes": 123,
            "samples": 2,
            "elapsed_seconds": 0.002,
            "sampling_errors": [],
        },
        "training_seconds": 100,
    }
    before = deepcopy(saved)
    display = _training_summary_for_report(saved)
    assert display["memory"]["cuda_allocated_bytes"] == 123
    assert display["memory"]["samples"] is None
    assert display["memory"]["elapsed_seconds"] is None
    assert display["memory"]["metadata_complete"] is False
    assert display["training_seconds"] == 100
    assert saved == before


def test_paired_disagreements_use_ids_and_include_both_directions(tmp_path) -> None:
    def row(identifier, correct):
        return {
            "id": identifier,
            "target": {"intent": "x", "slots": []},
            "utterance": "request",
            "joint_exact_match": correct,
            "prediction_text": "{}",
        }

    write_jsonl(
        tmp_path / "evaluations" / "lora" / "test" / "predictions.jsonl",
        [row("a", True), row("b", False)],
    )
    write_jsonl(
        tmp_path / "evaluations" / "qlora" / "test" / "predictions.jsonl",
        [row("b", True), row("a", False)],
    )
    result = _disagreements(fake_run(tmp_path), {"lora": {}, "qlora": {}})
    assert result["pairs"][0]["left_only"] == 1
    assert result["pairs"][0]["right_only"] == 1
    assert [row["id"] for row in result["examples"]] == ["a", "b"]


def test_disagreements_reject_nonpaired_test_sets(tmp_path) -> None:
    write_jsonl(tmp_path / "evaluations" / "base" / "test" / "predictions.jsonl", [{"id": "a"}])
    write_jsonl(tmp_path / "evaluations" / "lora" / "test" / "predictions.jsonl", [{"id": "b"}])
    with pytest.raises(ValueError, match="IDs differ"):
        _disagreements(fake_run(tmp_path), {"base": {}, "lora": {}})


def test_comparison_preserves_quality_direction_and_separate_memory_measures() -> None:
    result = {
        "test": {"lora": {"joint_exact_match": 0.8}, "qlora": {"joint_exact_match": 0.7}},
        "training": {
            "lora": {"memory": {"rss_bytes": 100, "mps_allocated_bytes": 80}},
            "qlora": {"memory": {"rss_bytes": 90, "mps_allocated_bytes": 40}},
        },
        "benchmarks": {},
    }
    comparison = _comparison_summary(result)
    assert comparison["quality"]["joint_exact_match"] == pytest.approx(-0.1)
    assert comparison["memory"]["training/rss_bytes"]["qlora_minus_lora_bytes"] == -10
    assert comparison["memory"]["training/mps_allocated_bytes"]["relative_reduction"] == 0.5


def test_missing_arm_has_no_invented_comparison() -> None:
    result = {"test": {"lora": {"joint_exact_match": 0.8}}, "training": {}, "benchmarks": {}}
    assert _comparison_summary(result) == {"quality": {}, "memory": {}, "runtime": {}}


def development_run(path):
    run = fake_run(path)
    values = run.config.to_dict()
    values.update(
        preset="development",
        epochs=1,
        train_example_limit=128,
        validation_example_limit=32,
        benchmark_requests=10,
        benchmark_warmup=1,
        benchmark_repetitions=1,
    )
    run.config = SimpleNamespace(**values, to_dict=lambda: dict(values))
    return run


def test_development_collect_labels_expected_budget_and_deliberate_test_omission(tmp_path):
    result = collect_results(development_run(tmp_path))
    assert result["preset"] == "development"
    assert "no full-protocol" in result["quality_scope"]
    assert result["expected"]["epochs_per_arm"] == 1
    assert result["expected"]["benchmark_requests"] == 10
    assert result["expected"]["test_enabled"] is False
    assert result["test"] == {}
    assert result["complete"] is False
    assert "disabled" in result["deliberately_omitted"][0]
    assert any("1-epoch lora" in item for item in result["missing"])
    assert any(
        "10-request benchmark (1 repetitions, 1 warmups)" in item for item in result["missing"]
    )
    assert not any("Full official test" in item for item in result["missing"])


def test_development_never_becomes_complete_scientific_comparison(tmp_path):
    for arm in ARMS:
        atomic_json(tmp_path / "trials" / arm / "training_summary.json", {"epochs_completed": 1})
        atomic_json(tmp_path / "stages" / f"train_{arm}.json", {"status": "complete"})
    for system in SYSTEMS:
        atomic_json(tmp_path / "benchmarks" / system / "summary.json", {})
        atomic_json(tmp_path / "stages" / f"benchmark_{system}.json", {"status": "complete"})
        # Even stray copied test evidence cannot turn a development run into a
        # scientific comparison or feed its paired quality calculation.
        atomic_json(
            tmp_path / "evaluations" / system / "test" / "metrics.json",
            {"joint_exact_match": 1.0},
        )
        atomic_json(tmp_path / "stages" / f"evaluate_{system}_test.json", {"status": "complete"})
    result = collect_results(development_run(tmp_path))
    assert result["preset_complete"] is True
    assert result["complete"] is False
    assert result["test"] == {}
    assert _comparison_summary(result)["quality"] == {}
    report = write_report(development_run(tmp_path)).read_text()
    assert report.startswith("# DEVELOPMENT PROBE")
    assert "full scientific comparison protocol is NOT COMPLETED" in report
    assert "COMPLETE measured comparison" not in report
    assert "Primary whole-record exact-match difference" not in report
    assert "no overall quality winner" in report


def test_development_report_retains_saved_validation_and_incomplete_training_history(tmp_path):
    atomic_json(
        tmp_path / "evaluations" / "base" / "validation" / "metrics.json",
        {"example_count": 32, "joint_exact_match": 0.125},
    )
    atomic_json(tmp_path / "stages" / "evaluate_base_validation.json", {"status": "complete"})
    write_jsonl(
        tmp_path / "trials" / "lora" / "metrics" / "train.jsonl",
        [{"global_step": 1, "loss": 2.75}],
    )
    atomic_json(
        tmp_path / "trials" / "lora" / "metrics" / "epoch_01.json",
        {"epoch": 1, "metrics": {"answer_token_nll": 2.25, "joint_exact_match": 0.25}},
    )
    run = development_run(tmp_path)
    result = collect_results(run)
    assert result["validation"]["base"]["example_count"] == 32
    assert result["history"]["lora"]["training"][0]["loss"] == 2.75
    assert result["training"] == {}
    report = write_report(run).read_text()
    assert '"loss": 2.75' in report
    assert '"answer_token_nll": 2.25' in report
    assert "training example limit=128" in report
    assert "10 requests, 1 excluded warmups, 1 measured repetition" in report
    assert "No saved training or epoch-validation history" in report
    assert read_json(tmp_path / "reports" / "paired_comparison.json")["quality"] == {}


def test_development_empty_plots_and_report_do_not_launch_experiments(tmp_path, monkeypatch):
    import matplotlib

    from src import evaluate, train

    matplotlib.use("Agg")

    def forbidden(*args, **kwargs):
        pytest.fail("Rendering must never launch model experiments")

    monkeypatch.setattr(train, "train_adapter", forbidden)
    monkeypatch.setattr(evaluate, "evaluate_model", forbidden)
    monkeypatch.setattr(evaluate, "benchmark_model", forbidden)
    run = development_run(tmp_path)
    for render in (
        plot_losses,
        plot_validation,
        plot_task_metrics,
        plot_memory_runtime,
        plot_disagreements,
    ):
        path = render(run)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert write_report(run).read_text().startswith("# DEVELOPMENT PROBE")
    assert not (tmp_path / "trials").exists()
    assert not (tmp_path / "evaluations").exists()
