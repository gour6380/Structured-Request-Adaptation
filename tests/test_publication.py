"""Publication audits use saved text fixtures, never model or backend execution."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.io import atomic_json, read_json, read_jsonl, sha256_file, write_jsonl
from src.metrics import evaluate_predictions, score_prediction
from src.publication import (
    ARMS,
    SYSTEMS,
    _qualify_memory,
    _selection,
    export_publication,
    verify_publication,
)


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "runs/run_001"
    run.mkdir(parents=True)
    atomic_json(run / "config.json", {"epochs": 1, "seed": 2027})
    atomic_json(run / "source_snapshot/manifest.json", {})
    (run / "source_snapshot/src").mkdir()
    (run / "source_snapshot/src/config.py").write_text("# original source\n")
    (run / "source_snapshot/notebooks").mkdir()
    atomic_json(run / "source_snapshot/notebooks/lora_qlora.ipynb", {"cells": []})
    for name in _selection(run):
        path = run / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if path.suffix == ".json" else "original evidence\n")
    ontology = {"intents": ["set"], "slots": ["time"]}
    atomic_json(run / "preparation/ontology.json", ontology)
    target = {"intent": "set", "slots": [{"type": "time", "value": "seven"}] * 2}
    memory = {
        "cuda_allocated_bytes": 1000,
        "cuda_reserved_bytes": 2000,
        "rss_bytes": 4000,
        "samples": 2,
        "elapsed_seconds": 0.001,
        "sampling_errors": [],
        "system_swap_out_bytes_delta": 0,
        "system_swap_activity_observed": False,
    }
    for system in SYSTEMS:
        texts = [json.dumps(target), "```json\n{}\n```"]
        rows = []
        for index, text in enumerate(texts):
            rows.append(
                {
                    "id": str(index),
                    "utterance": "seven seven",
                    "target": target,
                    "prediction_text": text,
                    "novel_input": index == 0,
                    "answer_nll_sum": 1.0,
                    "answer_tokens": 2,
                    **score_prediction(text, target, ontology["intents"], ontology["slots"]),
                }
            )
        metrics = evaluate_predictions(rows, ontology["intents"], ontology["slots"])
        metrics.update(answer_nll_sum=2.0, answer_tokens=4, answer_token_nll=0.5)
        metrics["novel_input_subset"].update(
            answer_nll_sum=1.0, answer_tokens=2, answer_token_nll=0.5
        )
        write_jsonl(run / f"evaluations/{system}/test/predictions.jsonl", rows)
        atomic_json(run / f"evaluations/{system}/test/metrics.json", metrics)
        benchmark_rows = [
            {
                **row,
                "repetition": 1,
                "end_to_end_seconds": 0.4,
                "generation_seconds": 0.3,
                "generated_tokens": 2,
            }
            for row in rows
        ]
        write_jsonl(run / f"benchmarks/{system}/requests.jsonl", benchmark_rows)
        atomic_json(run / "benchmarks/requests.json", {"example_ids": ["0", "1"], "repetitions": 1})
        atomic_json(
            run / f"benchmarks/{system}/summary.json",
            {
                "observed_memory": memory,
                "median_seconds": 0.4,
                "p95_seconds": 0.4,
                "generated_tokens_per_second": 5,
                "measured_requests": 2,
                "repetitions": 1,
                "requests_per_repetition": 2,
                "generated_tokens": 4,
                "end_to_end_seconds": 0.8,
                "generation_seconds": 0.6,
                "requests_per_second": 2.5,
                "metrics_by_repetition": [
                    evaluate_predictions(rows, ontology["intents"], ontology["slots"])
                ],
            },
        )
        if system in ARMS:
            atomic_json(
                run / f"trials/{system}/training_summary.json",
                {
                    "completed": True,
                    "training_seconds": 20,
                    "wall_seconds": 24,
                    "epochs_completed": 1,
                    **{
                        f"{phase}_memory": memory
                        for phase in ("training", "fit", "loading", "validation")
                    },
                },
            )
            atomic_json(
                run / f"trials/{system}/metrics/epoch_01.json",
                {
                    "epoch": 1,
                    "global_step": 1,
                    "validation_seconds": 0.4,
                    "metrics": metrics,
                },
            )
    (run / "inputs/base_model/model.safetensors").write_bytes(b"excluded weight")
    (run / "optimizer.pt").write_bytes(b"excluded optimizer")
    (run / "trials/lora/metrics/epoch_01_predictions.jsonl").write_text(
        "excluded epoch predictions\n"
    )
    atomic_json(
        run / "manifest.json",
        {
            "artifacts": {
                "config.json": {"sha256": sha256_file(run / "config.json")},
            }
        },
    )
    return run


def _hashes(root):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_export_recomputes_metrics_preserves_originals_and_omits_weights(completed_run, tmp_path):
    before = _hashes(completed_run)
    output = export_publication(completed_run, tmp_path / "results/run_001")
    assert _hashes(completed_run) == before
    result = verify_publication(output)
    assert result["verified"]
    assert result["rescored_test_examples"] == dict.fromkeys(SYSTEMS, 2)
    assert not list(output.rglob("*.safetensors"))
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("epoch_*_predictions.jsonl"))
    assert (output / "original_reports/results.md").read_bytes() == (
        completed_run / "reports/results.md"
    ).read_bytes()
    assert (output / "source_snapshot/notebooks/lora_qlora.ipynb").is_file()
    summary = read_json(output / "summary.json")
    assert summary["systems"]["lora"]["test"]["joint_exact_match"] == 0.5
    assert (
        summary["diagnostics"]["fenced_baselines"]["quantized_base"]["outputs_starting_code_fence"]
        == 1
    )
    training = summary["systems"]["lora"]["training"]["training_memory"]
    assert training["samples"] is None and training["elapsed_seconds"] is None
    assert training["cuda_allocated_bytes"] == 1000
    assert training["memory_metadata_status"] == "legacy_aggregate_maxima_only"
    benchmark = summary["systems"]["lora"]["benchmark"]["memory"]
    assert benchmark["samples"] == 2 and benchmark["elapsed_seconds"] == 0.001
    with (output / "memory_runtime.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert next(row for row in rows if row["phase"] == "training")["samples"] == ""
    (output / "report.md").write_text("Separately authored narrative\n")
    assert verify_publication(output)["verified"]


def test_tampered_prediction_fails_hash_check(completed_run, tmp_path):
    output = export_publication(completed_run, tmp_path / "publication")
    path = output / "evaluations/lora/test/predictions.jsonl"
    path.write_text(
        path.read_text().replace('"joint_exact_match": true', '"joint_exact_match": false')
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_publication(output)


def test_consistent_hash_tamper_still_fails_metric_rescoring(completed_run, tmp_path):
    output = export_publication(completed_run, tmp_path / "publication")
    relative = "evaluations/lora/test/metrics.json"
    path = output / relative
    metrics = read_json(path)
    metrics["joint_exact_match"] = 1.0
    atomic_json(path, metrics)
    manifest = read_json(output / "publication_manifest.json")
    manifest["files"][relative].update(sha256=sha256_file(path), bytes=path.stat().st_size)
    atomic_json(output / "publication_manifest.json", manifest)
    with pytest.raises(ValueError, match="Metric mismatch"):
        verify_publication(output)


def test_export_rejects_in_run_or_existing_output(completed_run, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        export_publication(completed_run, completed_run / "public")
    output = tmp_path / "exists"
    output.mkdir()
    (output / "keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        export_publication(completed_run, output)
    assert (output / "keep").read_text() == "unchanged"


def test_export_rejects_symlink_evidence(completed_run, tmp_path):
    outside = tmp_path / "secret"
    outside.write_text("outside source")
    path = completed_run / "source_snapshot/src/config.py"
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        export_publication(completed_run, tmp_path / "publication")


def test_bad_original_checksum_stops_export(completed_run, tmp_path):
    atomic_json(completed_run / "config.json", {"epochs": 1, "seed": 999})
    with pytest.raises(ValueError, match="Original artifact checksum"):
        export_publication(completed_run, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_duplicate_test_ids_rejected(completed_run, tmp_path):
    path = completed_run / "evaluations/lora/test/predictions.jsonl"
    rows = read_jsonl(path)
    rows[1]["id"] = rows[0]["id"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Prediction IDs must be unique"):
        export_publication(completed_run, tmp_path / "publication")


def test_duplicate_benchmark_ids_rejected(completed_run, tmp_path):
    path = completed_run / "benchmarks/lora/requests.jsonl"
    rows = read_jsonl(path)
    rows[1]["id"] = rows[0]["id"]
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="duplicate-ID"):
        export_publication(completed_run, tmp_path / "publication")


def test_benchmark_summary_inconsistency_rejected(completed_run, tmp_path):
    path = completed_run / "benchmarks/lora/summary.json"
    summary = read_json(path)
    summary["median_seconds"] = 9.0
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="Metric mismatch: benchmark.lora.median_seconds"):
        export_publication(completed_run, tmp_path / "publication")


def test_failed_rescore_leaves_no_partial_export(completed_run, tmp_path):
    path = completed_run / "evaluations/lora/test/metrics.json"
    metrics = read_json(path)
    metrics["joint_exact_match"] = 1.0
    atomic_json(path, metrics)
    with pytest.raises(ValueError, match="Metric mismatch"):
        export_publication(completed_run, tmp_path / "publication")
    assert not list(tmp_path.glob("*publication*"))
    assert not list(tmp_path.glob(".*publication*"))


def test_versioned_complete_memory_preserved_and_legacy_unknown_swap():
    complete = {"metadata_version": 2, "metadata_complete": True, "samples": 17}
    assert _qualify_memory(complete, aggregate=True)["samples"] == 17
    assert (
        _qualify_memory({"system_swap_activity_observed": True}, aggregate=True)[
            "system_swap_activity_observed"
        ]
        is True
    )
    assert (
        _qualify_memory({"system_swap_activity_observed": False}, aggregate=True)[
            "system_swap_activity_observed"
        ]
        is None
    )


def test_publication_import_requires_no_ml_packages():
    project = Path(__file__).resolve().parents[1]
    code = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
blocked = {'torch', 'transformers', 'peft', 'numpy', 'bitsandbytes', 'src.runs', 'src.runtime', 'src.models'}
def guard(name, *args, **kwargs):
    if name in blocked or name.split('.')[0] in blocked:
        raise AssertionError('Unexpected backend import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from src.publication import export_publication, verify_publication
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code, str(project)],
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
