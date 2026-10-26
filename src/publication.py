"""Export and verify saved evidence offline, without a model or execution backend.

The original run is never changed. An explicit allowlist omits model weights,
optimizer state, caches and event logs. A publication is evidence for auditing
results, not a self-contained model checkpoint or a substitute for the full run.
"""

from __future__ import annotations

import csv
import io
import math
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from .io import atomic_json, atomic_text, read_json, read_jsonl, sha256_file
from .metrics import evaluate_predictions, score_prediction

SYSTEMS = ("base", "quantized_base", "lora", "qlora")
ARMS = ("lora", "qlora")
METRICS = (
    "example_count",
    "joint_exact_match",
    "json_valid_rate",
    "schema_valid_rate",
    "intent_accuracy",
    "intent_macro_f1",
    "slot_precision",
    "slot_recall",
    "slot_f1",
    "answer_token_nll",
)
FIGURES = ("losses", "validation", "task_metrics", "memory_runtime", "disagreements")
AUDITS = (
    "data.json",
    "duplicate_audit.json",
    "tokenization.json",
    "ontology.json",
    "system_prompt.txt",
    "architecture.json",
    "architecture.txt",
    "preflight.json",
    "preflight_lora.json",
    "preflight_qlora.json",
)
EXCLUSIONS = [
    "All base/adaptor model weights, optimizer and scheduler state, recovery checkpoints",
    "Raw dataset archives and prepared/tokenized examples (test references remain in predictions)",
    "Epoch validation predictions (epoch metric summaries are included)",
    "TensorBoard event logs, worker logs/jobs, caches, virtual environments and credentials",
    "Other runs and live workspace source not present in this run's source snapshot",
]


def _safe_file(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Unsafe evidence path: {relative}")
    path = root / name
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError(f"Symlink evidence is not supported: {relative}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or unsafe evidence file: {relative}")
    return path


def _selection(run: Path) -> dict[str, str]:
    """Map original paths to publication paths; never recursively copy arbitrary data."""
    fixed = [
        "config.json",
        "environment.json",
        "identity.json",
        "backend_binding.json",
        "manifest.json",
        "evaluations/frozen_protocol.json",
        "benchmarks/requests.json",
        "inputs/dataset/manifest.json",
        "inputs/model_snapshot.json",
        "inputs/quantization.json",
        "inputs/base_model/config.json",
        "inputs/base_model/generation_config.json",
        "inputs/base_model/LICENSE",
        "inputs/quantized_base/config.json",
        "inputs/quantized_base/generation_config.json",
        "initial_adapter/adapter_config.json",
        *[f"preparation/{name}" for name in AUDITS],
        *[f"figures/{name}.png" for name in FIGURES],
    ]
    epochs = read_json(_safe_file(run, "config.json"))["epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("A positive epoch budget is required")
    for system in SYSTEMS:
        fixed.extend(
            f"evaluations/{system}/test/{name}" for name in ("metrics.json", "predictions.jsonl")
        )
        fixed.extend(f"benchmarks/{system}/{name}" for name in ("summary.json", "requests.jsonl"))
    for arm in ARMS:
        fixed.append(f"trials/{arm}/training_summary.json")
        fixed.extend(f"trials/{arm}/metrics/{name}" for name in ("train.jsonl", "train.csv"))
        fixed.extend(
            f"trials/{arm}/metrics/epoch_{epoch:02d}.json" for epoch in range(1, epochs + 1)
        )
    selected = {name: name for name in fixed}
    for path in sorted((run / "source_snapshot").rglob("*")):
        if path.is_symlink():
            raise ValueError("Source snapshot contains a symlink")
        if path.is_file() and path.suffix in {".py", ".json", ".ipynb", ".md", ".txt", ".toml"}:
            relative = path.relative_to(run).as_posix()
            selected[relative] = relative
    if "source_snapshot/manifest.json" not in selected:
        raise ValueError("The original source snapshot is required")
    for name in (
        "results.md",
        "execution.md",
        "paired_comparison.json",
        "disagreements.json",
        "error_examples.json",
    ):
        selected[f"reports/{name}"] = f"original_reports/{name}"
    return selected


def _qualify_memory(memory: dict[str, Any], *, aggregate: bool) -> dict[str, Any]:
    result = dict(memory)
    if aggregate and not (
        memory.get("metadata_version") == 2 and memory.get("metadata_complete") is True
    ):
        result.update(
            memory_metadata_status="legacy_aggregate_maxima_only",
            metadata_complete=False,
            sampling_errors_complete=False,
            samples=None,
            elapsed_seconds=None,
            intervals=None,
            system_swap_out_bytes_delta=None,
            system_swap_activity_observed=True
            if memory.get("system_swap_activity_observed") is True
            else None,
        )
    else:
        result["memory_metadata_status"] = (
            "complete_disjoint_intervals" if aggregate else "single_interval"
        )
    return result


def _csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _derive(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    ontology = read_json(root / "preparation/ontology.json")
    summaries: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    test_rows, epochs, memory_rows = [], [], []
    for system in SYSTEMS:
        rows = read_jsonl(root / f"evaluations/{system}/test/predictions.jsonl")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError("Prediction IDs must be unique")
        metrics = read_json(root / f"evaluations/{system}/test/metrics.json")
        predictions[system] = rows
        summaries[system] = {
            "test": {key: metrics[key] for key in METRICS},
            "novel_test": {key: metrics["novel_input_subset"].get(key) for key in METRICS},
            "test_counts": metrics["counts"],
        }
        test_rows.append({"system": system, **summaries[system]["test"]})
        benchmark = read_json(root / f"benchmarks/{system}/summary.json")
        memory = _qualify_memory(benchmark["observed_memory"], aggregate=False)
        summaries[system]["benchmark"] = {
            key: benchmark.get(key)
            for key in (
                "median_seconds",
                "p95_seconds",
                "generated_tokens_per_second",
                "measured_requests",
                "requests_per_repetition",
                "repetitions",
                "warmup_requests",
                "batch_size",
                "execution_context",
                "latency_boundary",
            )
        }
        summaries[system]["benchmark"]["memory"] = memory
        memory_rows.append(_memory_row(system, "warm_inference", memory, benchmark))
        if system in ARMS:
            training = read_json(root / f"trials/{system}/training_summary.json")
            if training.get("completed") is not True:
                raise ValueError(f"Training is incomplete: {system}")
            summaries[system]["training"] = {
                key: training.get(key)
                for key in (
                    "epochs_completed",
                    "optimizer_updates",
                    "examples_seen",
                    "target_tokens_seen",
                    "training_seconds",
                    "wall_seconds",
                    "effective_batch_size",
                    "primary_checkpoint",
                    "initial_adapter_sha256",
                    "train_ids_hash",
                )
            }
            for phase in ("training", "fit", "loading", "validation"):
                mem = _qualify_memory(training.get(f"{phase}_memory", {}), aggregate=True)
                summaries[system]["training"][f"{phase}_memory"] = mem
                memory_rows.append(_memory_row(system, phase, mem, training))
            for path in sorted((root / f"trials/{system}/metrics").glob("epoch_*.json")):
                epoch = read_json(path)
                epochs.append(
                    {
                        "arm": system,
                        "epoch": epoch["epoch"],
                        "global_step": epoch["global_step"],
                        "validation_seconds": epoch["validation_seconds"],
                        **{key: epoch["metrics"].get(key) for key in METRICS},
                    }
                )
    left = {row["id"]: row for row in predictions["lora"]}
    right = {row["id"]: row for row in predictions["qlora"]}
    if left.keys() != right.keys():
        raise ValueError("The paired test example IDs differ")
    lora_only = sum(
        row["joint_exact_match"] and not right[key]["joint_exact_match"]
        for key, row in left.items()
    )
    qlora_only = sum(
        row["joint_exact_match"] and not left[key]["joint_exact_match"]
        for key, row in right.items()
    )
    diagnostics = {
        "fenced_baselines": {
            system: {
                "example_count": len(predictions[system]),
                "outputs_starting_code_fence": sum(
                    row["prediction_text"].lstrip().startswith("```") for row in predictions[system]
                ),
            }
            for system in ("base", "quantized_base")
        },
        "lora_qlora_paired": {
            "lora_only_correct": lora_only,
            "qlora_only_correct": qlora_only,
            "lora_minus_qlora_correct_examples": lora_only - qlora_only,
            "lora_minus_qlora_percentage_points": 100 * (lora_only - qlora_only) / len(left),
        },
        "format_policy": "Code fences are failures under the primary strict protocol; no outputs repaired.",
        "memory_policy": "Legacy aggregate maxima are preserved; cumulative sample, duration, error-completeness and swap metadata are unknown. Single-interval benchmark metadata is retained.",
        "seed_limitation": "One paired seed; observed differences do not establish superiority or equivalence.",
    }
    summary = {
        "schema_version": 1,
        "source_run": root.name,
        "systems": summaries,
        "diagnostics": diagnostics,
        "ontology_counts": {key: len(value) for key, value in ontology.items()},
        "configuration": read_json(root / "config.json"),
    }
    return summary, {
        "test_metrics.csv": _csv_text(test_rows),
        "epochs.csv": _csv_text(epochs),
        "memory_runtime.csv": _csv_text(memory_rows),
    }


def _memory_row(
    system: str, phase: str, memory: dict[str, Any], timing: dict[str, Any]
) -> dict[str, Any]:
    return {
        "system": system,
        "phase": phase,
        **{
            key: memory.get(key)
            for key in (
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
                "rss_bytes",
                "samples",
                "elapsed_seconds",
                "system_swap_out_bytes_delta",
                "system_swap_activity_observed",
                "memory_metadata_status",
            )
        },
        "training_seconds": timing.get("training_seconds") if phase == "training" else None,
        "fit_wall_seconds": timing.get("wall_seconds") if phase == "fit" else None,
        "median_request_seconds": timing.get("median_seconds")
        if phase == "warm_inference"
        else None,
        "p95_request_seconds": timing.get("p95_seconds") if phase == "warm_inference" else None,
    }


def _assert_subset(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"Metric mismatch: {label}")
        for key, value in expected.items():
            _assert_subset(value, actual.get(key), f"{label}.{key}")
    elif expected != actual:
        raise ValueError(f"Metric mismatch: {label}")


def _verify_benchmarks(root: Path, ontology: dict[str, list[str]]) -> dict[str, int]:
    selection = read_json(root / "benchmarks/requests.json")
    expected_ids = selection["example_ids"]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Benchmark selection IDs must be unique")
    counts: dict[str, int] = {}
    shared_references = None
    for system in SYSTEMS:
        rows = read_jsonl(root / f"benchmarks/{system}/requests.jsonl")
        summary = read_json(root / f"benchmarks/{system}/summary.json")
        repetitions = selection["repetitions"]
        if summary["repetitions"] != repetitions or len(rows) != len(expected_ids) * repetitions:
            raise ValueError(f"Benchmark repetition/count mismatch: {system}")
        for row in rows:
            _assert_subset(
                score_prediction(
                    row["prediction_text"], row["target"], ontology["intents"], ontology["slots"]
                ),
                row,
                f"benchmark.{system}.{row['id']}",
            )
        current_references = None
        for repetition in range(1, repetitions + 1):
            selected = [row for row in rows if row["repetition"] == repetition]
            if [row["id"] for row in selected] != expected_ids:
                raise ValueError(f"Benchmark selection or duplicate-ID mismatch: {system}")
            references = {
                row["id"]: (row["utterance"], row["target"], row["novel_input"]) for row in selected
            }
            if current_references is not None and references != current_references:
                raise ValueError("Benchmark references changed between repetitions")
            current_references = references
            _assert_subset(
                evaluate_predictions(selected, ontology["intents"], ontology["slots"]),
                summary["metrics_by_repetition"][repetition - 1],
                f"benchmark.{system}.{repetition}",
            )
        if shared_references is not None and current_references != shared_references:
            raise ValueError("Benchmark references changed between systems")
        shared_references = current_references
        timings = sorted(row["end_to_end_seconds"] for row in rows)
        position = (len(timings) - 1) * 0.95
        low, high = math.floor(position), math.ceil(position)
        # Sum original order to reproduce the persisted floating-point total.
        elapsed = sum(row["end_to_end_seconds"] for row in rows)
        generated = sum(row["generated_tokens"] for row in rows)
        _assert_subset(
            {
                "median_seconds": statistics.median(timings),
                "p95_seconds": timings[low] + (timings[high] - timings[low]) * (position - low),
                "end_to_end_seconds": elapsed,
                "generated_tokens": generated,
                "generated_tokens_per_second": generated / elapsed,
                "generation_seconds": sum(row["generation_seconds"] for row in rows),
                "requests_per_second": len(rows) / elapsed,
                "measured_requests": len(rows),
                "requests_per_repetition": len(expected_ids),
            },
            summary,
            f"benchmark.{system}",
        )
        counts[system] = len(rows)
    return counts


def verify_publication(output_path: Path) -> dict[str, Any]:
    """Check copied and derived hashes, then rescore every saved test prediction."""
    root = Path(output_path).resolve(strict=True)
    manifest = read_json(_safe_file(root, "publication_manifest.json"))
    for name, entry in manifest["files"].items():
        path = _safe_file(root, name)
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Publication checksum mismatch: {name}")
    ontology = read_json(root / "preparation/ontology.json")
    counts = {}
    references = None
    for system in SYSTEMS:
        rows = read_jsonl(root / f"evaluations/{system}/test/predictions.jsonl")
        stored = read_json(root / f"evaluations/{system}/test/metrics.json")
        recomputed = evaluate_predictions(rows, ontology["intents"], ontology["slots"])
        _assert_subset(recomputed, stored, system)
        for row in rows:
            _assert_subset(
                score_prediction(
                    row["prediction_text"], row["target"], ontology["intents"], ontology["slots"]
                ),
                row,
                f"{system}.{row['id']}",
            )
        current = {row["id"]: (row["utterance"], row["target"], row["novel_input"]) for row in rows}
        if references is not None and current != references:
            raise ValueError("System test references differ")
        references = current
        for selected, metrics in (
            (rows, stored),
            ([row for row in rows if row["novel_input"]], stored["novel_input_subset"]),
        ):
            nll = sum(row["answer_nll_sum"] for row in selected)
            tokens = sum(row["answer_tokens"] for row in selected)
            _assert_subset(
                {
                    "answer_nll_sum": nll,
                    "answer_tokens": tokens,
                    "answer_token_nll": nll / tokens if tokens else None,
                },
                metrics,
                system,
            )
        counts[system] = len(rows)
    benchmark_counts = _verify_benchmarks(root, ontology)
    summary, tables = _derive(root)
    summary["source_run"] = manifest["source_run_name"]
    if summary != read_json(root / "summary.json"):
        raise ValueError("Derived publication summary differs from its evidence")
    for name, text in tables.items():
        if (root / name).read_text() != text:
            raise ValueError(f"Derived publication table differs: {name}")
    return {
        "verified": True,
        "checked_files": len(manifest["files"]),
        "rescored_test_examples": counts,
        "rescored_benchmark_requests": benchmark_counts,
        "narrative_report": "report.md is separately authored and outside the export manifest",
    }


def export_publication(run_path: Path, output_path: Path) -> Path:
    """Create a new audit-only export atomically; never overwrite or mutate a run."""
    if Path(run_path).is_symlink() or Path(output_path).is_symlink():
        raise ValueError("Run and publication roots cannot be symlinks")
    run = Path(run_path).resolve(strict=True)
    output = Path(output_path).resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Publication output must be outside the original run")
    if output.exists():
        raise FileExistsError(f"Publication already exists: {output}")
    selected = _selection(run)
    original_manifest = read_json(_safe_file(run, "manifest.json"))
    files: dict[str, Any] = {}
    for original, destination in selected.items():
        source = _safe_file(run, original)
        digest = sha256_file(source)
        registered = original_manifest.get("artifacts", {}).get(original)
        if registered and registered.get("sha256") != digest:
            raise ValueError(f"Original artifact checksum mismatch: {original}")
        files[destination] = {
            "source_relative_path": original,
            "source_sha256": digest,
            "sha256": digest,
            "bytes": source.stat().st_size,
            "kind": "unchanged_original",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-publication-", dir=output.parent))
    try:
        for original, destination in selected.items():
            target = temporary / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_safe_file(run, original), target)
            if sha256_file(target) != files[destination]["sha256"]:
                raise ValueError(f"Evidence changed while copying: {original}")
        summary, tables = _derive(temporary)
        summary["source_run"] = run.name
        atomic_json(temporary / "summary.json", summary)
        for name, text in tables.items():
            atomic_text(temporary / name, text)
        for name in ("summary.json", *tables):
            path = temporary / name
            files[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "kind": "derived_offline",
            }
        atomic_json(
            temporary / "publication_manifest.json",
            {
                "schema_version": 1,
                "source_run_name": run.name,
                "original_run_manifest_sha256": sha256_file(run / "manifest.json"),
                "original_identity_sha256": sha256_file(run / "identity.json"),
                "exporter_source_sha256": sha256_file(Path(__file__)),
                "files": files,
                "exclusions": EXCLUSIONS,
                "purpose": "Offline audit evidence; original run and hashes remain unchanged. Not a reloadable model bundle.",
                "narrative_report": "report.md may be added later; it is not covered by this export manifest.",
            },
        )
        verify_publication(temporary)
        if output.exists():
            raise FileExistsError(f"Publication appeared during export: {output}")
        temporary.rename(output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
