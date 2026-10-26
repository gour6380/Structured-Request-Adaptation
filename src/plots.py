"""Read-only result collection and explicit rendering; never fit or evaluate."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ARMS, SYSTEMS
from .io import atomic_json, atomic_text, read_json, read_jsonl

COLORS = {"base": "#85929e", "quantized_base": "#4e79a7", "lora": "#e08e45", "qlora": "#3c9d91"}
if TYPE_CHECKING:
    from .runs import RunContext


def collect_results(run: RunContext) -> dict[str, Any]:
    """Collect existing evidence only; incomplete stages are explicitly listed."""
    run.verify_identity()
    run.verify_artifacts()
    stages = {path.stem: read_json(path) for path in sorted((run.path / "stages").glob("*.json"))}
    preset = getattr(run.config, "preset", "full")
    development = preset == "development"
    expected = {
        "epochs_per_arm": getattr(run.config, "epochs", 5),
        "benchmark_requests": getattr(run.config, "benchmark_requests", 100),
        "benchmark_warmup": getattr(run.config, "benchmark_warmup", 5),
        "benchmark_repetitions": getattr(run.config, "benchmark_repetitions", 3),
        "train_example_limit": getattr(run.config, "train_example_limit", None),
        "validation_example_limit": getattr(run.config, "validation_example_limit", None),
        "test_enabled": not development,
    }
    result: dict[str, Any] = {
        "run": run.path.name,
        "preset": preset,
        "quality_scope": (
            "Development subsets only; no full-protocol or held-out test quality claim"
            if development
            else "Official test set and novel-input subset, final equal-budget checkpoints"
        ),
        "expected": expected,
        "stages": stages,
        "training": {},
        "history": {},
        "validation": {},
        "test": {},
        "benchmarks": {},
        "missing": [],
        "deliberately_omitted": (
            ["Official test evaluation is disabled for the development preset"]
            if development
            else []
        ),
    }
    for arm in ARMS:
        path = run.path / "trials" / arm / "training_summary.json"
        if path.exists() and stages.get(f"train_{arm}", {}).get("status") == "complete":
            result["training"][arm] = read_json(path)
        else:
            result["missing"].append(f"Completed {expected['epochs_per_arm']}-epoch {arm} training")
        directory = run.path / "trials" / arm / "metrics"
        training_log = directory / "train.jsonl"
        result["history"][arm] = {
            "training": read_jsonl(training_log) if training_log.exists() else [],
            "validation": [read_json(path) for path in sorted(directory.glob("epoch_??.json"))],
        }
    for system in SYSTEMS:
        if not development:
            path = run.path / "evaluations" / system / "test" / "metrics.json"
            if (
                path.exists()
                and stages.get(f"evaluate_{system}_test", {}).get("status") == "complete"
            ):
                result["test"][system] = read_json(path)
            else:
                result["missing"].append(f"Full official test evaluation: {system}")
        path = run.path / "evaluations" / system / "validation" / "metrics.json"
        if (
            path.exists()
            and stages.get(f"evaluate_{system}_validation", {}).get("status") == "complete"
        ):
            result["validation"][system] = read_json(path)
        path = run.path / "benchmarks" / system / "summary.json"
        if path.exists() and stages.get(f"benchmark_{system}", {}).get("status") == "complete":
            result["benchmarks"][system] = read_json(path)
        else:
            result["missing"].append(
                f"{expected['benchmark_requests']}-request benchmark "
                f"({expected['benchmark_repetitions']} repetitions, "
                f"{expected['benchmark_warmup']} warmups): {system}"
            )
    result["preset_complete"] = not result["missing"]
    result["complete"] = result["preset_complete"] and not development
    result["failed_stages"] = {
        name: value
        for name, value in stages.items()
        if value.get("status") in ("failed", "interrupted")
    }
    return result


def _axes(ncols: int = 1, *, height: float = 4.5) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt.subplots(1, ncols, figsize=(6 * ncols, height), squeeze=False, layout="constrained")


def _empty(axis: Any, text: str = "No completed measurements available") -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes, color="#64748b")


def _save(run: Any, figure: Any, name: str) -> Path:
    import matplotlib.pyplot as plt

    path = run.path / "figures" / f"{name}.png"
    with run.writer_lock():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(path, dpi=170, facecolor="white")
        finally:
            plt.close(figure)
        run.register_artifacts([path], replace=True)
    return path


def plot_losses(run: RunContext) -> Path:
    result = collect_results(run)
    fig, axes = _axes(2)
    for arm in ARMS:
        directory = run.path / "trials" / arm / "metrics"
        log = directory / "train.jsonl"
        if log.exists():
            rows = read_jsonl(log)
            axes[0, 0].plot(
                [r["global_step"] for r in rows],
                [r["loss"] for r in rows],
                label=arm,
                color=COLORS[arm],
                linewidth=1.1,
            )
        epochs = [read_json(p) for p in sorted(directory.glob("epoch_??.json"))]
        if epochs:
            axes[0, 1].plot(
                [r["epoch"] for r in epochs],
                [r.get("metrics", r)["answer_token_nll"] for r in epochs],
                marker="o",
                label=arm,
                color=COLORS[arm],
            )
    for axis, title, xlabel in zip(
        axes[0],
        (
            "Training answer-token NLL",
            "Development validation answer-token NLL"
            if result["preset"] == "development"
            else "Full validation answer-token NLL",
        ),
        ("Optimizer update", "Epoch"),
        strict=True,
    ):
        axis.set(title=title, xlabel=xlabel, ylabel="NLL per supervised answer token")
        if axis.lines:
            axis.legend()
        else:
            _empty(axis)
    return _save(run, fig, "losses")


def plot_validation(run: RunContext) -> Path:
    result = collect_results(run)
    fig, axes = _axes(2)
    for arm in ARMS:
        rows = [
            read_json(p)
            for p in sorted((run.path / "trials" / arm / "metrics").glob("epoch_??.json"))
        ]
        for axis, key in zip(axes[0], ("joint_exact_match", "schema_valid_rate"), strict=True):
            if rows:
                axis.plot(
                    [r["epoch"] for r in rows],
                    [r.get("metrics", r)[key] for r in rows],
                    marker="o",
                    label=arm,
                    color=COLORS[arm],
                )
    for axis, title in zip(
        axes[0], ("Validation whole-record exact match", "Validation schema validity"), strict=True
    ):
        axis.set(
            title=title,
            xlabel="Epoch",
            ylabel=(
                "Fraction of development validation subset"
                if result["preset"] == "development"
                else "Fraction of all validation requests"
            ),
            ylim=(0, 1.02),
        )
        if axis.lines:
            axis.legend()
        else:
            _empty(axis)
    return _save(run, fig, "validation")


def plot_task_metrics(run: RunContext) -> Path:
    result = collect_results(run)
    fig, axes = _axes(2)
    keys = (
        "joint_exact_match",
        "schema_valid_rate",
        "intent_accuracy",
        "intent_macro_f1",
        "slot_f1",
    )
    labels = ("Record exact", "Schema valid", "Intent acc.", "Intent macro F1", "Slot micro F1")
    width = 0.18
    for axis, subset, title in zip(
        axes[0],
        (False, True),
        ("Official test — final checkpoints", "Novel-input test subset"),
        strict=True,
    ):
        for index, system in enumerate(SYSTEMS):
            values = result["test"].get(system)
            if values is None:
                continue
            values = values.get("novel_input_subset", {}) if subset else values
            axis.bar(
                [i + (index - 1.5) * width for i in range(len(keys))],
                [values.get(key) if values.get(key) is not None else float("nan") for key in keys],
                width,
                label=system,
                color=COLORS[system],
            )
        axis.set(title=title, ylabel="Fraction / F1", ylim=(0, 1.06))
        axis.set_xticks(range(len(keys)), labels, rotation=20, ha="right")
        if result["test"]:
            axis.legend(fontsize=9)
        else:
            _empty(
                axis,
                "Official test disabled for development preset"
                if result["preset"] == "development"
                else "No completed measurements available",
            )
    return _save(run, fig, "task_metrics")


def plot_memory_runtime(run: RunContext) -> Path:
    result = collect_results(run)
    fig, axes = _axes(3)
    systems = [system for system in SYSTEMS if system in result["benchmarks"]]
    if systems:
        for offset, field, label in (
            (-0.18, "median_seconds", "Median"),
            (0.18, "p95_seconds", "p95"),
        ):
            axes[0, 0].bar(
                [i + offset for i in range(len(systems))],
                [result["benchmarks"][s][field] for s in systems],
                0.36,
                label=label,
            )
        axes[0, 0].set_xticks(range(len(systems)), systems, rotation=20)
        axes[0, 0].legend()
        fields = [
            ("rss_bytes", "Process RSS"),
            ("mps_allocated_bytes", "MPS allocated"),
            ("mps_driver_bytes", "MPS driver"),
            ("cuda_allocated_bytes", "CUDA allocated"),
            ("cuda_reserved_bytes", "CUDA reserved"),
        ]
        present = [
            (key, label)
            for key, label in fields
            if any(key in result["benchmarks"][s]["observed_memory"] for s in systems)
        ]
        width = 0.8 / max(len(present), 1)
        for index, (key, label) in enumerate(present):
            axes[0, 1].bar(
                [i - 0.4 + width * (index + 0.5) for i in range(len(systems))],
                [
                    result["benchmarks"][s]["observed_memory"].get(key, float("nan")) / 2**30
                    for s in systems
                ],
                width,
                label=label,
            )
        axes[0, 1].set_xticks(range(len(systems)), systems, rotation=20)
        axes[0, 1].legend(fontsize=8)
    else:
        _empty(axes[0, 0])
        _empty(axes[0, 1])
    arms = [arm for arm in ARMS if arm in result["training"]]
    if arms:
        axes[0, 2].bar(
            arms,
            [result["training"][arm]["training_seconds"] / 3600 for arm in arms],
            color=[COLORS[a] for a in arms],
        )
    else:
        _empty(axes[0, 2])
    axes[0, 0].set(title="Warm end-to-end request latency", ylabel="Seconds per request")
    axes[0, 1].set(
        title="Maximum observed inference memory", ylabel="GiB — overlapping measures; do not add"
    )
    axes[0, 2].set(title="Training loop time", ylabel="Hours — excludes validation/checkpoint IO")
    return _save(run, fig, "memory_runtime")


def _disagreements(run: Any, available: dict[str, Any]) -> dict[str, Any]:
    predictions = {}
    for system in available:
        rows = read_jsonl(run.path / "evaluations" / system / "test" / "predictions.jsonl")
        predictions[system] = {row["id"]: row for row in rows}
        if len(predictions[system]) != len(rows):
            raise ValueError("Duplicate prediction IDs cannot form a paired comparison")
    pairs, examples = [], []
    for left, right in combinations(predictions, 2):
        if predictions[left].keys() != predictions[right].keys():
            raise ValueError("Test prediction IDs differ between systems")
        counts = {"both_correct": 0, "left_only": 0, "right_only": 0, "both_wrong": 0}
        pair_examples = []
        for identifier in sorted(predictions[left]):
            a, b = predictions[left][identifier], predictions[right][identifier]
            if a["target"] != b["target"]:
                raise ValueError("Paired predictions disagree on ground truth")
            ac, bc = a["joint_exact_match"], b["joint_exact_match"]
            key = (
                "both_correct"
                if ac and bc
                else "left_only"
                if ac
                else "right_only"
                if bc
                else "both_wrong"
            )
            counts[key] += 1
            if ac != bc and len(pair_examples) < 5:
                pair_examples.append(
                    {
                        "id": identifier,
                        "left": left,
                        "right": right,
                        "utterance": a["utterance"],
                        "target": a["target"],
                        "left_prediction": a["prediction_text"],
                        "right_prediction": b["prediction_text"],
                    }
                )
        pairs.append({"left": left, "right": right, **counts})
        examples.extend(pair_examples)
    return {
        "pairs": pairs,
        "examples": examples,
        "selection": "First five correctness disagreements by sorted ID per pair; no favorable selection",
    }


def plot_disagreements(run: RunContext) -> Path:
    result = collect_results(run)
    comparison = _disagreements(run, result["test"])
    fig, axes = _axes()
    axis, pairs = axes[0, 0], comparison["pairs"]
    if pairs:
        positions = list(range(len(pairs)))
        axis.barh(
            [i - 0.18 for i in positions],
            [p["left_only"] for p in pairs],
            0.36,
            label="Left correct / right wrong",
            color="#4e79a7",
        )
        axis.barh(
            [i + 0.18 for i in positions],
            [p["right_only"] for p in pairs],
            0.36,
            label="Right correct / left wrong",
            color="#e08e45",
        )
        axis.set_yticks(positions, [f"{p['left']} vs {p['right']}" for p in pairs])
        axis.legend(fontsize=9)
    else:
        _empty(
            axis,
            "Official test disabled for development preset"
            if result["preset"] == "development"
            else "At least two completed test evaluations are required",
        )
    axis.set(title="Paired whole-record correctness disagreements", xlabel="Official test requests")
    with run.writer_lock():
        path = run.path / "reports" / "disagreements.json"
        atomic_json(path, comparison)
        run.register_artifacts([path], replace=True)
    return _save(run, fig, "disagreements")


def _number(value: Any, digits: int = 4) -> str:
    return "not measured" if value is None else f"{value:.{digits}f}"


def _training_summary_for_report(summary: dict[str, Any]) -> dict[str, Any]:
    """Present legacy aggregate coverage as unknown without altering saved evidence."""
    from .train import _merge_memory

    result = dict(summary)
    for key in ("memory", "training_memory", "validation_memory", "loading_memory", "fit_memory"):
        if key in result:
            result[key] = _merge_memory({}, result[key])
    return result


def _comparison_summary(result: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"quality": {}, "memory": {}, "runtime": {}}
    left, right = result["test"].get("lora"), result["test"].get("qlora")
    if result.get("preset", "full") != "development" and left is not None and right is not None:
        for key in (
            "joint_exact_match",
            "schema_valid_rate",
            "intent_accuracy",
            "intent_macro_f1",
            "slot_f1",
            "answer_token_nll",
        ):
            if left.get(key) is not None and right.get(key) is not None:
                comparison["quality"][key] = right[key] - left[key]
    for phase, section, memory_key in (
        ("training", "training", "memory"),
        ("warm_inference", "benchmarks", "observed_memory"),
    ):
        a, b = result[section].get("lora"), result[section].get("qlora")
        if a is None or b is None:
            continue
        for key in (
            "rss_bytes",
            "mps_allocated_bytes",
            "mps_driver_bytes",
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
        ):
            lvalue, qvalue = a.get(memory_key, {}).get(key), b.get(memory_key, {}).get(key)
            if lvalue is not None and qvalue is not None:
                comparison["memory"][f"{phase}/{key}"] = {
                    "lora": lvalue,
                    "qlora": qvalue,
                    "qlora_minus_lora_bytes": qvalue - lvalue,
                    "relative_reduction": (lvalue - qvalue) / lvalue if lvalue else None,
                }
        for key in (
            ("training_seconds", "wall_seconds")
            if phase == "training"
            else ("median_seconds", "p95_seconds", "generated_tokens_per_second")
        ):
            if a.get(key) is not None and b.get(key) is not None:
                comparison["runtime"][key] = b[key] - a[key]
    return comparison


def _failure_examples(run: Any, available: dict[str, Any]) -> dict[str, Any]:
    examples = []
    for system in available:
        rows = sorted(
            read_jsonl(run.path / "evaluations" / system / "test" / "predictions.jsonl"),
            key=lambda row: row["id"],
        )
        counts: dict[str, int] = {}
        for row in rows:
            for category in row.get("failure_reasons", []):
                if counts.get(category, 0) < 3:
                    examples.append(
                        {
                            "system": system,
                            "category": category,
                            "id": row["id"],
                            "utterance": row["utterance"],
                            "target": row["target"],
                            "prediction_text": row["prediction_text"],
                            "raw_continuation": row.get("raw_continuation"),
                            "generated_token_ids": row.get("generated_token_ids"),
                        }
                    )
                    counts[category] = counts.get(category, 0) + 1
    return {
        "selection": "First three errors by sorted example ID per system/category; an example can occur in several categories",
        "examples": examples,
    }


def _history_lines(result: dict[str, Any]) -> list[str]:
    lines = ["", "## Saved training and validation history", ""]
    for arm, history in result["history"].items():
        lines += [f"### {arm}", ""]
        if not history["training"] and not history["validation"]:
            lines.append("No saved training or epoch-validation history is available.")
            continue
        lines += [
            "These are saved observations; their presence does not imply a completed stage.",
            "",
            "```json",
            json.dumps(history, indent=2, sort_keys=True),
            "```",
            "",
        ]
    return lines


def _write_development_report(run: RunContext, result: dict[str, Any]) -> Path:
    """Keep development evidence separate from official held-out conclusions."""
    expected = result["expected"]
    lines = [
        f"# DEVELOPMENT PROBE — {result['run']}",
        "",
        "The full scientific comparison protocol is NOT COMPLETED. Official test "
        "evaluation is deliberately disabled for this preset; there is no held-out "
        "test quality claim and no overall quality winner.",
        "",
        "Development artifact status: "
        + ("requested probe evidence present" if result["preset_complete"] else "INCOMPLETE")
        + ". This status never certifies the full five-epoch experiment.",
        "",
        f"Quality scope: {result['quality_scope']}.",
        "",
        f"Configured budget: {expected['epochs_per_arm']} epoch(s) per arm; "
        f"training example limit={expected['train_example_limit']}; "
        f"validation example limit={expected['validation_example_limit']}. "
        f"Benchmark: {expected['benchmark_requests']} requests, "
        f"{expected['benchmark_warmup']} excluded warmups, "
        f"{expected['benchmark_repetitions']} measured repetition(s), batch size one.",
        "",
        "These subset runs check execution, memory behavior, and observed runtime. "
        "They cannot establish full-data convergence, general task quality, or "
        "long-run runtime and memory guarantees.",
        "",
        "## Frozen configuration and provenance",
        "",
        "```json",
        json.dumps(run.config.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
        f"Immutable model revision: `{run.config.model_revision}`. "
        f"Backend: `{run.environment['device']}`; "
        f"compute dtype: `{run.environment['compute_dtype']}`. "
        "Run identity, dependency versions, and source hashes remain in "
        "`../identity.json`, `../environment.json`, and `../source_snapshot/`.",
        "",
        "## Saved development validation metrics",
        "",
    ]
    if result["validation"]:
        lines += [
            "Subset validation observations only; these are not official test results.",
            "",
            "```json",
            json.dumps(result["validation"], indent=2, sort_keys=True),
            "```",
        ]
    else:
        lines.append("No completed development validation metrics are available.")
    lines.extend(_history_lines(result))
    lines += ["", "## Saved training, runtime, and memory measurements", ""]
    if result["training"] or result["benchmarks"]:
        lines += [
            "```json",
            json.dumps(
                {
                    "training": {
                        arm: _training_summary_for_report(value)
                        for arm, value in result["training"].items()
                    },
                    "benchmarks": result["benchmarks"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    else:
        lines.append("No completed training, timing, or memory measurements are available.")
    lines += [
        "",
        "RSS and device allocation overlap and must not be added. Memory is "
        "maximum observed sampling, not a guaranteed peak. GPU activity and system "
        "swap are system-wide and cannot be attributed entirely to this process.",
        "",
        "## Deliberate omissions and incomplete work",
        "",
    ]
    lines.extend(f"- Deliberately omitted: {item}." for item in result["deliberately_omitted"])
    lines.extend(f"- Pending development evidence: {item}." for item in result["missing"])
    lines.extend(
        f"- {name}: {value.get('status')}; {value.get('error_type', '')}: {value.get('error', '')}"
        for name, value in result["failed_stages"].items()
    )
    lines += [
        "",
        "The full preset still requires complete official partitions, five training "
        "epochs for both arms, frozen final checkpoints, official test evaluation, "
        "and the full benchmark protocol. Development artifacts cannot satisfy "
        "those scientific-comparison gates.",
        "",
        "## Artifacts",
        "",
        "Training/validation history: `trials/{arm}/metrics/`. Development validation: "
        "`evaluations/{system}/validation/`. Benchmarks: `benchmarks/{system}/`. "
        "This report reads saved artifacts only and never starts training, evaluation, "
        "or benchmarks. Figures require explicit plotting calls.",
        "",
    ]
    path = run.path / "reports" / "results.md"
    comparison_path = run.path / "reports" / "paired_comparison.json"
    with run.writer_lock():
        atomic_text(path, "\n".join(lines))
        atomic_json(
            comparison_path,
            {
                "preset": "development",
                "quality_scope": result["quality_scope"],
                "scientific_comparison_complete": False,
                **_comparison_summary(result),
            },
        )
        run.register_artifacts([path, comparison_path], replace=True)
    return path


def write_report(run: RunContext) -> Path:
    """Write an honest report from existing artifacts; never create measurements."""
    result = collect_results(run)
    if result["preset"] == "development":
        return _write_development_report(run, result)
    lines = [
        f"# Structured request adaptation — {result['run']}",
        "",
        f"Status: {'COMPLETE measured comparison' if result['complete'] else 'INCOMPLETE — evidence missing; no overall winner claim'}.",
        "",
        f"One paired seed ({run.config.seed}), official MASSIVE {run.config.locale} splits, {run.config.model_id}. Primary adapted-model results use the final checkpoint after {run.config.epochs} full epochs. Validation-selected best checkpoints are supplementary and are not silently substituted.",
        "",
        "## Frozen configuration and provenance",
        "",
        "```json",
        json.dumps(run.config.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
        f"Immutable model revision: `{run.config.model_revision}`. Execution backend: `{run.environment['device']}`; compute dtype: `{run.environment['compute_dtype']}`.",
        "",
        "The exact OS, Python, backend, package versions and runtime flags are recorded in `../environment.json`; source and pinned dependency files are copied into `../source_snapshot/` with hashes. `../identity.json`, `../manifest.json` and `../evaluations/frozen_protocol.json` identify this run and its registered artifacts.",
        "",
        "## Protocol and interpretation",
        "",
        "LoRA uses the unquantized frozen backbone; QLoRA uses the frozen quantized backbone. Compare each adapted arm to its own unadapted base, and compare base to quantized_base to expose the quantization effect. A direct LoRA–QLoRA difference alone does not isolate adaptation from quantization.",
        "",
        "Whole-record exact match requires valid JSON, the known schema, correct intent and exact case-sensitive slot-value/type multisets. Slot order is ignored, duplicate occurrences retained. Invalid generations stay in denominators; no JSON repair, retry or constrained decoding is used. The novel-input subset excludes normalized utterances observed in either training or validation.",
        "",
        "## Official test metrics",
        "",
        "| System | N | Record exact | JSON valid | Schema valid | Intent acc. | Intent macro F1 | Slot P | Slot R | Slot F1 | Answer NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_keys = (
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
    for system in SYSTEMS:
        value = result["test"].get(system, {})
        lines.append(
            f"| {system} | {value.get('example_count', '—')} | "
            + " | ".join(_number(value.get(key)) for key in metric_keys)
            + " |"
        )
    lines += ["", "## Novel-input subset and failures", ""]
    lines += [
        "| System | Novel N | Record exact | JSON valid | Schema valid | Intent acc. | Intent macro F1 | Slot P | Slot R | Slot F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        value = result["test"].get(system, {}).get("novel_input_subset", {})
        lines.append(
            f"| {system} | {value.get('example_count', '—')} | "
            + " | ".join(_number(value.get(key)) for key in metric_keys[:-1])
            + " |"
        )
    lines += [
        "",
        "Full-test failure counts below use the entire test partition; categories can overlap.",
        "",
    ]
    for system, value in result["test"].items():
        lines += [
            f"- {system}: full-test n={value.get('example_count', 0)}; failure categories={value.get('failure_counts', {})}."
        ]
    if not result["test"]:
        lines.append("No completed test predictions are available. No task quality is claimed.")
    lines += [
        "",
        "## Runtime and memory",
        "",
        f"Benchmark: the same seeded {result['expected']['benchmark_requests']} validation requests for all systems, {result['expected']['benchmark_warmup']} excluded warmups, {result['expected']['benchmark_repetitions']} repetitions, batch size one. Median/p95 include prompt tokenization, device transfer, generation and decoding, with device synchronization. Correctness is recorded for every request in every repetition. Generated-token throughput includes EOS tokens if emitted; model loading and scoring are outside the latency interval.",
        "",
    ]
    for system, value in result["benchmarks"].items():
        accuracies = [row["joint_exact_match"] for row in value["metrics_by_repetition"]]
        lines.append(
            f"- {system}: median={_number(value.get('median_seconds'))} s, p95={_number(value.get('p95_seconds'))} s, generated tokens/s={_number(value.get('generated_tokens_per_second'))}; per-repetition record accuracy={accuracies}; memory={value.get('observed_memory', {})}."
        )
    if not result["benchmarks"]:
        lines.append("No completed timing or memory measurements are available.")
    for arm, value in result["training"].items():
        value = _training_summary_for_report(value)
        lines.append(
            f"- {arm} training: epochs={value.get('epochs_completed')}, updates={value.get('optimizer_updates')}, target tokens={value.get('target_tokens_seen')}, training loop seconds={value.get('training_seconds')}, wall seconds={value.get('wall_seconds')}; memory={value.get('memory', {})}."
        )
    lines += [
        "",
        "RSS, MPS allocated/driver or CUDA allocated/reserved measures are overlapping and must not be added. Memory telemetry reports maximum observed samples, not an exact peak. System swap activity is machine-wide and is disclosed without attributing it to this process. Cold/model-loading latency is not measured by the warm benchmark.",
        "For legacy training aggregates, null coverage fields mean unavailable cumulative sampler metadata. Memory maxima are preserved; last-interval counts and durations are not treated as whole-training coverage.",
        "",
        "## Paired comparisons",
        "",
    ]
    comparison = _comparison_summary(result)
    if comparison["quality"]:
        lines.append(f"QLoRA minus LoRA metric deltas: {comparison['quality']}.")
        delta = comparison["quality"].get("joint_exact_match")
        if delta is not None:
            lines.append(
                f"Primary whole-record exact-match difference: {delta * 100:+.3f} percentage points. "
                + (
                    "There is no observed quality drop in this paired run."
                    if delta >= 0
                    else "The paired run shows an observed quality drop for QLoRA."
                )
                + " No non-inferiority margin or multi-seed uncertainty was registered, so this does not establish equivalent quality or statistical superiority."
            )
    else:
        lines.append(
            "Quality-preservation assessment is pending both final official-test evaluations."
        )
    if comparison["memory"]:
        for key, value in comparison["memory"].items():
            outcome = (
                "observed reduction supports the expected memory benefit for this measure"
                if value["qlora_minus_lora_bytes"] < 0
                else "no observed reduction; the expected memory benefit is not supported for this measure"
            )
            lines.append(
                f"- {key}: QLoRA−LoRA={value['qlora_minus_lora_bytes'] / 2**30:+.4f} GiB; relative reduction={_number(value['relative_reduction'])}; {outcome}."
            )
        lines.append(
            "Whole-fit samples include training and validation; loading, validation NLL and generation phase samples are stored separately. Interpret mixed memory measures separately and qualify any result with recorded swapping/sampling errors."
        )
    else:
        lines.append(
            "The expected memory reduction remains unmeasured; paired training/inference memory is pending."
        )
    if comparison["runtime"]:
        lines.append(
            f"QLoRA minus LoRA runtime/throughput deltas (units match the source metric): {comparison['runtime']}."
        )
    lines.append("")
    comparisons = _disagreements(run, result["test"])
    for pair in comparisons["pairs"]:
        lines.append(
            f"- {pair['left']} vs {pair['right']}: left-only correct={pair['left_only']}, right-only correct={pair['right_only']}, both correct={pair['both_correct']}, both wrong={pair['both_wrong']}."
        )
    if not comparisons["pairs"]:
        lines.append("At least two completed test evaluations are required for paired comparisons.")
    errors = _failure_examples(run, result["test"])
    lines += [
        "",
        "## Representative error categories",
        "",
        errors["selection"]
        + ". Full utterances, gold records, raw output and token IDs are in `error_examples.json`.",
        "",
    ]
    for row in errors["examples"]:
        lines.append(f"- {row['system']} / {row['category']}: example `{row['id']}`.")
    if not errors["examples"]:
        lines.append("No error examples are available from the completed evaluations.")
    lines += [
        "",
        "## Limits and incomplete work",
        "",
        "This is one paired seed on one task/model/hardware/version combination, without multi-seed uncertainty estimates or a universal superiority claim. The official test may contain normalized duplicates of train/dev inputs; both full and novel-input results are reported. No causal claim is made from incomparable training losses or unpaired prompts. Schema/intent/slot metrics assess this structured task, not general language-model ability.",
        "",
    ]
    lines.extend(f"- Pending: {item}." for item in result["missing"])
    lines.extend(
        f"- {name}: {value.get('status')}; {value.get('error_type', '')}: {value.get('error', '')}"
        for name, value in result["failed_stages"].items()
    )
    if result["complete"]:
        lines.append(
            "All required stored evidence is present. Interpret observed differences within the limits above."
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "Raw per-request predictions and token IDs: `evaluations/{system}/test/predictions.jsonl`. Benchmarks: `benchmarks/{system}/requests.jsonl`. Training/validation curves: `trials/{arm}/metrics/`. Figures are rendered only by explicit plot calls; this report does not fit, evaluate or benchmark any model.",
        "",
    ]
    path = run.path / "reports" / "results.md"
    with run.writer_lock():
        atomic_text(path, "\n".join(lines))
        error_path = run.path / "reports" / "error_examples.json"
        comparison_path = run.path / "reports" / "paired_comparison.json"
        atomic_json(error_path, errors)
        atomic_json(comparison_path, comparison)
        run.register_artifacts([path, error_path, comparison_path], replace=True)
    return path


def run_summary(run: RunContext) -> dict[str, Any]:
    result = collect_results(run)
    result["run_path"] = str(run.path.resolve())
    result["artifact_paths"] = {
        name: str((run.path / name).resolve())
        for name in (
            "config.json",
            "environment.json",
            "identity.json",
            "manifest.json",
            "evaluations",
            "benchmarks",
            "figures",
            "reports",
            "trials",
            "logs",
        )
    }
    return result
