"""Explicit, reproducible evaluation and single-model inference.

Importing this module does not load a model, evaluate examples, or access the
network. Final test evaluation is unavailable until both five-epoch arms and
their data/checkpoints have been frozen.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .config import ARMS, SYSTEMS
from .io import atomic_json, canonical_hash, read_json, sha256_file, write_jsonl

_INFERENCE_ACTIVE = False
if TYPE_CHECKING:
    from .runs import RunContext


def assert_no_active_predictor() -> None:
    if _INFERENCE_ACTIVE:
        raise RuntimeError("Close the active predictor before loading another model")


def _stored_outputs_complete(run: RunContext, paths: list[Any], stage: str) -> bool:
    """Registered output hashes are authoritative after a stage-status crash.

    Atomic registration commits the complete output set. Unregistered files are
    partial work and may be replaced by this explicit action. A mixed registered
    set is an integrity error, not permission to overwrite frozen evidence.
    """
    manifest = read_json(run.path / "manifest.json")["artifacts"]
    relatives = [str(path.relative_to(run.path)) for path in paths]
    present = [relative in manifest for relative in relatives]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(
            f"Incomplete registered output set for {stage}; preserve this run for inspection"
        )
    run.verify_artifacts(prefixes=relatives)
    stage_path = run.path / "stages" / f"{stage}.json"
    if not stage_path.exists() or read_json(stage_path).get("status") != "complete":
        with run.stage(stage):
            pass  # The verified artifacts were committed before the interrupted status update.
    return True


def _check_system(system: str) -> None:
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose from {SYSTEMS}")


def _require_test_eligible(run: Any) -> None:
    if getattr(run.config, "preset", "full") != "full":
        raise RuntimeError(
            "Official test evaluation is disabled for development runs. Create a fresh full comparison."
        )


def _decode_continuation(tokenizer: Any, token_ids: list[int]) -> dict[str, Any]:
    """Remove only terminal EOS/PAD; unexpected special tokens remain failures."""
    terminal_ids = {
        value for value in (tokenizer.eos_token_id, tokenizer.pad_token_id) if value is not None
    }
    stop = len(token_ids)
    while stop and token_ids[stop - 1] in terminal_ids:
        stop -= 1
    return {
        "prediction_text": tokenizer.decode(
            token_ids[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "raw_continuation": tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "terminal_token_ids": token_ids[stop:],
    }


def answer_token_nll(logits: Any, labels: Any) -> tuple[float, int]:
    """Return summed NLL and count, explicitly shifting causal-LM targets.

    Label -100 masks prompt/padding tokens. Chunking the vocabulary-sized float32
    cross-entropy input bounds the temporary allocation on the 16 GB machine.
    Call under torch.inference_mode(); this helper is for measurements, not loss
    backpropagation.
    """
    from .losses import token_nll

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, sequence, vocab] and matching labels")
    return token_nll(logits[:, :-1, :], labels[:, 1:])


def _generate_batch(
    run: Any, model: Any, tokenizer: Any, prompts: list[list[int]]
) -> dict[str, Any]:
    """Greedy left-padded decoding, with one synchronized time for the whole batch.

    Timing excludes input transfer and output decoding. It is not a per-request
    latency measurement. Only padding appended after a row's first EOS is removed
    from its token accounting; repeated emitted tokens remain intact.
    """
    import torch

    from .runtime import autocast_context, synchronize

    if not prompts or any(
        not prompt or len(prompt) + run.config.max_new_tokens > run.config.max_sequence_length
        for prompt in prompts
    ):
        raise ValueError("Prompt exceeds the frozen context budget; no silent truncation")
    if tokenizer.pad_token_id is None:
        raise ValueError("Batched generation requires the saved tokenizer's pad_token_id")
    width = max(map(len, prompts))
    ids = torch.full(
        (len(prompts), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=run.environment["device"],
    )
    mask = torch.zeros_like(ids)
    for index, prompt in enumerate(prompts):
        ids[index, -len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=ids.device)
        mask[index, -len(prompt) :] = 1
    synchronize(run.environment["device"])
    start = time.perf_counter()
    with torch.inference_mode(), autocast_context(run.environment):
        output = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=run.config.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    synchronize(run.environment["device"])
    seconds = time.perf_counter() - start
    if output.ndim != 2 or output.shape[0] != len(prompts) or output.shape[1] < width:
        raise RuntimeError("Generation returned an unexpected batch/sequence shape")
    continuations = output[:, width:].detach().cpu().tolist()
    predictions = []
    for prompt, generated in zip(prompts, continuations, strict=True):
        padding_tokens = 0
        ended_with_eos = tokenizer.eos_token_id in generated
        if ended_with_eos:
            stop = generated.index(tokenizer.eos_token_id) + 1
            if any(token != tokenizer.pad_token_id for token in generated[stop:]):
                raise RuntimeError("Generation emitted non-padding tokens after EOS")
            padding_tokens = len(generated) - stop
            generated = generated[:stop]
        predictions.append(
            {
                **_decode_continuation(tokenizer, generated),
                "generated_token_ids": generated,
                "generated_tokens": len(generated),
                "prompt_tokens": len(prompt),
                "post_eos_padding_tokens": padding_tokens,
                "hit_generation_limit": (
                    not ended_with_eos and len(generated) >= run.config.max_new_tokens
                ),
            }
        )
    return {
        "predictions": predictions,
        "batch_size": len(prompts),
        "padded_prompt_tokens": width,
        "generation_seconds": seconds,
    }


def _generate(run: Any, model: Any, tokenizer: Any, prompt_ids: list[int]) -> dict[str, Any]:
    """Single-request path retained for the latency benchmark and predictor."""
    result = _generate_batch(run, model, tokenizer, [prompt_ids])
    return {**result["predictions"][0], "generation_seconds": result["generation_seconds"]}


def evaluate_loaded(run: RunContext, model: Any, tokenizer: Any, split: str) -> dict[str, Any]:
    """Evaluate an already loaded model; no stages, writes, or fitting.

    This low-level helper is used by epoch validation. Public held-out evaluation
    must enter through evaluate_model, which enforces the freeze first.
    Both NLL and generation use the frozen evaluation batch size without changing
    example order. Timing records describe complete batches, not request latency.
    """
    import torch
    from tqdm.auto import tqdm

    from .checkpoints import capture_rng, restore_rng
    from .data import load_examples, load_ontology
    from .losses import collate_answer_rows, forward_answer_nll_per_example
    from .metrics import evaluate_predictions, score_prediction
    from .runtime import MemorySampler, autocast_context, seed_everything

    if split not in ("validation", "test"):
        raise ValueError("Evaluation accepts validation or test only")
    if split == "test":
        _require_test_eligible(run)
        _require_frozen(run)
    ontology = load_ontology(run)
    examples = load_examples(run, split, tokenized=True)
    expected = run.config.expected_split_counts[1 if split == "validation" else 2]
    if split == "validation" and getattr(run.config, "preset", "full") == "development":
        expected = run.config.validation_example_limit
    if len(examples) != expected:
        raise ValueError(f"Expected {expected} registered {split} examples, found {len(examples)}")
    batch_size = getattr(run.config, "evaluation_batch_size", 1)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("evaluation_batch_size must be a positive integer")
    original_rng = capture_rng()
    original_mode = model.training
    original_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    original_cache = model.config.use_cache
    predictions: list[dict[str, Any]] = []
    numerator, tokens = 0.0, 0
    row_nll: dict[str, tuple[float, int]] = {}
    generation_batches: list[dict[str, Any]] = []
    phase_memory: dict[str, Any] = {}
    try:
        seed_everything(run.config.seed)
        model.eval()
        if original_checkpointing:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        with MemorySampler(
            run.environment["device"], run.config.telemetry_interval_seconds
        ) as sampler:
            for start in tqdm(
                range(0, len(examples), batch_size),
                desc=f"{len(examples)} {split}: answer NLL",
                unit="batch",
            ):
                rows = examples[start : start + batch_size]
                batch, labels = collate_answer_rows(
                    rows, run.environment["device"], tokenizer.pad_token_id
                )
                with torch.inference_mode(), autocast_context(run.environment):
                    row_losses = forward_answer_nll_per_example(
                        model,
                        batch,
                        labels,
                        answer_only_logits=run.config.answer_only_logits,
                    )
                del batch, labels
                for row, (nll, count) in zip(rows, row_losses, strict=True):
                    numerator += nll
                    tokens += count
                    row_nll[row["id"]] = nll, count
        phase_memory["validation_nll" if split == "validation" else "test_nll"] = sampler.summary()
        with MemorySampler(
            run.environment["device"], run.config.telemetry_interval_seconds
        ) as sampler:
            for start in tqdm(
                range(0, len(examples), batch_size),
                desc=f"{len(examples)} {split}: greedy generation",
                unit="batch",
            ):
                rows = examples[start : start + batch_size]
                result = _generate_batch(run, model, tokenizer, [row["prompt_ids"] for row in rows])
                batch_index = len(generation_batches)
                generation_batches.append(
                    {
                        "batch_index": batch_index,
                        "example_ids": [row["id"] for row in rows],
                        **{key: value for key, value in result.items() if key != "predictions"},
                        "generated_tokens": sum(
                            p["generated_tokens"] for p in result["predictions"]
                        ),
                    }
                )
                for row, prediction in zip(rows, result["predictions"], strict=True):
                    prediction.update(
                        {key: row[key] for key in ("id", "utterance", "target", "novel_input")}
                    )
                    prediction.update(
                        score_prediction(
                            prediction["prediction_text"],
                            row["target"],
                            ontology["intents"],
                            ontology["slots"],
                        )
                    )
                    nll, count = row_nll[row["id"]]
                    prediction.update(
                        answer_nll_sum=nll, answer_tokens=count, generation_batch_index=batch_index
                    )
                    predictions.append(prediction)
        phase_memory["generation"] = sampler.summary()
    finally:
        model.config.use_cache = original_cache
        if original_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train(original_mode)
        restore_rng(original_rng)
    metrics = evaluate_predictions(predictions, ontology["intents"], ontology["slots"])
    metrics.update(
        answer_token_nll=numerator / tokens,
        answer_nll_sum=numerator,
        answer_tokens=tokens,
        split=split,
        phase_memory=phase_memory,
        evaluation_batch_size=batch_size,
        generation_batches=generation_batches,
        generation_seconds=sum(batch["generation_seconds"] for batch in generation_batches),
        generation_timing_boundary=(
            "Synchronized model.generate per batch; excludes input transfer/decoding/scoring. "
            "Batch times are not per-request latency; use the batch-one benchmark for latency."
        ),
    )
    novel = [row for row in predictions if row["novel_input"]]
    novel_tokens = sum(row["answer_tokens"] for row in novel)
    novel_nll = sum(row["answer_nll_sum"] for row in novel)
    metrics["novel_input_subset"].update(
        answer_nll_sum=novel_nll,
        answer_tokens=novel_tokens,
        answer_token_nll=novel_nll / novel_tokens if novel_tokens else None,
    )
    return {"metrics": metrics, "predictions": predictions}


def _freeze_identity(run: Any) -> dict[str, Any]:
    run.verify_identity()
    run.require_stage("tokenization")
    run.verify_artifacts()
    if getattr(run.config, "preset", "full") == "full" and run.config.epochs != 5:
        raise ValueError("The final protocol requires exactly five epochs in both arms")
    files: dict[str, str] = {}
    for path in sorted((run.path / "preparation").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(run.path))] = sha256_file(path)
    if not files:
        raise RuntimeError("Prepared data artifacts are missing")
    training = {}
    for arm in ARMS:
        run.require_stage(f"train_{arm}")
        summary_path = run.path / "trials" / arm / "training_summary.json"
        summary = read_json(summary_path)
        expected_epochs = run.config.epochs
        if (
            summary.get("completed") is not True
            or summary.get("epochs_completed") != expected_epochs
        ):
            raise RuntimeError(f"{arm} has not completed all {expected_epochs} configured epochs")
        from .checkpoints import resolve_checkpoint

        checkpoint = resolve_checkpoint(run, arm, "final")
        checkpoint_files = [path for path in sorted(checkpoint.rglob("*")) if path.is_file()]
        if not checkpoint_files:
            raise RuntimeError(f"Final checkpoint missing for {arm}")
        for path in [summary_path, *checkpoint_files]:
            files[str(path.relative_to(run.path))] = sha256_file(path)
        training[arm] = {
            key: summary.get(key)
            for key in (
                "epochs_completed",
                "optimizer_updates",
                "target_tokens_seen",
                "final_checkpoint",
            )
        }
    if not training["lora"]["target_tokens_seen"] or not training["lora"]["optimizer_updates"]:
        raise RuntimeError("Training summaries lack a positive token/update budget")
    if training["lora"]["target_tokens_seen"] != training["qlora"]["target_tokens_seen"]:
        raise RuntimeError("LoRA and QLoRA supervised-token budgets differ")
    if training["lora"]["optimizer_updates"] != training["qlora"]["optimizer_updates"]:
        raise RuntimeError("LoRA and QLoRA optimizer-update budgets differ")
    return {
        "schema_version": 1,
        "preset": getattr(run.config, "preset", "full"),
        "protocol_hash": canonical_hash(run.config.to_dict()),
        "model_revision": run.config.model_revision,
        "primary_checkpoint": "final",
        "generation": {
            "max_new_tokens": run.config.max_new_tokens,
            "evaluation_batch_size": getattr(run.config, "evaluation_batch_size", 1),
            "padding": "Left-padded prompts with explicit attention masks",
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "terminal_token_policy": "Saved tokenizer eos_token_id and pad_token_id",
            "decoding": "Strip terminal EOS/PAD only; retain other special tokens; no text repair",
            "inherited_configuration": (
                read_json(run.path / "inputs" / "base_model" / "generation_config.json")
                if (run.path / "inputs" / "base_model" / "generation_config.json").is_file()
                else "Transformers defaults from the frozen dependency version"
            ),
        },
        "systems": list(SYSTEMS),
        "artifacts": files,
        "training": training,
    }


def freeze_evaluation(run: RunContext) -> dict[str, Any]:
    """Verify both completed arms, then lock test identity before any test use."""
    path = run.path / "evaluations" / "frozen_protocol.json"
    with run.writer_lock():
        identity = _freeze_identity(run)
        if path.exists():
            if read_json(path) != identity:
                raise RuntimeError("Frozen evaluation identity no longer matches the run")
            if _stored_outputs_complete(run, [path], "freeze_evaluation"):
                return identity
        with run.stage("freeze_evaluation"):
            atomic_json(path, identity)
            run.register_artifacts([path])
    return identity


def _require_frozen(run: Any) -> dict[str, Any]:
    path = run.path / "evaluations" / "frozen_protocol.json"
    if not path.exists():
        raise RuntimeError("Call freeze_evaluation after both complete training arms")
    with run.writer_lock():
        if not _stored_outputs_complete(run, [path], "freeze_evaluation"):
            raise RuntimeError(
                "Call freeze_evaluation to finish registering the interrupted freeze"
            )
    identity = read_json(path)
    run.verify_identity()
    if identity["protocol_hash"] != canonical_hash(run.config.to_dict()):
        raise RuntimeError("Configuration changed after evaluation freeze")
    for relative, expected in identity["artifacts"].items():
        path = run.path / relative
        path.resolve().relative_to(run.path.resolve())
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen artifact changed: {relative}")
    return identity


@contextmanager
def _loaded(
    run: RunContext, system: str, phase_memory: dict[str, Any] | None = None
) -> Iterator[tuple[Any, Any]]:
    from .models import load_system
    from .runtime import MemorySampler, release_memory

    global _INFERENCE_ACTIVE
    assert_no_active_predictor()
    _INFERENCE_ACTIVE = True
    model = tokenizer = None
    try:
        with MemorySampler(
            run.environment["device"], run.config.telemetry_interval_seconds
        ) as sampler:
            model, tokenizer = load_system(run, system, checkpoint="final", training=False)
        if phase_memory is not None:
            phase_memory["loading"] = sampler.summary()
        model.eval()
        yield model, tokenizer
    finally:
        del model, tokenizer
        _INFERENCE_ACTIVE = False
        release_memory(run.environment["device"])


def evaluate_model(run: RunContext, system: str, split: str = "test") -> dict[str, Any]:
    """Explicit full-split action; return saved metrics and artifact paths only."""
    _check_system(system)
    if split not in ("validation", "test"):
        raise ValueError("Evaluation accepts validation or test only")
    if split == "test":
        _require_test_eligible(run)
    run.verify_identity()
    run.require_stage("tokenization")
    if split == "test":
        _require_frozen(run)
    directory = run.path / "evaluations" / system / split
    predictions_path, metrics_path = directory / "predictions.jsonl", directory / "metrics.json"
    with run.writer_lock():
        if _stored_outputs_complete(
            run, [predictions_path, metrics_path], f"evaluate_{system}_{split}"
        ):
            return {
                "system": system,
                "split": split,
                "metrics": read_json(metrics_path),
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
            }
    with run.writer_lock(), run.stage(f"evaluate_{system}_{split}"):
        phases: dict[str, Any] = {}
        with _loaded(run, system, phases) as (model, tokenizer):
            try:
                result = evaluate_loaded(run, model, tokenizer, split)
            finally:
                del model, tokenizer
        result["metrics"]["phase_memory"].update(phases)
        write_jsonl(predictions_path, result["predictions"])
        atomic_json(metrics_path, result["metrics"])
        run.register_artifacts([predictions_path, metrics_path])
    return {
        "system": system,
        "split": split,
        "metrics": result["metrics"],
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot summarize empty timings")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def benchmark_model(run: RunContext, system: str) -> dict[str, Any]:
    """Benchmark the frozen preset's validation sample, warmups, and repetitions.

    The full protocol requires 100 requests, five warmups, and three repetitions.
    Development uses its explicit, smaller configured budget.

    End-to-end latency includes prompt tokenization, device transfer, generation
    and decoding; correctness scoring is excluded. RSS and backend memory remain
    separately labeled; neither is an assertion of exact peak accelerator use.
    """
    from tqdm.auto import tqdm

    from .data import chat_token_ids, load_examples, load_ontology, prompt_messages
    from .metrics import evaluate_predictions, score_prediction
    from .runtime import MemorySampler, seed_everything, synchronize

    _check_system(system)
    _require_frozen(run)
    request_count = run.config.benchmark_requests
    warmup_count = run.config.benchmark_warmup
    repetitions = run.config.benchmark_repetitions
    if run.config.preset == "full" and (request_count, warmup_count, repetitions) != (100, 5, 3):
        raise ValueError("Benchmark protocol is fixed at 100 requests, 5 warmups and 3 repetitions")
    if (
        any(
            type(value) is not int or value <= 0
            for value in (request_count, warmup_count, repetitions)
        )
        or warmup_count > request_count
    ):
        raise ValueError("Benchmark counts must be positive and warmups cannot exceed requests")
    examples = load_examples(run, "validation", tokenized=True)
    if request_count > len(examples):
        raise ValueError("Benchmark requests exceed the available validation examples")
    indices = random.Random(run.config.seed).sample(range(len(examples)), request_count)
    rows = [examples[index] for index in indices]
    selection = {
        "split": "validation",
        "seed": run.config.seed,
        "preset": run.config.preset,
        "requests_per_repetition": request_count,
        "warmup_requests": warmup_count,
        "repetitions": repetitions,
        "example_ids": [row["id"] for row in rows],
        "prompt_hash": canonical_hash([row["prompt_ids"] for row in rows]),
    }
    ontology = load_ontology(run)

    def request(model: Any, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
        synchronize(run.environment["device"])
        start = time.perf_counter()
        messages = prompt_messages(row["utterance"], ontology["intents"], ontology["slots"])
        prompt = chat_token_ids(tokenizer, messages, add_generation_prompt=True)
        prediction = _generate(run, model, tokenizer, prompt)
        synchronize(run.environment["device"])
        prediction["end_to_end_seconds"] = time.perf_counter() - start
        if prompt != row["prompt_ids"]:
            raise RuntimeError("Benchmark prompt differs from the frozen prepared prompt")
        return prediction

    records = []
    directory = run.path / "benchmarks" / system
    selection_path = run.path / "benchmarks" / "requests.json"
    predictions_path, summary_path = directory / "requests.jsonl", directory / "summary.json"
    with run.writer_lock():
        if _stored_outputs_complete(run, [predictions_path, summary_path], f"benchmark_{system}"):
            if (
                not _stored_outputs_complete(run, [selection_path], f"benchmark_{system}")
                or read_json(selection_path) != selection
            ):
                raise RuntimeError(
                    "Saved benchmark selection was not registered or no longer matches"
                )
            return read_json(summary_path)
    with run.writer_lock(), run.stage(f"benchmark_{system}"):
        if selection_path.exists() and read_json(selection_path) != selection:
            raise RuntimeError("Benchmark request identity changed")
        atomic_json(selection_path, selection)
        seed_everything(run.config.seed)
        phases: dict[str, Any] = {}
        with _loaded(run, system, phases) as (model, tokenizer):
            with MemorySampler(
                run.environment["device"], run.config.telemetry_interval_seconds
            ) as sampler:
                for row in rows[:warmup_count]:
                    request(model, tokenizer, row)
            phases["warmup"] = sampler.summary()
            with MemorySampler(
                run.environment["device"], run.config.telemetry_interval_seconds
            ) as sampler:
                for repetition in range(repetitions):
                    for row in tqdm(
                        rows,
                        desc=f"{system} benchmark {repetition + 1}/{repetitions}",
                        unit="request",
                    ):
                        record = request(model, tokenizer, row)
                        record.update(
                            {key: row[key] for key in ("id", "utterance", "target", "novel_input")}
                        )
                        record["repetition"] = repetition + 1
                        record.update(
                            score_prediction(
                                record["prediction_text"],
                                row["target"],
                                ontology["intents"],
                                ontology["slots"],
                            )
                        )
                        records.append(record)
            memory = sampler.summary()
            phases["generation"] = memory
            del model, tokenizer
        timings = [row["end_to_end_seconds"] for row in records]
        generated = sum(row["generated_tokens"] for row in records)
        elapsed = sum(timings)
        summary = {
            "system": system,
            "split": "validation",
            "preset": run.config.preset,
            "development_only": run.config.preset == "development",
            "requests_per_repetition": request_count,
            "repetitions": repetitions,
            "warmup_requests": warmup_count,
            "measured_requests": len(records),
            "batch_size": 1,
            "latency_boundary": "synchronized tokenize/device-transfer/generate/decode; excludes scoring and load",
            "median_seconds": statistics.median(timings),
            "p95_seconds": _percentile(timings, 0.95),
            "end_to_end_seconds": elapsed,
            "generation_seconds": sum(row["generation_seconds"] for row in records),
            "generated_tokens": generated,
            "generated_tokens_per_second": generated / elapsed,
            "requests_per_second": len(records) / elapsed,
            "observed_memory": memory,
            "phase_memory": phases,
            "metrics_by_repetition": [
                evaluate_predictions(
                    [row for row in records if row["repetition"] == rep],
                    ontology["intents"],
                    ontology["slots"],
                )
                for rep in range(1, repetitions + 1)
            ],
        }
        write_jsonl(predictions_path, records)
        atomic_json(summary_path, summary)
        run.register_artifacts([selection_path, predictions_path, summary_path])
    return summary


class Predictor:
    """One managed model. Close it before constructing another predictor."""

    def __init__(self, run: RunContext, system: str):
        from .data import load_ontology
        from .models import load_system

        global _INFERENCE_ACTIVE
        _check_system(system)
        if _INFERENCE_ACTIVE:
            raise RuntimeError("Close the active predictor before loading another model")
        _require_frozen(run)
        self.run, self.system = run, system
        self.ontology = load_ontology(run)
        _INFERENCE_ACTIVE = True
        try:
            self.model, self.tokenizer = load_system(
                run, system, checkpoint="final", training=False
            )
            self.model.eval()
        except BaseException:
            _INFERENCE_ACTIVE = False
            raise
        self.closed = False
        run._active_predictor = self

    def predict(self, text: str) -> dict[str, Any]:
        from .data import chat_token_ids, prompt_messages
        from .metrics import validate_prediction

        if self.closed:
            raise RuntimeError("Predictor is closed")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be non-empty text")
        from .runtime import synchronize

        synchronize(self.run.environment["device"])
        start = time.perf_counter()
        messages = prompt_messages(text, self.ontology["intents"], self.ontology["slots"])
        prompt = chat_token_ids(self.tokenizer, messages, add_generation_prompt=True)
        result = _generate(self.run, self.model, self.tokenizer, prompt)
        synchronize(self.run.environment["device"])
        result["end_to_end_seconds"] = time.perf_counter() - start
        # There is no ground truth for an interactive request. Report structure
        # through the strict parser without inventing a correctness score.
        scored = validate_prediction(
            result["prediction_text"], self.ontology["intents"], self.ontology["slots"]
        )
        result.update(scored)
        result.update(system=self.system, model_revision=self.run.config.model_revision)
        return result

    def close(self) -> None:
        from .runtime import release_memory

        global _INFERENCE_ACTIVE
        if not self.closed:
            self.model = self.tokenizer = None
            self.closed = True
            self.run._active_predictor = None
            _INFERENCE_ACTIVE = False
            release_memory(self.run.environment["device"])

    def __enter__(self) -> Predictor:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def load_predictor(run: RunContext, system: str = "qlora") -> Predictor:
    return Predictor(run, system)
