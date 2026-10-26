"""Bounded measurements using disposable adapters and the real batching helpers.

No checkpoints, training journals, final adapters, or validation-selection state
are written. A pilot's updated in-memory adapter is discarded before returning.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from itertools import islice

from .io import atomic_json, read_json, write_jsonl

WARMUP_UPDATES = 2
MEASURED_UPDATES = 20
VALIDATION_EXAMPLES = 32


def _gradient_scaler(policy: dict):
    """Honor the frozen precision policy without changing device or precision."""
    import torch

    enabled = policy.get("use_grad_scaler", False)
    if type(enabled) is not bool:
        raise ValueError("use_grad_scaler must be a boolean")
    needs_scaling = (
        str(policy["device"]).split(":")[0] == "cuda" and policy["compute_dtype"] == "float16"
    )
    if enabled != needs_scaling:
        raise ValueError("Gradient scaling must be enabled exactly for CUDA FP16")
    return torch.amp.GradScaler("cuda") if enabled else None


def training_projection(
    seconds: list[float], training_examples: int, effective_batch: int, epochs: int
) -> dict:
    if not seconds or any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("Pilot needs positive finite measured update times")
    if min(training_examples, effective_batch, epochs) <= 0:
        raise ValueError("Projection workload must be positive")
    updates = math.ceil(training_examples / effective_batch) * epochs
    ordered = sorted(seconds)
    median = statistics.median(ordered)
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    return {
        "optimizer_updates": updates,
        "seconds_per_update_median": median,
        "seconds_per_update_p95_observed": p95,
        "training_compute_seconds_at_median": median * updates,
        "training_compute_seconds_at_observed_p95": p95 * updates,
        "qualification": "Conditional extrapolation from this sample; excludes loading, checkpoint writes, validation, test and benchmarks. Not a completion guarantee or confidence interval.",
    }


def run_pilot(run, arm: str) -> dict:
    """Measure disposable work while preserving the caller's random state."""
    from .checkpoints import capture_rng, restore_rng

    original_rng = capture_rng()
    try:
        return _run_pilot(run, arm)
    finally:
        restore_rng(original_rng)


def _run_pilot(run, arm: str) -> dict:
    import torch
    from tqdm.auto import tqdm
    from transformers import get_cosine_schedule_with_warmup

    from .data import load_examples, load_ontology
    from .evaluate import _generate_batch, assert_no_active_predictor
    from .losses import collate_answer_rows, forward_answer_loss, forward_answer_nll_per_example
    from .metrics import score_prediction
    from .models import _quarantine_partial, _registered_files, load_system
    from .runtime import (
        MemorySampler,
        autocast_context,
        release_memory,
        seed_everything,
        synchronize,
    )
    from .telemetry import ActivityMonitor
    from .train import accumulation_windows, epoch_order, microbatch_windows, supervised_count

    if arm not in {"lora", "qlora"}:
        raise ValueError("Pilot requires a named LoRA or QLoRA arm")
    run.verify_identity()
    run.require_stage("preflight")
    directory = run.path / "trials" / arm / "pilot"
    summary_path = directory / "summary.json"
    if _registered_files(run, [summary_path]):
        run.verify_artifacts(prefixes=[str(directory.relative_to(run.path))])
        return read_json(summary_path)
    run.assert_training_open()
    assert_no_active_predictor()
    _quarantine_partial(run, directory)
    directory.mkdir(parents=True, exist_ok=True)
    training = load_examples(run, "train", tokenized=True)
    validation = load_examples(run, "validation", tokenized=True)
    if not training or not validation:
        raise ValueError("Pilot requires nonempty prepared training and validation data")
    selected = random.Random(run.config.seed).sample(
        validation, min(VALIDATION_EXAMPLES, len(validation))
    )
    ontology = load_ontology(run)
    order = epoch_order(len(training), run.config.seed, 0)
    windows = list(
        islice(
            accumulation_windows(order, 0, run.config.effective_batch_size),
            WARMUP_UPDATES + MEASURED_UPDATES,
        )
    )
    warmup_updates = min(WARMUP_UPDATES, len(windows) - 1)
    model = tokenizer = optimizer = scheduler = scaler = None
    batch = labels = numerator = loss = norm = None
    parameters = []
    phase_memory = {}
    update_records = []
    predictions = []
    batch_records = []
    device = run.environment["device"]
    total_updates = math.ceil(len(training) / run.config.effective_batch_size) * run.config.epochs
    activity = ActivityMonitor(directory / "activity.jsonl", device=device, interval=1.0)
    started = time.perf_counter()
    try:
        scaler = _gradient_scaler(run.environment)
        with activity:
            seed_everything(run.config.seed)
            synchronize(device)
            loading_start = time.perf_counter()
            with MemorySampler(device, run.config.telemetry_interval_seconds) as memory:
                model, tokenizer = load_system(run, arm, checkpoint="initial", training=True)
                synchronize(device)
            loading_seconds = time.perf_counter() - loading_start
            phase_memory["loading"] = memory.summary()
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
                raise AssertionError("Pilot requires FP32 adapter parameters")
            optimizer = torch.optim.AdamW(
                parameters,
                lr=run.config.learning_rate,
                betas=run.config.adam_betas,
                eps=run.config.adam_epsilon,
                weight_decay=run.config.weight_decay,
                foreach=False,
                fused=False,
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=math.ceil(run.config.warmup_ratio * total_updates),
                num_training_steps=total_updates,
            )
            seed_everything(run.config.seed)
            model.train()
            model.config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            optimizer.zero_grad(set_to_none=True)
            with MemorySampler(device, run.config.telemetry_interval_seconds) as memory:
                for update, (_, window) in enumerate(
                    tqdm(windows, desc=f"{arm} bounded pilot", unit="update")
                ):
                    activity.check_safety()
                    denominator = sum(supervised_count(training[index]) for index in window)
                    synchronize(device)
                    before = time.perf_counter()
                    summed_loss = 0.0
                    for indices in microbatch_windows(window, run.config.microbatch_size):
                        rows = [training[index] for index in indices]
                        batch, labels = collate_answer_rows(rows, device, tokenizer.pad_token_id)
                        with autocast_context(run.environment):
                            numerator, count = forward_answer_loss(
                                model,
                                batch,
                                labels,
                                answer_only_logits=run.config.answer_only_logits,
                            )
                            if count != sum(supervised_count(row) for row in rows):
                                raise AssertionError(
                                    "Pilot supervised-token accounting differs from training"
                                )
                            loss = numerator / denominator
                        if not torch.isfinite(loss).item():
                            raise FloatingPointError(
                                "Nonfinite pilot loss; no precision or batch fallback"
                            )
                        if scaler is None:
                            loss.backward()
                        else:
                            scaler.scale(loss).backward()
                        summed_loss += float(numerator.detach().item())
                        del batch, labels, numerator, loss
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, run.config.gradient_clip, error_if_nonfinite=True, foreach=False
                    )
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    synchronize(device)
                    update_records.append(
                        {
                            "update": update + 1,
                            "warmup": update < warmup_updates,
                            "seconds": time.perf_counter() - before,
                            "examples": len(window),
                            "target_tokens": denominator,
                            "loss": summed_loss / denominator,
                            "gradient_norm": float(norm.item()),
                        }
                    )
            phase_memory["training"] = memory.summary()
            # Optimizer state must not inflate evaluation's observed allocation.
            parameters.clear()
            scheduler = optimizer = scaler = norm = None
            release_memory(device)
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            batch_size = run.config.evaluation_batch_size
            synchronize(device)
            nll_start = time.perf_counter()
            nll_sum, target_tokens = 0.0, 0
            with MemorySampler(device, run.config.telemetry_interval_seconds) as memory:
                for start in range(0, len(selected), batch_size):
                    activity.check_safety()
                    batch, labels = collate_answer_rows(
                        selected[start : start + batch_size], device, tokenizer.pad_token_id
                    )
                    with torch.inference_mode(), autocast_context(run.environment):
                        values = forward_answer_nll_per_example(
                            model, batch, labels, answer_only_logits=run.config.answer_only_logits
                        )
                    for nll, count in values:
                        nll_sum += nll
                        target_tokens += count
                    del batch, labels
                synchronize(device)
            nll_seconds = time.perf_counter() - nll_start
            phase_memory["validation_nll"] = memory.summary()
            # One discarded generation batch warms decoder kernels separately.
            _generate_batch(
                run, model, tokenizer, [row["prompt_ids"] for row in selected[:batch_size]]
            )
            synchronize(device)
            generation_start = time.perf_counter()
            with MemorySampler(device, run.config.telemetry_interval_seconds) as memory:
                for start in range(0, len(selected), batch_size):
                    activity.check_safety()
                    rows = selected[start : start + batch_size]
                    result = _generate_batch(
                        run, model, tokenizer, [row["prompt_ids"] for row in rows]
                    )
                    batch_records.append(
                        {key: value for key, value in result.items() if key != "predictions"}
                    )
                    for row, prediction in zip(rows, result["predictions"], strict=True):
                        predictions.append(
                            {
                                "id": row["id"],
                                "target": row["target"],
                                **prediction,
                                **score_prediction(
                                    prediction["prediction_text"],
                                    row["target"],
                                    ontology["intents"],
                                    ontology["slots"],
                                ),
                            }
                        )
                synchronize(device)
            generation_seconds = time.perf_counter() - generation_start
            phase_memory["validation_generation"] = memory.summary()
        measured = [row for row in update_records if not row["warmup"]]
        result = {
            "arm": arm,
            "completed": True,
            "disposable_adapter": True,
            "primary_training_changed": False,
            "checkpoint": "initial",
            "warmup_updates": warmup_updates,
            "warmup_qualification": (
                "No optimizer warmup was possible; this single-window timing includes cold-start overhead."
                if warmup_updates == 0
                else f"Excluded {warmup_updates} optimizer warmup update(s); at least one measured update retained."
            ),
            "measured_updates": len(measured),
            "microbatch_size": run.config.microbatch_size,
            "effective_batch_size": run.config.effective_batch_size,
            "evaluation_batch_size": batch_size,
            "max_new_tokens": run.config.max_new_tokens,
            "device": device,
            "compute_dtype": run.environment["compute_dtype"],
            "use_grad_scaler": run.environment.get("use_grad_scaler", False),
            "loading_seconds": loading_seconds,
            "wall_seconds": time.perf_counter() - started,
            "training_examples": len(training),
            "validation_examples": len(validation),
            "epochs": run.config.epochs,
            "updates": update_records,
            "validation_sample": {
                "examples": len(selected),
                "example_ids": [row["id"] for row in selected],
                "nll_seconds": nll_seconds,
                "answer_token_nll": nll_sum / target_tokens,
                "target_tokens": target_tokens,
                "generation_seconds": generation_seconds,
                "generation_batches": batch_records,
                "generated_tokens": sum(row["generated_tokens"] for row in predictions),
                "hit_generation_limit_count": sum(
                    row["hit_generation_limit"] for row in predictions
                ),
            },
            "training_projection": training_projection(
                [row["seconds"] for row in measured],
                len(training),
                run.config.effective_batch_size,
                run.config.epochs,
            ),
            "activity": activity.summary(),
            "phase_memory": phase_memory,
            "qualification": "Both arms may run concurrently and share host resources. Sample validation uses a disposable partially trained adapter; output length and full-run timing may change. Pilot is not an experiment result.",
        }
        predictions_path = directory / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        atomic_json(summary_path, result)
        run.register_artifacts([summary_path, predictions_path, directory / "activity.jsonl"])
        return result
    finally:
        parameters.clear()
        model = tokenizer = optimizer = scheduler = scaler = None
        # An exception traceback can retain this frame; discard incomplete
        # forward/backward tensors before asking the backend to release memory.
        batch = labels = numerator = loss = norm = None
        del batch, labels, numerator, loss, norm
        release_memory(device)
