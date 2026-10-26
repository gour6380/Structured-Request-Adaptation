"""Visible PyTorch optimization with token-weighted accumulation and exact resume."""

from __future__ import annotations

import csv
import gc
import io
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoints import read_index, resolve_checkpoint, restore_training_state, save_checkpoint
from .config import ARMS
from .io import (
    atomic_json,
    atomic_text,
    canonical_hash,
    read_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)

if TYPE_CHECKING:
    from .runs import RunContext


def summed_answer_loss(logits, labels):
    """One causal shift, FP32 cross entropy, sum reduction; returns differentiable sum."""
    from .losses import summed_token_loss

    return summed_token_loss(logits[..., :-1, :], labels[..., 1:])


def epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def accumulation_windows(order: list[int], start: int, size: int):
    if (
        type(start) is not int
        or type(size) is not int
        or start < 0
        or start > len(order)
        or size <= 0
        or (start != len(order) and start % size)
    ):
        raise ValueError("Invalid accumulation boundary")
    for position in range(start, len(order), size):
        yield position, order[position : position + size]


def microbatch_windows(window: list[int], size: int):
    """Batch consecutive examples inside one fixed optimizer window."""
    if type(size) is not int or size <= 0:
        raise ValueError("Microbatch size must be a positive integer")
    for position in range(0, len(window), size):
        yield window[position : position + size]


def supervised_count(row: dict) -> int:
    if len(row["input_ids"]) != len(row["labels"]):
        raise ValueError("Input/label lengths differ")
    count = sum(value != -100 for value in row["labels"][1:])
    if count <= 0 or count != row["target_tokens"]:
        raise ValueError(f"Invalid supervised-token accounting for {row['id']}")
    return count


def is_better(candidate: dict, incumbent: dict | None) -> bool:
    """Validation joint exact match, then lower token NLL, retaining earliest tie."""
    if incumbent is None:
        return True
    left = (candidate["joint_exact_match"], -candidate["answer_token_nll"])
    right = (incumbent["joint_exact_match"], -incumbent["answer_token_nll"])
    return left > right


def _write_logs(path: Path, records: list[dict]) -> None:
    write_jsonl(path / "train.jsonl", records)
    stream = io.StringIO()
    columns = [
        "global_step",
        "epoch",
        "loss",
        "learning_rate",
        "target_tokens",
        "examples",
        "elapsed_seconds",
        "gradient_norm",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    atomic_text(path / "train.csv", stream.getvalue())


def reconcile_logs(path: Path, state: dict) -> list[dict]:
    """Drop uncommitted events after a crash and prove the committed prefix unchanged."""
    journal = path / "train.jsonl"
    records = read_jsonl(journal) if journal.exists() else []
    committed = state["journal_length"]
    if len(records) < committed:
        raise ValueError("Training journal is shorter than its committed checkpoint")
    records = records[:committed]
    if canonical_hash(records) != state["journal_hash"]:
        raise ValueError("Committed training metrics were modified")
    committed_epoch_paths = set()
    for epoch in state.get("epochs", []):
        result = path / f"epoch_{epoch['epoch']:02d}.json"
        predictions = path / f"epoch_{epoch['epoch']:02d}_predictions.jsonl"
        if not result.exists() or read_json(result) != epoch:
            raise ValueError("Committed epoch metrics are missing or modified")
        if not predictions.exists() or sha256_file(predictions) != epoch["predictions_sha256"]:
            raise ValueError("Committed epoch predictions are missing or modified")
        committed_epoch_paths.update((result, predictions))
    for result in path.glob("epoch_*"):
        if result.is_file() and result not in committed_epoch_paths:
            recovery = path.parent / ".recovery"
            recovery.mkdir(parents=True, exist_ok=True)
            os.replace(result, recovery / f"uncommitted-{result.name}-{uuid.uuid4().hex}")
    _write_logs(path, records)
    return records


def _log_train_record(writer, record: dict) -> None:
    step = record["global_step"]
    for key in ("loss", "learning_rate", "gradient_norm"):
        writer.add_scalar(f"train/{key}", record[key], step)
    writer.add_scalar("train/answer_tokens_seen", record["target_tokens_seen"], step)
    writer.add_scalar(
        "train/answer_tokens_per_second", record["target_tokens"] / record["elapsed_seconds"], step
    )
    for key, value in record["memory"].items():
        writer.add_scalar(f"memory/train/{key}", value, step)


def _log_validation_record(writer, record: dict) -> None:
    for name, value in record["metrics"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            writer.add_scalar(f"validation/{name}", value, record["global_step"])


def replay_tensorboard_boundary(writer, records: list[dict], state: dict) -> None:
    """After inclusive purge, replay only committed events at the resume boundary."""
    for record in records:
        if record["global_step"] >= state["global_step"]:
            _log_train_record(writer, record)
    for record in state.get("epochs", []):
        if record["global_step"] >= state["global_step"]:
            _log_validation_record(writer, record)


def _merge_memory(old: dict, current: dict) -> dict:
    result = dict(current)
    for key in (
        "rss_bytes",
        "mps_allocated_bytes",
        "mps_driver_bytes",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
    ):
        if key in old:
            result[key] = max(old[key], current.get(key, 0))
    result["system_swap_activity_observed"] = bool(
        old.get("system_swap_activity_observed") or current.get("system_swap_activity_observed")
    )
    return result


def train_adapter(run: RunContext, arm: str) -> dict[str, Any]:
    """Resume the requested arm, or train its full fixed budget. No CLI is involved."""
    if arm not in ARMS:
        raise ValueError(f"Unknown training arm: {arm}")
    from .evaluate import assert_no_active_predictor
    from .models import _registered_files

    with run.writer_lock():
        run.verify_identity()
        root = run.path / "trials" / arm
        summary_path = root / "training_summary.json"
        if summary_path.exists() and _registered_files(run, [summary_path]):
            run.verify_artifacts(prefixes=[str(root.relative_to(run.path))])
            summary = read_json(summary_path)
            if summary.get("completed"):
                resolve_checkpoint(run, arm, "final")
                try:
                    run.require_stage(f"train_{arm}")
                except RuntimeError:
                    # The summary was committed but process shutdown interrupted
                    # the subsequent stage marker. Recover from verified evidence.
                    with run.stage(f"train_{arm}"):
                        pass
                return summary
        # A cache read after evaluation freeze must not alter a completed stage.
        assert_no_active_predictor()
        if getattr(run, "_active_predictor", None) is not None:
            raise RuntimeError("Close the resident predictor before training")
        run.assert_training_open()
        return _train_adapter(run, arm)


def _train_adapter(run, arm: str) -> dict[str, Any]:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm
    from transformers import get_cosine_schedule_with_warmup

    from .data import load_examples
    from .evaluate import assert_no_active_predictor, evaluate_loaded
    from .losses import collate_answer_rows, forward_answer_loss
    from .models import _registered_files, describe_model, load_system
    from .runtime import (
        MemorySampler,
        autocast_context,
        release_memory,
        seed_everything,
        synchronize,
    )

    with run.writer_lock(), run.stage(f"train_{arm}"):
        run.verify_identity()
        assert_no_active_predictor()
        if getattr(run, "_active_predictor", None) is not None:
            raise RuntimeError("Close the resident predictor before training")
        run.assert_training_open()
        run.require_stage("preflight")
        run.require_stage("tokenization")
        run.verify_artifacts(
            prefixes=[
                "preparation/preflight.json",
                "preparation/tokenized_train.jsonl",
                "initial_adapter",
            ]
        )
        preflight = read_json(run.path / "preparation" / "preflight.json")
        if not preflight.get("passed"):
            raise ValueError("Numerical preflight has not passed")
        root = run.path / "trials" / arm
        summary_path = root / "training_summary.json"
        metrics_path = root / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)
        rows = load_examples(run, "train", tokenized=True)
        counts = [supervised_count(row) for row in rows]
        if not rows:
            raise ValueError("Training data is empty")
        effective_batch_size = run.config.effective_batch_size
        per_epoch_updates = math.ceil(len(rows) / effective_batch_size)
        total_updates = per_epoch_updates * run.config.epochs
        expected_tokens = sum(counts) * run.config.epochs
        train_ids_hash = canonical_hash([row["id"] for row in rows])
        index = read_index(run, arm)
        checkpoint = resolve_checkpoint(run, arm, "last") if index.get("last") else None
        state = {
            "schema_version": 1,
            "arm": arm,
            "epoch": 0,
            "next_example": 0,
            "global_step": 0,
            "target_tokens_seen": 0,
            "examples_seen": 0,
            "training_seconds": 0.0,
            "wall_seconds": 0.0,
            "epochs": [],
            "best_metrics": None,
            "journal_length": 0,
            "journal_hash": canonical_hash([]),
            "train_ids_hash": train_ids_hash,
            "memory": {},
            "training_memory": {},
            "validation_memory": {},
            "loading_memory": {},
            "completed": False,
        }
        model = tokenizer = optimizer = scheduler = scaler = writer = None
        segment_start = time.perf_counter()
        memory = MemorySampler(run.environment["device"], run.config.telemetry_interval_seconds)
        phase_memory = None
        try:
            with memory:
                seed_everything(run.config.seed)
                with MemorySampler(
                    run.environment["device"], run.config.telemetry_interval_seconds
                ) as loading_memory:
                    synchronize(run.environment["device"])
                    loading_start = time.perf_counter()
                    model, tokenizer = load_system(
                        run, arm, checkpoint="last" if checkpoint else "initial", training=True
                    )
                    synchronize(run.environment["device"])
                    loading_seconds = time.perf_counter() - loading_start
                specification_path = root / "model_specification.json"
                if not _registered_files(run, [specification_path, root / "architecture.txt"]):
                    atomic_json(specification_path, describe_model(run, model))
                    atomic_text(root / "architecture.txt", str(model) + "\n")
                    run.register_artifacts([specification_path, root / "architecture.txt"])
                else:
                    run.verify_artifacts(
                        prefixes=[
                            str(specification_path.relative_to(run.path)),
                            str((root / "architecture.txt").relative_to(run.path)),
                        ]
                    )
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
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
                if run.environment["use_grad_scaler"]:
                    scaler = torch.amp.GradScaler("cuda")
                if checkpoint:
                    state = restore_training_state(
                        checkpoint, optimizer, scheduler, scaler, run.environment["device"]
                    )
                    if state["train_ids_hash"] != train_ids_hash or state["arm"] != arm:
                        raise ValueError("Checkpoint training data/arm identity mismatch")
                else:
                    # Model/adapter construction may consume different RNG streams.
                    # Pair dropout from the same seed after construction is finished.
                    seed_everything(run.config.seed)
                state["loading_memory"] = _merge_memory(
                    state.get("loading_memory", {}), loading_memory.summary()
                )
                records = reconcile_logs(metrics_path, state)
                old_wall = state["wall_seconds"]
                old_memory = state.get("memory", {})
                writer = SummaryWriter(
                    log_dir=str(run.path / "tensorboard" / arm), purge_step=state["global_step"]
                )
                replay_tensorboard_boundary(writer, records, state)
                optimizer.zero_grad(set_to_none=True)
                for epoch in range(state["epoch"], run.config.epochs):
                    order = epoch_order(len(rows), run.config.seed, epoch)
                    start = state["next_example"] if epoch == state["epoch"] else 0
                    model.train()
                    model.config.use_cache = False
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    windows = accumulation_windows(order, start, effective_batch_size)
                    phase_memory = MemorySampler(
                        run.environment["device"], run.config.telemetry_interval_seconds
                    )
                    phase_memory.__enter__()
                    with tqdm(
                        total=per_epoch_updates,
                        initial=math.ceil(start / effective_batch_size),
                        desc=f"{arm} epoch {epoch + 1}/{run.config.epochs}",
                        unit="update",
                    ) as progress:
                        for position, window in windows:
                            denominator = sum(counts[row_index] for row_index in window)
                            synchronize(run.environment["device"])
                            update_start = time.perf_counter()
                            total_loss = 0.0
                            used_lr = optimizer.param_groups[0]["lr"]
                            for indices in microbatch_windows(window, run.config.microbatch_size):
                                batch, labels = collate_answer_rows(
                                    [rows[index] for index in indices],
                                    run.environment["device"],
                                    tokenizer.pad_token_id,
                                )
                                with autocast_context(run.environment):
                                    numerator, actual_count = forward_answer_loss(
                                        model,
                                        batch,
                                        labels,
                                        answer_only_logits=run.config.answer_only_logits,
                                    )
                                    if actual_count != sum(counts[index] for index in indices):
                                        raise AssertionError(
                                            "Supervised token count changed in training"
                                        )
                                    loss = numerator / denominator
                                if not torch.isfinite(loss).item():
                                    raise FloatingPointError(
                                        f"Non-finite loss in {arm}; no fallback attempted"
                                    )
                                total_loss += float(numerator.detach().item())
                                if scaler is None:
                                    loss.backward()
                                else:
                                    scaler.scale(loss).backward()
                                del batch, labels, numerator, loss
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            norm = torch.nn.utils.clip_grad_norm_(
                                parameters,
                                run.config.gradient_clip,
                                error_if_nonfinite=True,
                                foreach=False,
                            )
                            if scaler is None:
                                optimizer.step()
                            else:
                                scaler.step(optimizer)
                                scaler.update()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            synchronize(run.environment["device"])
                            seconds = time.perf_counter() - update_start
                            state["global_step"] += 1
                            state["next_example"] = position + len(window)
                            state["target_tokens_seen"] += denominator
                            state["examples_seen"] += len(window)
                            state["training_seconds"] += seconds
                            record = {
                                "global_step": state["global_step"],
                                "epoch": epoch + 1,
                                "loss": total_loss / denominator,
                                "learning_rate": used_lr,
                                "target_tokens": denominator,
                                "examples": len(window),
                                "elapsed_seconds": seconds,
                                "gradient_norm": float(norm.item()),
                                "target_tokens_seen": state["target_tokens_seen"],
                                "memory": phase_memory.sample(),
                            }
                            records.append(record)
                            _write_logs(metrics_path, records)
                            _log_train_record(writer, record)
                            progress.update(1)
                            progress.set_postfix(loss=f"{record['loss']:.4f}")
                            state["journal_length"], state["journal_hash"] = (
                                len(records),
                                canonical_hash(records),
                            )
                            state["wall_seconds"] = old_wall + time.perf_counter() - segment_start
                            state["memory"] = _merge_memory(old_memory, memory.summary())
                            if state["global_step"] % run.config.checkpoint_interval == 0:
                                phase_memory.__exit__(None, None, None)
                                state["training_memory"] = _merge_memory(
                                    state.get("training_memory", {}), phase_memory.summary()
                                )
                                phase_memory = None
                                writer.flush()
                                save_checkpoint(
                                    run,
                                    arm,
                                    model,
                                    optimizer,
                                    scheduler,
                                    scaler,
                                    state,
                                    name=f"step_{state['global_step']:08d}",
                                )
                                phase_memory = MemorySampler(
                                    run.environment["device"], run.config.telemetry_interval_seconds
                                )
                                phase_memory.__enter__()
                    phase_memory.__exit__(None, None, None)
                    state["training_memory"] = _merge_memory(
                        state.get("training_memory", {}), phase_memory.summary()
                    )
                    phase_memory = None
                    synchronize(run.environment["device"])
                    validation_start = time.perf_counter()
                    with MemorySampler(
                        run.environment["device"], run.config.telemetry_interval_seconds
                    ) as validation_memory:
                        validation = evaluate_loaded(run, model, tokenizer, "validation")
                        synchronize(run.environment["device"])
                    state["validation_memory"] = _merge_memory(
                        state.get("validation_memory", {}), validation_memory.summary()
                    )
                    validation_seconds = time.perf_counter() - validation_start
                    metrics = validation["metrics"]
                    for name in ("joint_exact_match", "answer_token_nll"):
                        if not math.isfinite(metrics[name]):
                            raise FloatingPointError(f"Non-finite validation metric: {name}")
                    predictions_path = metrics_path / f"epoch_{epoch + 1:02d}_predictions.jsonl"
                    write_jsonl(predictions_path, validation["predictions"])
                    epoch_record = {
                        "epoch": epoch + 1,
                        "global_step": state["global_step"],
                        "metrics": metrics,
                        "validation_seconds": validation_seconds,
                        "predictions_sha256": sha256_file(predictions_path),
                    }
                    _log_validation_record(writer, epoch_record)
                    atomic_json(metrics_path / f"epoch_{epoch + 1:02d}.json", epoch_record)
                    state["epochs"].append(epoch_record)
                    better = is_better(metrics, state["best_metrics"])
                    if better:
                        state["best_metrics"] = dict(metrics)
                    state["epoch"], state["next_example"] = epoch + 1, 0
                    state["wall_seconds"] = old_wall + time.perf_counter() - segment_start
                    state["memory"] = _merge_memory(old_memory, memory.summary())
                    writer.flush()
                    save_checkpoint(
                        run,
                        arm,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        state,
                        name=f"epoch_{epoch + 1:02d}_step_{state['global_step']:08d}",
                        best=better,
                    )
                if (
                    state["global_step"] != total_updates
                    or state["target_tokens_seen"] != expected_tokens
                ):
                    raise AssertionError(
                        "The fixed optimizer-update/token budget was not completed"
                    )
                state["completed"] = True
                state["wall_seconds"] = old_wall + time.perf_counter() - segment_start
                state["memory"] = _merge_memory(old_memory, memory.summary())
                if checkpoint is not None and checkpoint.name == "final":
                    final_path = checkpoint
                else:
                    final_path = save_checkpoint(
                        run,
                        arm,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        state,
                        name="final",
                        final=True,
                    )
                writer.flush()
            summary = {
                "completed": True,
                "arm": arm,
                "epochs_completed": run.config.epochs,
                "optimizer_updates": state["global_step"],
                "microbatch_size": run.config.microbatch_size,
                "gradient_accumulation": run.config.gradient_accumulation,
                "effective_batch_size": effective_batch_size,
                "target_tokens_seen": state["target_tokens_seen"],
                "examples_seen": state["examples_seen"],
                "training_seconds": state["training_seconds"],
                "wall_seconds": old_wall + time.perf_counter() - segment_start,
                "memory": state["training_memory"],
                "training_memory": state["training_memory"],
                "validation_memory": state["validation_memory"],
                "loading_memory": state["loading_memory"],
                "last_segment_loading_seconds": loading_seconds,
                "fit_memory": _merge_memory(state["memory"], memory.summary()),
                "epochs": state["epochs"],
                "best_metrics": state["best_metrics"],
                "final_checkpoint": str(final_path.relative_to(run.path)),
                "seed": run.config.seed,
                "primary_checkpoint": "final",
                "train_ids_hash": train_ids_hash,
                "initial_adapter_sha256": preflight["initial_adapter_sha256"],
                "runtime_scope": "training_seconds includes synchronized optimization windows; wall_seconds includes fit setup, validation and checkpointing",
            }
            atomic_json(summary_path, summary)
            if writer is not None:
                writer.close()
                writer = None
            run.register_artifacts(
                [summary_path]
                + [path for path in metrics_path.rglob("*") if path.is_file()]
                + [path for path in (run.path / "tensorboard" / arm).rglob("*") if path.is_file()],
                replace=True,
            )
            return summary
        finally:
            if phase_memory is not None:
                phase_memory.__exit__(None, None, None)
            if writer is not None:
                writer.close()
            model = optimizer = scheduler = tokenizer = scaler = None
            if "parameters" in locals():
                parameters.clear()
            gc.collect()
            release_memory(run.environment["device"])
