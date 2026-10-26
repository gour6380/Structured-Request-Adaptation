"""Process-isolated notebook execution with immutable hardware and source identities."""

from __future__ import annotations

import fcntl
import logging
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig
from src.io import atomic_json, atomic_text, canonical_hash, read_json, sha256_file
from src.runs import RUN_FOLDERS, RunContext, source_files, utc_now

ARM_SYSTEMS = {"lora": ("base", "lora"), "qlora": ("quantized_base", "qlora")}


def _environment(plan: dict[str, Any]) -> dict[str, Any]:
    from src.hardware import validate_hardware_plan
    from src.runtime import environment_record

    validate_hardware_plan(plan)
    record = environment_record()
    if record["device"] != plan["device"]:
        raise RuntimeError("The selected backend is unavailable; no automatic fallback is allowed")
    if record["compute_dtype"] == "float16" and plan["compute_dtype"] == "bfloat16":
        raise RuntimeError("Native BF16 is unavailable on the assigned device")
    record.update(compute_dtype=plan["compute_dtype"], use_grad_scaler=plan["use_grad_scaler"])
    return record


def validate_worker_environment(
    current: dict[str, Any],
    recorded: dict[str, Any],
    binding: dict[str, Any],
    device_index: int | None,
) -> None:
    """Match a masked CUDA worker to its physical UUID; preserve CPU/MPS identity."""
    plan = binding["hardware"]
    actual_gpu = dict(current.get("cuda_device") or {})
    expected_gpu = dict(plan["gpus"][str(device_index)]) if plan["device"] == "cuda" else {}
    actual_gpu.pop("index", None)
    expected_gpu.pop("index", None)
    if actual_gpu != expected_gpu:
        raise ValueError(
            "Device identity changed; explicitly register a compatible replacement session"
        )
    actual = {key: value for key, value in current.items() if key != "cuda_device"}
    expected = {key: value for key, value in recorded.items() if key != "cuda_device"}
    if actual != expected:
        raise ValueError("Environment/backend changed; start a fresh run with this environment")


def _compatible_hardware(previous: dict, current: dict) -> None:
    """Allow replacement device UUIDs, never topology, precision or GPU model changes."""
    for key in previous.keys() | current.keys():
        if key in {"gpus", "arm_masks"}:
            continue
        if key == "host" and previous["device"] == "cuda":
            if (
                previous[key]["machine"] != current[key]["machine"]
                or previous[key]["system"] != current[key]["system"]
            ):
                raise ValueError("Replacement host changed OS or machine architecture")
            continue
        if previous.get(key) != current.get(key):
            raise ValueError(f"Replacement hardware changed {key}; create a fresh run")
    if set(previous["gpus"]) != set(current["gpus"]):
        raise ValueError("Replacement hardware changed the visible GPU inventory")
    for index, gpu in previous["gpus"].items():
        original = {key: value for key, value in gpu.items() if key != "uuid"}
        replacement = {key: value for key, value in current["gpus"][index].items() if key != "uuid"}
        if original != replacement:
            raise ValueError(f"Replacement GPU {index} changed model, memory or capability")


@dataclass
class ExecutionRunContext(RunContext):
    role: str = "coordinator"
    device_index: int | None = None

    @property
    def logger(self) -> logging.Logger:
        logger = logging.getLogger(f"experiment.{self.path}.{self.role}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(self.path / "logs" / f"{self.role}.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger

    def _verify_saved_identity(self) -> dict[str, Any]:
        self.config.validate()
        identity = self.identity
        if any(
            canonical_hash(value) != identity["config_hash"]
            for value in (self.config.to_dict(), read_json(self.path / "config.json"))
        ):
            raise ValueError("Run configuration changed; create a fresh run")
        sources = source_files(self.project_root)
        snapshot = read_json(self.path / "source_snapshot" / "manifest.json")
        if any(canonical_hash(value) != identity["source_hash"] for value in (sources, snapshot)):
            raise ValueError("Project source/dependency identity changed; create a fresh run")
        binding = read_json(self.path / "backend_binding.json")
        if canonical_hash(binding) != identity["backend_binding_hash"]:
            raise ValueError("Backend binding changed")
        for relative, digest in snapshot.items():
            if sha256_file(self.path / "source_snapshot" / relative) != digest:
                raise ValueError(f"Source snapshot changed: {relative}")
        recorded = read_json(self.path / "environment.json")
        if any(
            canonical_hash(value) != identity["environment_hash"]
            for value in (recorded, self.environment)
        ):
            raise ValueError("Saved environment identity changed")
        return binding

    def active_binding(self) -> dict[str, Any]:
        original = self._verify_saved_identity()
        entries = read_json(self.path / "manifest.json")["artifacts"]
        sessions = sorted(
            name for name in entries if re.fullmatch(r"sessions/session_\d{4}\.json", name)
        )
        previous = canonical_hash(original)
        active = original
        for name in sessions:
            path = self.path / name
            if sha256_file(path) != entries[name]["sha256"]:
                raise ValueError("Saved execution session binding changed")
            active = read_json(path)
            if active["previous_binding_hash"] != previous:
                raise ValueError("Execution session binding chain changed")
            previous = canonical_hash(active)
        return active

    def verify_identity(self) -> None:
        binding = self.active_binding()
        recorded = binding.get("environment", self.environment)
        validate_worker_environment(
            _environment(binding["hardware"]), recorded, binding, self.device_index
        )

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        """Shared project gate plus exclusive arm gate; coordinators are exclusive."""
        if self.role not in {"coordinator", "lora", "qlora"}:
            raise ValueError("Unknown execution role")
        with self._mutex:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            with (self.project_root / "runs" / ".execution.lock").open("a+") as project:
                parallel = (
                    self.role != "coordinator"
                    and read_json(self.path / "backend_binding.json")["hardware"]["parallel"]
                )
                mode = fcntl.LOCK_SH if parallel else fcntl.LOCK_EX
                try:
                    fcntl.flock(project, mode | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(
                        "Another coordinator or worker is using this project"
                    ) from exc
                arm_stream = None
                try:
                    if self.role != "coordinator":
                        arm_stream = (self.path / f".{self.role}.lock").open("a+")
                        try:
                            fcntl.flock(arm_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError as exc:
                            raise RuntimeError(
                                f"The {self.role} worker is already running"
                            ) from exc
                    self._lock_depth = 1
                    yield
                finally:
                    self._lock_depth = 0
                    if arm_stream is not None:
                        fcntl.flock(arm_stream, fcntl.LOCK_UN)
                        arm_stream.close()
                    fcntl.flock(project, fcntl.LOCK_UN)

    def register_artifacts(self, paths: list[Path], *, replace: bool = False) -> None:
        # The arm gate spans expensive work; only this short manifest transaction
        # serializes the two arms, avoiding read-modify-write lost updates.
        with self.writer_lock(), (self.path / ".manifest.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                super().register_artifacts(paths, replace=replace)
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def assert_stage_allowed(self, name: str) -> None:
        if self.role == "coordinator":
            return
        allowed = {f"train_{self.role}", f"preflight_{self.role}"}
        allowed.update(
            f"arm_{self.role}_{phase}"
            for phase in ("baseline", "training", "test", "benchmark", "evaluation", "pilot")
        )
        for system in ARM_SYSTEMS[self.role]:
            allowed.add(f"benchmark_{system}")
            allowed.update(f"evaluate_{system}_{split}" for split in ("validation", "test"))
        if name not in allowed:
            raise RuntimeError(f"The {self.role} worker cannot write stage {name!r}")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.assert_stage_allowed(name)
        with super().stage(name):
            yield


def _create_run(root: Path, config: ExperimentConfig, plan: dict[str, Any]) -> ExecutionRunContext:
    from src.hardware import validate_hardware_plan

    config.validate()
    validate_hardware_plan(plan)
    environment = _environment(plan)
    sources = source_files(root)
    binding = {
        "version": 1,
        "hardware": plan,
        "execution": "One isolated process per arm; concurrent only with two selected CUDA devices",
        "resume": "Explicit run name; hardware replacement requires compatible session registration",
        "timing_context": "Concurrent workers share CPU, RAM and storage; these are workload measurements",
    }
    (root / "runs").mkdir(exist_ok=True)
    run = ExecutionRunContext(
        root, root / "runs", config, environment, device_index=plan["arm_gpu"]["lora"]
    )
    with run.writer_lock():
        numbers = [
            int(match[1])
            for item in (root / "runs").iterdir()
            if (match := re.fullmatch(r"run_(\d+)", item.name))
        ]
        number = max(numbers, default=0) + 1
        while True:
            path = root / "runs" / f"run_{number:03d}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1
        run.path = path
        for folder in RUN_FOLDERS:
            (path / folder).mkdir(parents=True, exist_ok=True)
        snapshots = []
        for relative in sources:
            destination = path / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
            snapshots.append(destination)
        for source in (root / "README.md", *sorted((root / "notebooks").glob("*.ipynb"))):
            if source.is_file():
                destination = path / "source_snapshot" / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == ".ipynb":
                    notebook = read_json(source)
                    for cell in notebook["cells"]:
                        if cell["cell_type"] == "code":
                            cell.update(outputs=[], execution_count=None)
                    atomic_json(destination, notebook)
                else:
                    shutil.copyfile(source, destination)
                snapshots.append(destination)
        atomic_json(path / "config.json", config.to_dict())
        atomic_json(path / "environment.json", environment)
        atomic_json(path / "backend_binding.json", binding)
        atomic_json(path / "source_snapshot" / "manifest.json", sources)
        atomic_json(path / "manifest.json", {"artifacts": {}, "updated_at": utc_now()})
        atomic_json(
            path / "identity.json",
            {
                "config_hash": canonical_hash(config.to_dict()),
                "source_hash": canonical_hash(sources),
                "environment_hash": canonical_hash(environment),
                "backend_binding_hash": canonical_hash(binding),
                "created_at": utc_now(),
            },
        )
        run.register_artifacts([*snapshots, path / "backend_binding.json"])
    return run


def _open_run(root: Path, name: str, role: str, device_index: int | None) -> ExecutionRunContext:
    if not re.fullmatch(r"run_\d{3,}", name):
        raise ValueError("Specify an existing run name such as run_001")
    path = root / "runs" / name
    if path.is_symlink():
        raise ValueError("Runs must be ordinary local directories")
    run = ExecutionRunContext(
        root,
        path,
        ExperimentConfig.from_dict(read_json(path / "config.json")),
        read_json(path / "environment.json"),
        role=role,
        device_index=device_index,
    )
    run.verify_identity()
    run.verify_artifacts(prefixes=["inputs/dataset", "backend_binding.json"])
    return run


def _resume_session(root: Path, name: str, plan: dict[str, Any]) -> ExecutionRunContext:
    """Record compatible replacement hardware without rewriting original identity."""
    if not re.fullmatch(r"run_\d{3,}", name):
        raise ValueError("Name the existing run to resume")
    path = root / "runs" / name
    if path.is_symlink():
        raise ValueError("Runs must be ordinary local directories")
    run = ExecutionRunContext(
        root,
        path,
        ExperimentConfig.from_dict(read_json(path / "config.json")),
        read_json(path / "environment.json"),
        device_index=plan["arm_gpu"]["lora"],
    )
    with run.writer_lock():
        previous = run.active_binding()
        run.verify_artifacts()
        _compatible_hardware(previous["hardware"], plan)
        current = _environment(plan)
        reference = previous.get("environment", run.environment)
        allowed = {"cuda_device", "platform"}
        if {k: v for k, v in current.items() if k not in allowed} != {
            k: v for k, v in reference.items() if k not in allowed
        }:
            raise ValueError("Resume requires the exact Python/packages/backend/precision policy")
        manifest = read_json(path / "manifest.json")["artifacts"]
        numbers = [
            int(match[1])
            for relative in manifest
            if (match := re.fullmatch(r"sessions/session_(\d{4})\.json", relative))
        ]
        output = path / "sessions" / f"session_{max(numbers, default=0) + 1:04d}.json"
        atomic_json(
            output,
            {
                "previous_binding_hash": canonical_hash(previous),
                "created_at": utc_now(),
                "hardware": plan,
                "environment": current,
                "changes_allowed": [
                    "GPU UUID",
                    "host platform/kernel string",
                    "CUDA host CPU and RAM, recorded for timing context",
                ],
                "qualification": "Same recorded protocol and checkpoint; no bitwise device reproducibility claim",
            },
        )
        run.register_artifacts([output])
        run.verify_identity()
    return run


def _preflight_arm(run: ExecutionRunContext, arm: str) -> dict[str, Any]:
    """The existing numerical/reload check, explicitly on the arm's own GPU."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from src.models import (
        _adapter_inventory,
        _base,
        _check_adapter_load,
        _check_rotary_precision,
        _files,
        _initial_path,
        _load_initial,
        _prepare_adapter,
        _registered_files,
        _release,
        _tokenizer,
    )
    from src.runtime import autocast_context, seed_everything
    from src.train import summed_answer_loss

    stage = f"preflight_{arm}"
    output_path = run.path / "preparation" / f"{stage}.json"
    with run.writer_lock():
        run.require_stage("tokenization")
        if output_path.is_file() and _registered_files(run, [output_path]):
            run.verify_artifacts(
                prefixes=[
                    f"preparation/{stage}.json",
                    f"preparation/preflight/{arm}",
                    "initial_adapter",
                ]
            )
            saved = read_json(output_path)
            if not saved.get("passed"):
                raise RuntimeError("A committed preflight summary did not pass")
            try:
                run.require_stage(stage)
            except RuntimeError:
                # A crash may land after the durable summary but before the
                # complete stage marker; repair only from verified artifacts.
                with run.stage(stage):
                    pass
            return saved
        if arm == "qlora":
            run.require_stage("preflight_lora")
        with run.stage(stage):
            seed_everything(run.config.seed)
            model = optimizer = None
            try:
                model = _prepare_adapter(run, _base(run, arm == "qlora"), arm)
                rotary_precision = _check_rotary_precision(run, model)
                if any(
                    parameter.device
                    != torch.device(
                        "cuda:0"
                        if run.environment["device"] == "cuda"
                        else run.environment["device"]
                    )
                    for parameter in model.parameters()
                ):
                    raise AssertionError("Model weights escaped the assigned device")
                if arm == "lora" and not _registered_files(run, _files(_initial_path(run))):
                    model.save_pretrained(
                        _initial_path(run), safe_serialization=True, save_embedding_layers=False
                    )
                    run.register_artifacts(_files(_initial_path(run)))
                _load_initial(run, model)
                inventory = _adapter_inventory(run, model)
                if arm == "qlora":
                    first = read_json(run.path / "preparation" / "preflight_lora.json")
                    if canonical_hash(inventory["parameters"]) != canonical_hash(
                        first["inventory"]["parameters"]
                    ):
                        raise AssertionError("The two arms have different trainable parameters")
                tokenizer = _tokenizer(run)
                ids = tokenizer(
                    'Return a JSON object: {"intent":"test","slots":[]}', add_special_tokens=False
                )["input_ids"][:63] + [tokenizer.eos_token_id]
                input_ids = torch.tensor([ids], device=run.environment["device"])
                labels = input_ids.clone()
                labels[:, : max(1, len(ids) - 8)] = -100
                attention = torch.ones_like(input_ids)
                optimizer = torch.optim.AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    lr=run.config.learning_rate,
                    foreach=False,
                    fused=False,
                )
                scaler = torch.amp.GradScaler("cuda", enabled=run.environment["use_grad_scaler"])
                model.train()
                with autocast_context(run.environment):
                    output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                    loss_sum, count = summed_answer_loss(output.logits, labels)
                    loss = loss_sum / count
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite preflight loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                if not gradients or any(
                    gradient is None or not torch.isfinite(gradient).all().item()
                    for gradient in gradients
                ):
                    raise FloatingPointError("Missing or non-finite preflight gradients")
                if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                    raise AssertionError("All adapter gradients are zero")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise AssertionError("Frozen base received gradients")
                scaler.step(optimizer)
                scaler.update()
                if not any(
                    torch.count_nonzero(parameter).item()
                    for name, parameter in model.named_parameters()
                    if "lora_B" in name and parameter.requires_grad
                ):
                    raise AssertionError("Preflight failed to update the adapter")
                if any(
                    not torch.isfinite(p).all().item()
                    for p in model.parameters()
                    if p.requires_grad
                ):
                    raise FloatingPointError("Non-finite updated adapter")
                optimizer.zero_grad(set_to_none=True)
                del output, loss, loss_sum, gradients
                model.eval()
                model.gradient_checkpointing_disable()
                with torch.inference_mode(), autocast_context(run.environment):
                    before = (
                        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                        .logits[:, -1]
                        .float()
                        .cpu()
                    )
                directory = run.path / "preparation" / "preflight" / arm
                model.save_pretrained(
                    directory, safe_serialization=True, save_embedding_layers=False
                )
                del optimizer, model
                optimizer = model = None
                _release(run)
                model = _prepare_adapter(run, _base(run, arm == "qlora"), arm)
                weights = load_file(directory / "adapter_model.safetensors", device="cpu")
                _check_adapter_load(set_peft_model_state_dict(model, weights))
                model.eval()
                model.gradient_checkpointing_disable()
                with torch.inference_mode(), autocast_context(run.environment):
                    after = (
                        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                        .logits[:, -1]
                        .float()
                        .cpu()
                    )
                torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)
                result = {
                    "passed": True,
                    "synthetic_check": True,
                    "rotary_precision": rotary_precision,
                    "gradients_finite": True,
                    "frozen_base": True,
                    "adapter_reload_parity": True,
                    "inventory": inventory,
                    "synthetic_tokens": count,
                    "device_index": run.device_index,
                    "device": run.environment["device"],
                    "cuda_device": _environment(run.active_binding()["hardware"]).get(
                        "cuda_device"
                    ),
                    "initial_adapter_sha256": sha256_file(
                        _initial_path(run) / "adapter_model.safetensors"
                    ),
                }
                atomic_json(output_path, result)
                run.register_artifacts([output_path, *_files(directory)])
                return result
            finally:
                del optimizer, model
                _release(run)


def _finish_preflight(run: ExecutionRunContext) -> dict[str, Any]:
    with run.writer_lock(), run.stage("preflight"):
        arms = {}
        for arm in ARM_SYSTEMS:
            run.require_stage(f"preflight_{arm}")
            run.verify_artifacts(prefixes=[f"preparation/preflight_{arm}.json", "initial_adapter"])
            arms[arm] = read_json(run.path / "preparation" / f"preflight_{arm}.json")
            if not arms[arm]["passed"]:
                raise RuntimeError(f"The {arm} preflight has not passed")
        if arms["lora"]["initial_adapter_sha256"] != arms["qlora"]["initial_adapter_sha256"]:
            raise AssertionError("The two arms did not use the same initial adapter")
        result = {
            "passed": True,
            "arms": arms,
            "synthetic_check": True,
            "initial_adapter_sha256": arms["lora"]["initial_adapter_sha256"],
        }
        path = run.path / "preparation" / "preflight.json"
        atomic_json(path, result)
        run.register_artifacts([path])
        return result


def _arm_phase(run: ExecutionRunContext, phase: str) -> dict[str, Any]:
    from src.evaluate import benchmark_model, evaluate_model
    from src.train import train_adapter

    arm = run.role
    base, adapted = ARM_SYSTEMS[arm]
    with run.writer_lock(), run.stage(f"arm_{arm}_{phase}"):
        run.require_stage("preflight")
        if phase == "pilot":
            from src.pilot import run_pilot

            result = run_pilot(run, arm)
        elif phase == "baseline":
            result = {base: evaluate_model(run, base, split="validation")}
        elif phase == "training":
            run.require_stage(f"evaluate_{base}_validation")
            result = train_adapter(run, arm)
        elif phase == "test":
            run.require_stage("freeze_evaluation")
            if run.config.preset != "full":
                result = {
                    "skipped": True,
                    "reason": "Development runs cannot evaluate held-out test",
                }
            else:
                result = {
                    system: evaluate_model(run, system, split="test") for system in (base, adapted)
                }
        elif phase == "benchmark":
            run.require_stage("freeze_evaluation")
            result = {}
            for system in (base, adapted):
                summary = benchmark_model(run, system)
                summary["execution_context"] = {
                    "mode": "concurrent"
                    if run.active_binding()["hardware"]["parallel"]
                    else "serial",
                    "device_index": run.device_index,
                    "local_cuda_device": 0 if run.environment["device"] == "cuda" else None,
                    "isolated_hardware_benchmark": False,
                    "qualification": "Workers may overlap; these timings measure this notebook workload, not isolated GPU speed",
                }
                path = run.path / "benchmarks" / system / "summary.json"
                atomic_json(path, summary)
                run.register_artifacts([path], replace=True)
                result[system] = summary
        elif phase == "evaluation":
            result = {part: _arm_phase(run, part) for part in ("test", "benchmark")}
        else:
            raise ValueError(f"Unknown arm phase: {phase}")
        path = run.path / "trials" / arm / f"arm_{phase}.json"
        atomic_json(
            path,
            {
                "arm": arm,
                "phase": phase,
                "device_index": run.device_index,
                "device": run.environment["device"],
                "cuda_device": _environment(run.active_binding()["hardware"]).get("cuda_device"),
                "completed_at": utc_now(),
                "result": result,
            },
        )
        run.register_artifacts([path], replace=True)
        return result


def _execution_report(run: ExecutionRunContext) -> Path:
    from src.plots import write_report

    report = write_report(run)
    hardware = run.active_binding()["hardware"]
    mode = "concurrent" if hardware["parallel"] else "serial"
    if hardware["device"] == "cuda":
        assigned = "; ".join(
            f"{arm}: {hardware['gpus'][str(index)]['name']} "
            f"({hardware['gpus'][str(index)]['total_memory'] / 1024**3:.2f} GiB)"
            for arm, index in hardware["arm_gpu"].items()
        )
    else:
        assigned = f"{hardware['device']} on {hardware['host'].get('cpu_model') or hardware['host']['machine']}"
    qualification = (
        "Both arms use the same device serially."
        if not hardware["parallel"]
        else "Selected GPU models and capacities match."
        if hardware["matched_gpus"]
        else "Selected GPUs differ: timing and memory differences cannot be attributed solely to LoRA versus QLoRA."
    )
    note = run.path / "reports" / "execution.md"
    atomic_text(
        note,
        "# Execution context\n\n"
        f"Backend: {hardware['device']}. Arm execution: {mode}. Compute dtype: {hardware['compute_dtype']}.\n\n"
        f"Assigned devices: {assigned}. {qualification}\n\n"
        "Each arm has its own process. CUDA workers see only their assigned GPU as local CUDA 0. "
        "CPU and MPS arms run serially. The arms share saved initialization, data order, effective batch and update budget.\n\n"
        "Concurrent workers share host CPU, RAM and storage; timings may include contention. "
        "Memory samples are observed maxima, not guaranteed allocation peaks.\n\n"
        "Original identities are in backend_binding.json and environment.json. Compatible replacement sessions "
        "are recorded in sessions/session_*.json; this does not imply bitwise cross-device reproducibility.\n",
    )
    run.register_artifacts([note], replace=True)
    atomic_text(
        report,
        report.read_text(encoding="utf-8")
        + f"\n\nAssigned devices: {assigned}. {qualification}\n\n"
        "Execution and timing qualifications: [execution context](execution.md).\n",
    )
    run.register_artifacts([report], replace=True)
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def execute(request: dict[str, Any]) -> dict[str, Any]:
    """Called once in a fresh subprocess; no CLI parser and no implicit experiment."""
    import torch

    from src import data, evaluate, models, plots
    from src.hardware import detect_hardware
    from src.runtime import configure_hub_cache

    root = Path(request["project_root"]).resolve()
    if not (root / "src" / "train.py").is_file():
        raise ValueError("Select the complete structured-request-adaptation project")
    configure_hub_cache(root)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    action = request["action"]
    if action == "inspect_hardware":
        return detect_hardware()
    if action == "run_arm":
        action = f"run_arm_{request['phase']}"
    worker_actions = {
        f"run_arm_{phase}"
        for phase in ("baseline", "training", "test", "benchmark", "evaluation", "pilot")
    }
    if action in {"create_run", "resume_session"}:
        plan = detect_hardware()
        if plan != request["hardware"]:
            raise RuntimeError(
                "Hardware changed after discovery; inspect it again before continuing"
            )
        if plan["device"] == "cuda":
            torch.cuda.set_device(0)
        run = (
            _create_run(root, ExperimentConfig.from_dict(request["config"]), plan)
            if action == "create_run"
            else _resume_session(root, request["run_name"], plan)
        )
        result = {
            "path": str(run.path),
            "config": run.config.to_dict(),
            "binding": run.active_binding(),
        }
    else:
        plan = request["hardware"]
        arm = request.get("arm", "qlora" if action == "predict" else None)
        if action in worker_actions or action == "predict":
            if arm not in ARM_SYSTEMS:
                raise ValueError("Select arm='lora' or arm='qlora'")
            role = arm
        elif action in {"preflight_lora", "preflight_qlora"}:
            role = action.removeprefix("preflight_")
        else:
            role = "coordinator"
        assigned_arm = (
            "qlora"
            if action == "prepare_quantized_base"
            else (role if role != "coordinator" else "lora")
        )
        device_index = plan["arm_gpu"][assigned_arm]
        if plan["device"] == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("The worker must expose exactly its assigned CUDA device")
            if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["arm_masks"][assigned_arm]:
                raise RuntimeError("CUDA mask does not match the assigned arm")
            torch.cuda.set_device(0)
        run = _open_run(root, request["run_name"], role, device_index)
        if plan != run.active_binding()["hardware"]:
            raise ValueError("Launcher hardware does not match this run's active session")
        operations = {
            "acquire_dataset": data.acquire_dataset,
            "prepare_data": data.prepare_data,
            "acquire_model": models.acquire_model,
            "prepare_quantized_base": models.prepare_quantized_base,
            "tokenize_data": data.tokenize_data,
            "inspect_architecture": models.inspect_architecture,
            "preflight": _finish_preflight,
            "freeze_evaluation": evaluate.freeze_evaluation,
            "collect_results": plots.collect_results,
            "plot_losses": plots.plot_losses,
            "plot_validation": plots.plot_validation,
            "plot_task_metrics": plots.plot_task_metrics,
            "plot_memory_runtime": plots.plot_memory_runtime,
            "plot_disagreements": plots.plot_disagreements,
            "write_report": _execution_report,
            "run_summary": plots.run_summary,
        }
        with run.writer_lock():
            if action in worker_actions:
                result = _arm_phase(run, action.removeprefix("run_arm_"))
            elif action in {"preflight_lora", "preflight_qlora"}:
                result = _preflight_arm(run, role)
            elif action == "open_run":
                result = {"path": str(run.path), "config": run.config.to_dict()}
            elif action == "inspect_example":
                result = data.inspect_example(run, index=request.get("index", 0))
            elif action == "predict":
                with evaluate.load_predictor(run, system=arm) as predictor:
                    result = predictor.predict(request["text"])
            elif action in operations:
                result = operations[action](run)
            else:
                raise ValueError(f"Unknown notebook action: {action}")
    return {"action": action, "run_name": run.path.name, "result": _jsonable(result)}
