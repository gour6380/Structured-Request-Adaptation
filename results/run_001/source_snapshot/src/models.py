"""Pinned local model acquisition, matched PEFT preparation, and explicit preflight.

Importing this module never downloads weights or executes model computations.
"""

from __future__ import annotations

import gc
import math
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ARMS, SYSTEMS
from .io import atomic_json, atomic_text, canonical_hash, read_json, sha256_file

if TYPE_CHECKING:
    from .runs import RunContext


def _files(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.rglob("*")) if path.is_file() and ".cache" not in path.parts
    ]


def _registered_files(run, paths: list[Path]) -> bool:
    registered = read_json(run.path / "manifest.json")["artifacts"]
    names = [str(path.relative_to(run.path)) for path in paths]
    return bool(names) and all(name in registered for name in names)


def _committed_summary(run, summary: Path, roots: list[Path]) -> bool:
    if not summary.exists() or not _registered_files(run, [summary]):
        return False
    files = [summary] + [path for root in roots for path in _files(root)]
    if not _registered_files(run, files):
        raise ValueError("Committed model stage contains unregistered artifacts")
    run.verify_artifacts(
        prefixes=[str(summary.relative_to(run.path))]
        + [str(root.relative_to(run.path)) for root in roots]
    )
    return True


def _quarantine_partial(run, path: Path) -> None:
    if not path.exists():
        return
    files = _files(path) if path.is_dir() else [path]
    registered = read_json(run.path / "manifest.json")["artifacts"]
    if any(str(file.relative_to(run.path)) in registered for file in files):
        raise ValueError(f"Refusing to replace partly committed artifacts: {path}")
    recovery = run.path / ".recovery"
    recovery.mkdir(exist_ok=True)
    os.replace(path, recovery / f"partial-{path.name}-{uuid.uuid4().hex}")


def _check_adapter_load(result) -> None:
    missing = [name for name in result.missing_keys if "lora_" in name]
    if result.unexpected_keys or missing:
        raise ValueError(
            f"Adapter keys mismatch; missing={missing}, unexpected={result.unexpected_keys}"
        )


def _policy(run) -> dict:
    return run.environment


def _dtype(run):
    import torch

    return getattr(torch, run.environment["compute_dtype"])


def _release(run):
    from .runtime import release_memory

    gc.collect()
    release_memory(run.environment["device"])


def acquire_model(run: RunContext) -> dict:
    """Explicit user action: download the pinned base snapshot and save its tokenizer."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from .evaluate import assert_no_active_predictor

    with run.writer_lock(), run.stage("acquire_model"):
        assert_no_active_predictor()
        run.verify_identity()
        destination = run.path / "inputs" / "base_model"
        manifest_path = run.path / "inputs" / "model_snapshot.json"
        tokenizer_path = run.path / "inputs" / "tokenizer"
        if _committed_summary(run, manifest_path, [destination, tokenizer_path]):
            return read_json(manifest_path)
        for partial in (destination, tokenizer_path, manifest_path):
            _quarantine_partial(run, partial)
        snapshot_download(
            repo_id=run.config.model_id,
            revision=run.config.model_revision,
            local_dir=destination,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "chat_templates/*.jinja",
                "tokenizer.*",
                "vocab.json",
                "merges.txt",
                "LICENSE*",
            ],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            destination, local_files_only=True, trust_remote_code=False
        )
        tokenizer.save_pretrained(run.path / "inputs" / "tokenizer")
        manifest = {
            "model_id": run.config.model_id,
            "revision": run.config.model_revision,
            "local_path": "inputs/base_model",
            "tokenizer_path": "inputs/tokenizer",
        }
        atomic_json(manifest_path, manifest)
        run.register_artifacts(
            _files(destination) + _files(run.path / "inputs" / "tokenizer") + [manifest_path]
        )
        return manifest


def _base(run, quantized: bool):
    from transformers import AutoModelForCausalLM

    directory = run.path / "inputs" / ("quantized_base" if quantized else "base_model")
    run.require_stage("quantized_base" if quantized else "acquire_model")
    run.verify_artifacts(prefixes=[str(directory.relative_to(run.path))])
    if quantized:
        manifest_path = run.path / "inputs" / "quantization.json"
        if not _registered_files(run, [manifest_path]):
            raise ValueError("The quantization backend manifest is not committed")
        run.verify_artifacts(prefixes=["inputs/quantization.json"])
        expected_backend = read_json(manifest_path)["backend"]
        if _quantization_backend(run) != expected_backend:
            raise RuntimeError(
                "The selected quantization kernel changed; restore the recorded backend or start a fresh run"
            )
    model = AutoModelForCausalLM.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        dtype=_dtype(run),
        device_map={"": run.environment["device"]},
        attn_implementation="sdpa",
    )
    _shared_precision(run, model)
    return model


def _shared_precision(run, model) -> None:
    """Use the same non-projection precision in all four evaluated systems."""
    import torch

    for name, parameter in model.named_parameters():
        is_projection_weight = (
            name.endswith(".weight") and name.split(".")[-2] in run.config.target_modules
        )
        if not is_projection_weight and parameter.is_floating_point():
            parameter.data = parameter.data.to(torch.float32)
    if model.config.tie_word_embeddings:
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise ValueError("Input embeddings and lm_head must remain tied")
    _install_mps_rotary_guard(model)


def _mps_fp32_rotary_forward(module, x, *args, **kwargs):
    """Keep RoPE frequency math FP32 while other model operations use autocast.

    The pinned Transformers Qwen2 forward disables CPU autocast when x is on
    MPS. That leaves outer MPS autocast active and rounds the frequency matmul
    to BF16, which corrupts positions in the long structured-request prompt.
    Delegate to the existing forward so dynamic RoPE handling and its output
    dtype remain unchanged; only the autocast context is corrected.
    """
    import torch

    original = module._forward_without_mps_rope_guard
    if x.device.type != "mps":
        return original(x, *args, **kwargs)
    with torch.autocast(device_type="mps", enabled=False):
        return original(x, *args, **kwargs)


def _install_mps_rotary_guard(model) -> list[str]:
    """Install an idempotent, instance-local Qwen2 guard without changing weights."""
    from types import MethodType

    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    guarded = []
    for name, module in model.named_modules():
        if not isinstance(module, Qwen2RotaryEmbedding):
            continue
        if not getattr(module, "_mps_fp32_rope_guard", False):
            # Both attributes are bound methods, so deepcopy rebinds them to
            # the copied module. They introduce no state_dict parameters/keys.
            module._forward_without_mps_rope_guard = module.forward
            module.forward = MethodType(_mps_fp32_rotary_forward, module)
            module._mps_fp32_rope_guard = True
        guarded.append(name)
    return guarded


def _check_rotary_precision(run, model) -> dict:
    """Check long-position RoPE against independent CPU FP32 frequency math."""
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    from .runtime import autocast_context, torch_dtype

    positions = sorted(
        {
            value
            for value in (0, 255, 256, 257, 511, 512, 513, 639, 1023, 2047)
            if value < run.config.max_sequence_length
        }
    )
    cpu_positions = torch.tensor([positions], dtype=torch.long)
    checked = {}
    dtypes = dict.fromkeys((torch.float32, torch_dtype(run.environment)))
    for name, module in model.named_modules():
        if not isinstance(module, Qwen2RotaryEmbedding):
            continue
        if module.rope_type != "default":
            raise ValueError("The pinned Qwen2.5 protocol requires default RoPE")
        if run.environment["device"] == "mps" and not getattr(
            module, "_mps_fp32_rope_guard", False
        ):
            raise AssertionError("MPS RoPE autocast guard is missing")
        inverse = module.inv_freq.detach().float().cpu()[None, :, None]
        with torch.autocast("cpu", enabled=False):
            frequencies = (inverse @ cpu_positions[:, None, :].float()).transpose(1, 2)
            phases = torch.cat((frequencies, frequencies), dim=-1)
            reference = (
                phases.cos() * module.attention_scaling,
                phases.sin() * module.attention_scaling,
            )
        errors = {}
        for dtype in dtypes:
            inputs = torch.zeros(
                1,
                len(positions),
                module.config.hidden_size,
                device=run.environment["device"],
                dtype=dtype,
            )
            device_positions = cpu_positions.to(run.environment["device"])
            with torch.inference_mode(), autocast_context(run.environment):
                actual = module(inputs, device_positions)
            tolerance = (
                0.004 if dtype == torch.bfloat16 else (0.001 if dtype == torch.float16 else 2e-5)
            )
            maximum_error = 0.0
            for expected, result in zip(reference, actual, strict=True):
                if result.dtype != dtype:
                    raise AssertionError("RoPE changed the incoming activation dtype")
                expected = expected.to(dtype)
                result = result.cpu()
                torch.testing.assert_close(result, expected, rtol=0, atol=tolerance)
                maximum_error = max(
                    maximum_error, (result.float() - expected.float()).abs().max().item()
                )
            errors[str(dtype)] = maximum_error
        checked[name] = errors
    if not checked:
        raise AssertionError("No Qwen2 rotary module was checked")
    return {
        "passed": True,
        "positions": positions,
        "reference": "Independent CPU FP32 frequency matmul and trigonometry",
        "maximum_absolute_errors": checked,
    }


def _tokenizer(run):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        run.path / "inputs" / "tokenizer", local_files_only=True, trust_remote_code=False
    )


def _quantized_layers(model) -> dict:
    import bitsandbytes as bnb

    result = {}
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            state = module.weight.quant_state
            if state is None or state.quant_type != "nf4" or not state.nested:
                raise ValueError(f"Expected initialized nested NF4 weights at {name}")
            result[name] = {
                "quant_type": state.quant_type,
                "nested": state.nested,
                "blocksize": state.blocksize,
                "storage_dtype": str(module.weight.dtype),
                "compute_dtype": str(module.compute_dtype),
            }
    return result


def _quantization_backend(run) -> dict:
    """Identify the implementation actually selected at the explicit quantization stage.

    bitsandbytes may lazily acquire an optional MPS Hub kernel here. Its own
    fallback remains on MPS, and both that choice and kernel bytes are frozen.
    Nothing calls this helper while importing modules or creating the environment.
    """
    from importlib.metadata import version

    record = {"device": run.environment["device"], "bitsandbytes_version": version("bitsandbytes")}
    if run.environment["device"] != "mps":
        return {**record, "implementation": "bitsandbytes_registered_backend", "kernel_assets": {}}
    from bitsandbytes.backends.default import ops as default_ops
    from bitsandbytes.backends.mps import ops

    record["backend_sources"] = {
        "mps_ops.py": sha256_file(Path(ops.__file__)),
        "default_ops.py": sha256_file(Path(default_ops.__file__)),
    }
    kernel = ops._get_kernel()
    record["policy"] = (
        "Freeze the selected MPS implementation and asset hashes; never change device, NF4 format, or compute precision on reload."
    )
    if kernel is None:
        record.update(
            implementation="bitsandbytes_pytorch_mps",
            kernel_assets={},
            selection_reason="macOS below 26"
            if ops._macos_major < 26
            else "Optional Hub kernel unavailable; bitsandbytes does not expose the swallowed exception",
        )
    else:
        module_file = Path(kernel.__file__).absolute()
        variant_root = module_file.parent
        if (
            not (variant_root / "metadata.json").is_file()
            and (variant_root.parent / "metadata.json").is_file()
        ):
            variant_root = variant_root.parent
        assets = {
            str(path.relative_to(variant_root)): sha256_file(path)
            for path in sorted(variant_root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        if not assets:
            raise RuntimeError("Loaded MPS kernel has no hashable assets")
        record.update(
            implementation="bitsandbytes_mps_hub_plus_pytorch",
            hub_repository="kernels-community/bitsandbytes-mps",
            requested_major_version=1,
            module_file=str(module_file),
            kernel_assets=assets,
            operation_policy="Hub quantize/dequantize/GEMV where supported; other operations use bitsandbytes PyTorch MPS implementations.",
        )
    logger = getattr(run, "logger", None)
    if logger is not None:
        logger.info(
            "Selected NF4 backend: %s; identity=%s",
            record["implementation"],
            canonical_hash(record),
        )
    return record


def prepare_quantized_base(run: RunContext) -> dict:
    """Explicitly create one immutable NF4 base for both zero-shot and QLoRA use."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from .evaluate import assert_no_active_predictor
    from .runtime import MemorySampler, synchronize

    with run.writer_lock(), run.stage("quantized_base"):
        assert_no_active_predictor()
        run.require_stage("acquire_model")
        run.verify_artifacts(prefixes=["inputs/base_model"])
        directory = run.path / "inputs" / "quantized_base"
        manifest_path = run.path / "inputs" / "quantization.json"
        if _committed_summary(run, manifest_path, [directory]):
            return read_json(manifest_path)
        for partial in (directory, manifest_path):
            _quarantine_partial(run, partial)
        import time

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype(run),
            llm_int8_skip_modules=["lm_head"],
        )
        synchronize(run.environment["device"])
        start = time.perf_counter()
        model = None
        try:
            with MemorySampler(
                run.environment["device"], run.config.telemetry_interval_seconds
            ) as memory:
                model = AutoModelForCausalLM.from_pretrained(
                    run.path / "inputs" / "base_model",
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=_dtype(run),
                    quantization_config=config,
                    device_map={"": run.environment["device"]},
                    attn_implementation="sdpa",
                )
                layers = _quantized_layers(model)
                expected = model.config.num_hidden_layers * len(run.config.target_modules)
                if len(layers) != expected:
                    raise ValueError(
                        f"Expected {expected} quantized projections; found {len(layers)}"
                    )
                model.save_pretrained(directory, safe_serialization=True)
                backend = _quantization_backend(run)
                synchronize(run.environment["device"])
            manifest = {
                "format": "bitsandbytes_nf4_double_quant",
                "config": config.to_dict(),
                "layers": layers,
                "seconds": time.perf_counter() - start,
                "memory": memory.summary(),
                "base_revision": run.config.model_revision,
                "backend": backend,
            }
            atomic_json(manifest_path, manifest)
            run.register_artifacts(_files(directory) + [manifest_path])
            return manifest
        finally:
            del model
            _release(run)


def _lora_config(run):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=run.config.lora_rank,
        lora_alpha=run.config.lora_alpha,
        lora_dropout=run.config.lora_dropout,
        target_modules=list(run.config.target_modules),
        bias="none",
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
    )


def _prepare_adapter(run, model, arm: str):
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training

    if arm == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        # The k-bit helper would cast every ordinary base weight to FP32.
        # Match QLoRA's unquantized parameters but keep projection weights BF16.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(False)
            is_projection_weight = (
                name.endswith(".weight") and name.split(".")[-2] in run.config.target_modules
            )
            if not is_projection_weight and parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, _lora_config(run), autocast_adapter_dtype=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"Unexpected trainable base parameter: {name}")
            parameter.data = parameter.data.float()
    return model


def _adapter_inventory(run, model) -> dict:
    targets = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    expected = model.config.num_hidden_layers * len(run.config.target_modules)
    if len(targets) != expected or any(
        name.split(".")[-1] not in run.config.target_modules for name in targets
    ):
        raise ValueError(f"Adapter target mismatch: found {len(targets)}, expected {expected}")
    trainable = {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only LoRA A/B matrices may be trainable")
    return {
        "targets": targets,
        "target_count": len(targets),
        "parameters": trainable,
        "trainable_parameters": sum(value["numel"] for value in trainable.values()),
    }


def describe_model(run, model) -> dict:
    """Small serializable architecture and realized storage-precision inventory."""
    tensors = {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "trainable": parameter.requires_grad,
            "stored_elements": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
    }
    return {
        "base_config": model.config.to_dict(),
        "adapter": _adapter_inventory(run, model),
        "parameters": tensors,
        "tied_embeddings": model.get_input_embeddings().weight
        is model.get_output_embeddings().weight,
        "frozen_tensor_count": sum(not item["trainable"] for item in tensors.values()),
        "frozen_stored_elements": sum(
            item["stored_elements"] for item in tensors.values() if not item["trainable"]
        ),
        "storage_note": "Quantized stored_elements counts packed storage, not original logical weights; tied parameters appear once.",
        "base_revision": run.config.model_revision,
        "device": run.environment["device"],
        "compute_dtype": run.environment["compute_dtype"],
        "rotary_precision": {
            "policy": "Disable MPS autocast only inside Qwen2 RoPE; frequency math stays FP32",
            "guarded_modules": [
                name
                for name, module in model.named_modules()
                if getattr(module, "_mps_fp32_rope_guard", False)
            ],
        },
    }


def _initial_path(run) -> Path:
    return run.path / "initial_adapter"


def _load_initial(run, model) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    run.verify_artifacts(prefixes=["initial_adapter"])
    state = load_file(_initial_path(run) / "adapter_model.safetensors", device="cpu")
    result = set_peft_model_state_dict(model, state)
    _check_adapter_load(result)


def load_system(run: RunContext, system: str, checkpoint: str = "final", training: bool = False):
    """Load from this run only. Caller owns model lifetime; no silent network access."""
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system: {system}")
    if training and system not in ARMS:
        raise ValueError("Only adapter arms are trainable")
    run.verify_identity()
    model = _base(run, system in ("quantized_base", "qlora"))
    tokenizer = _tokenizer(run)
    if system in ARMS:
        model = _prepare_adapter(run, model, system)
        if checkpoint == "initial":
            _load_initial(run, model)
        else:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            from .checkpoints import resolve_checkpoint

            path = resolve_checkpoint(run, system, checkpoint)
            state = load_file(path / "adapter" / "adapter_model.safetensors", device="cpu")
            result = set_peft_model_state_dict(model, state)
            _check_adapter_load(result)
    if training:
        model.train()
        model.config.use_cache = False
    else:
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, tokenizer


def inspect_architecture(run: RunContext) -> dict:
    """Read config and tensor headers; do not load model tensors onto an accelerator."""
    with run.writer_lock(), run.stage("architecture"):
        run.require_stage("acquire_model")
        run.verify_artifacts(prefixes=["inputs/base_model"])
        from safetensors import safe_open

        config = read_json(run.path / "inputs" / "base_model" / "config.json")
        tensors = {}
        for file in sorted((run.path / "inputs" / "base_model").glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor = handle.get_slice(key)
                    tensors[key] = {
                        "shape": list(tensor.get_shape()),
                        "dtype": str(tensor.get_dtype()),
                    }
        architecture = {
            "config": config,
            "tensors": tensors,
            "target_modules": list(run.config.target_modules),
            "adapter": {
                "rank": run.config.lora_rank,
                "alpha": run.config.lora_alpha,
                "dropout": run.config.lora_dropout,
            },
            "note": "Header inspection only; no forward pass or empirical memory estimate.",
        }
        output = run.path / "preparation" / "architecture.json"
        atomic_json(output, architecture)
        readable = [
            f"{run.config.model_id}@{run.config.model_revision}",
            f"Architecture: {config['architectures']}",
            f"Layers: {config['num_hidden_layers']}; hidden: {config['hidden_size']}; MLP: {config['intermediate_size']}",
            f"Attention heads: {config['num_attention_heads']}; KV heads: {config['num_key_value_heads']}",
            f"Tied input embedding / output head: {config['tie_word_embeddings']}",
            f"Adapter targets: {', '.join(run.config.target_modules)}",
            "",
            "Tensor names, shapes and source storage:",
        ]
        readable.extend(
            f"{name}: shape={item['shape']} dtype={item['dtype']} elements={math.prod(item['shape'])}"
            for name, item in tensors.items()
        )
        text_path = output.with_suffix(".txt")
        atomic_text(text_path, "\n".join(readable) + "\n")
        run.register_artifacts([output, text_path])
        return {"architecture_path": str(output), "tensor_count": len(tensors), "config": config}


def preflight(run: RunContext) -> dict:
    """Explicit notebook action: tiny numerical checks, never called on import or setup.

    Uses a synthetic short sequence and long-position RoPE parity, not scored
    dataset examples. Each arm resets to the immutable adapter before training.
    """
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from .evaluate import assert_no_active_predictor
    from .runtime import autocast_context, seed_everything
    from .train import summed_answer_loss

    with run.writer_lock(), run.stage("preflight"):
        assert_no_active_predictor()
        run.require_stage("acquire_model")
        run.require_stage("quantized_base")
        run.require_stage("tokenization")
        report_path = run.path / "preparation" / "preflight.json"
        if _committed_summary(
            run, report_path, [_initial_path(run), run.path / "preparation" / "preflight"]
        ):
            return read_json(report_path)
        _quarantine_partial(run, report_path)
        report = {"passed": False, "arms": {}, "synthetic_check": True}
        expected_signature = None
        for arm in ARMS:
            seed_everything(run.config.seed)
            model = None
            optimizer = None
            try:
                model = _prepare_adapter(run, _base(run, arm == "qlora"), arm)
                rotary_precision = _check_rotary_precision(run, model)
                if not _registered_files(run, _files(_initial_path(run))):
                    _quarantine_partial(run, _initial_path(run))
                    model.save_pretrained(
                        _initial_path(run), safe_serialization=True, save_embedding_layers=False
                    )
                    run.register_artifacts(_files(_initial_path(run)))
                _load_initial(run, model)
                inventory = _adapter_inventory(run, model)
                signature = canonical_hash(inventory["parameters"])
                if expected_signature is not None and signature != expected_signature:
                    raise AssertionError(
                        "The two arms do not have identical trainable shapes/dtypes"
                    )
                expected_signature = signature
                tokenizer = _tokenizer(run)
                ids = tokenizer(
                    'Return a JSON object: {"intent":"test","slots":[]}', add_special_tokens=False
                )["input_ids"]
                ids = ids[:63] + [tokenizer.eos_token_id]
                input_ids = torch.tensor([ids], device=run.environment["device"])
                labels = input_ids.clone()
                labels[:, : max(1, len(ids) - 8)] = -100
                attention = torch.ones_like(input_ids)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=run.config.learning_rate,
                    foreach=False,
                    fused=False,
                )
                scaler = (
                    torch.amp.GradScaler("cuda") if run.environment["use_grad_scaler"] else None
                )
                model.train()
                with autocast_context(_policy(run)):
                    output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                    loss_sum, count = summed_answer_loss(output.logits, labels)
                    loss = loss_sum / count
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite {arm} preflight loss")
                if scaler is None:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                gradients = [
                    parameter.grad for parameter in model.parameters() if parameter.requires_grad
                ]
                if not gradients or any(
                    gradient is None or not torch.isfinite(gradient).all().item()
                    for gradient in gradients
                ):
                    raise AssertionError(f"Missing or non-finite adapter gradient in {arm}")
                if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                    raise AssertionError(f"All adapter gradients are zero in {arm}")
                if any(
                    parameter.grad is not None
                    for parameter in model.parameters()
                    if not parameter.requires_grad
                ):
                    raise AssertionError("Frozen base received gradients")
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                if not any(
                    torch.count_nonzero(parameter).item()
                    for name, parameter in model.named_parameters()
                    if "lora_B" in name and parameter.requires_grad
                ):
                    raise AssertionError(
                        "The preflight optimizer did not update initially zero LoRA B weights"
                    )
                if any(
                    not torch.isfinite(parameter).all().item()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ):
                    raise FloatingPointError(
                        "Non-finite adapter values after the preflight optimizer step"
                    )
                optimizer.zero_grad(set_to_none=True)
                del output, loss, loss_sum, gradients
                model.eval()
                model.gradient_checkpointing_disable()
                model.config.use_cache = True
                with torch.inference_mode(), autocast_context(_policy(run)):
                    before = (
                        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                        .logits[:, -1]
                        .float()
                        .cpu()
                    )
                adapter_path = run.path / "preparation" / "preflight" / arm
                if adapter_path.exists() and not _registered_files(run, _files(adapter_path)):
                    _quarantine_partial(run, adapter_path)
                model.save_pretrained(
                    adapter_path, safe_serialization=True, save_embedding_layers=False
                )
                # Reload into a fresh base and fresh adapter wrapper, not the live object.
                del optimizer, model
                optimizer = model = None
                _release(run)
                model = _prepare_adapter(run, _base(run, arm == "qlora"), arm)
                loaded = load_file(adapter_path / "adapter_model.safetensors", device="cpu")
                _check_adapter_load(set_peft_model_state_dict(model, loaded))
                model.eval()
                model.gradient_checkpointing_disable()
                with torch.inference_mode(), autocast_context(_policy(run)):
                    after = (
                        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                        .logits[:, -1]
                        .float()
                        .cpu()
                    )
                torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)
                report["arms"][arm] = {
                    "gradients_finite": True,
                    "frozen_base": True,
                    "adapter_reload_parity": True,
                    "inventory": inventory,
                    "synthetic_tokens": count,
                    "rotary_precision": rotary_precision,
                }
                del input_ids, labels, attention, before, after, loaded
            finally:
                del optimizer, model
                _release(run)
        report["initial_adapter_sha256"] = sha256_file(
            _initial_path(run) / "adapter_model.safetensors"
        )
        report["passed"] = True
        atomic_json(report_path, report)
        run.register_artifacts([report_path] + _files(run.path / "preparation" / "preflight"))
        return report
