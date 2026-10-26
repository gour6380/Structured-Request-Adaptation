"""Atomic local checkpoints with a hash chain and exact optimizer-boundary resume."""

from __future__ import annotations

import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

from .io import atomic_json, read_json, sha256_file


def checkpoint_root(run, arm: str) -> Path:
    if arm not in ("lora", "qlora"):
        raise ValueError(f"Unknown training arm: {arm}")
    return run.path / "trials" / arm / "checkpoints"


def capture_rng() -> dict[str, Any]:
    import numpy as np
    import torch

    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy_kind": np_state[0],
        "numpy_keys": np_state[1].tolist(),
        "numpy_pos": int(np_state[2]),
        "numpy_has_gauss": int(np_state[3]),
        "numpy_gauss": float(np_state[4]),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(
        (
            state["numpy_kind"],
            np.asarray(state["numpy_keys"], dtype=np.uint32),
            state["numpy_pos"],
            state["numpy_has_gauss"],
            state["numpy_gauss"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if "torch_mps" in state:
        if not torch.backends.mps.is_available():
            raise RuntimeError("An MPS checkpoint cannot resume on another backend")
        torch.mps.set_rng_state(state["torch_mps"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA checkpoint cannot resume on another backend")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint path leaves its run")
    return path


def verify_checkpoint(path: Path, expected_manifest_hash: str | None = None) -> dict:
    manifest_path = path / "hashes.json"
    if expected_manifest_hash and sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError(f"Changed checkpoint manifest: {path}")
    manifest = read_json(manifest_path)
    for relative, expected in manifest["files"].items():
        candidate = _contained(path, relative)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"Missing or modified checkpoint artifact: {candidate}")
    return manifest


def read_index(run, arm: str) -> dict:
    root = checkpoint_root(run, arm)
    prefix = str((root / "commits").relative_to(run.path)) + "/index_"
    artifacts = read_json(run.path / "manifest.json")["artifacts"]
    committed = sorted(
        name for name in artifacts if name.startswith(prefix) and name.endswith(".json")
    )
    if not committed:
        return {"schema_version": 1, "entries": {}, "last": None, "best": None, "final": None}
    index_path = run.path / committed[-1]
    run.verify_artifacts(prefixes=[str(index_path.relative_to(run.path))])
    return read_json(index_path)


def resolve_checkpoint(run, arm: str, selector: str = "final") -> Path:
    index = read_index(run, arm)
    name = index.get(selector) if selector in ("last", "best", "final") else selector
    if not name or name not in index["entries"]:
        raise FileNotFoundError(f"No {selector} checkpoint for {arm}")
    path = _contained(checkpoint_root(run, arm), name)
    verify_checkpoint(path, index["entries"][name]["manifest_sha256"])
    return path


def save_checkpoint(
    run,
    arm,
    model,
    optimizer,
    scheduler,
    scaler,
    state: dict,
    *,
    name: str,
    best: bool = False,
    final: bool = False,
) -> Path:
    """Commit complete files, then index, then prune. No partial adapter is loadable."""
    import torch

    root = checkpoint_root(run, arm)
    root.mkdir(parents=True, exist_ok=True)
    if Path(name).name != name or name.startswith("."):
        raise ValueError("Checkpoint name must be a plain directory name")
    destination = root / name
    index = read_index(run, arm)
    if destination.exists():
        if name in index["entries"]:
            raise FileExistsError(f"Committed checkpoint already exists: {destination}")
        # The old attempt may have crashed after directory publication but before
        # the atomic manifest commit. Verify then quarantine; never adopt it.
        verify_checkpoint(destination)
        os.replace(destination, root / f".orphan-{name}-{uuid.uuid4().hex}")
    temporary = root / f".pending-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        model.save_pretrained(
            temporary / "adapter", safe_serialization=True, save_embedding_layers=False
        )
        atomic_json(temporary / "progress.json", state)
        payload = {
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": capture_rng(),
        }
        with (temporary / "training_state.pt").open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        hashes = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        atomic_json(temporary / "hashes.json", {"schema_version": 1, "files": hashes})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    index["entries"][name] = {
        "manifest_sha256": sha256_file(destination / "hashes.json"),
        "global_step": state["global_step"],
        "epoch": state["epoch"],
    }
    index["last"] = name
    if best:
        index["best"] = name
    if final:
        index["final"] = name
    ordered = sorted(index["entries"], key=lambda key: index["entries"][key]["global_step"])
    keep = set(ordered[-2:]) | {index.get("best"), index.get("final"), name}
    removed = [key for key in index["entries"] if key not in keep]
    for key in removed:
        del index["entries"][key]
    commits = root / "commits"
    commits.mkdir(exist_ok=True)
    numbers = [int(path.stem.split("_")[-1]) for path in commits.glob("index_*.json")]
    commit_path = commits / f"index_{max(numbers, default=0) + 1:08d}.json"
    atomic_json(commit_path, index)
    # Registering this immutable index atomically updates the run manifest.
    # Until this single commit point, all prior checkpoints remain authoritative.
    run.register_artifacts([commit_path])
    logger = getattr(run, "logger", None)
    if logger is not None:
        logger.info(
            "Checkpoint %s/%s committed at optimizer step %s (best=%s, final=%s)",
            arm,
            name,
            state["global_step"],
            best,
            final,
        )
    atomic_json(root / "index.json", index)  # convenience only; never trusted on load
    # The index no longer refers to pruned directories, even if cleanup is interrupted.
    for key in removed:
        path = _contained(root, key)
        if path.exists():
            shutil.rmtree(path)
    return destination


def restore_training_state(path: Path, optimizer, scheduler, scaler, device: str) -> dict:
    """Only load hash-verified locally created checkpoints; never arbitrary pickle."""
    import torch

    verify_checkpoint(path)
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    # AdamW scalar step tensors remain on CPU when capturable=False.
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and key != "step":
                values[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    if payload["scaler"] is not None:
        if scaler is None:
            raise ValueError("Checkpoint requires the original gradient scaler")
        scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"])
    return read_json(path / "progress.json")
