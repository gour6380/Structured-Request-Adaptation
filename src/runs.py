"""Numbered runs, shared-data provenance and explicit stage records."""

from __future__ import annotations

import fcntl
import logging
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .io import atomic_json, canonical_hash, read_json, sha256_file
from .runtime import environment_record

RUN_FOLDERS = (
    "source_snapshot",
    "inputs/dataset",
    "inputs/tokenizer",
    "inputs/base_model",
    "inputs/quantized_base",
    "preparation",
    "initial_adapter",
    "trials/lora",
    "trials/qlora",
    "logs",
    "tensorboard",
    "evaluations",
    "benchmarks",
    "figures",
    "reports",
    "stages",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_files(root: Path) -> dict[str, str]:
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "src").rglob("*.json"))
    paths.append(root / "requirements.txt")
    if any(path.is_symlink() for path in paths):
        raise ValueError("Source and dependency records must be ordinary local files")
    return {str(path.relative_to(root)): sha256_file(path) for path in paths if path.is_file()}


@dataclass
class RunContext:
    project_root: Path
    path: Path
    config: ExperimentConfig
    environment: dict[str, Any]
    _mutex: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _lock_depth: int = field(default=0, repr=False)

    @property
    def identity(self) -> dict[str, Any]:
        return read_json(self.path / "identity.json")

    @property
    def logger(self) -> logging.Logger:
        logger = logging.getLogger(f"structured_request_adaptation.{self.path}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(self.path / "logs" / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger

    def verify_identity(self) -> None:
        self.config.validate()
        identity = self.identity
        if (
            canonical_hash(self.config.to_dict()) != identity["config_hash"]
            or canonical_hash(read_json(self.path / "config.json")) != identity["config_hash"]
        ):
            raise ValueError("Run configuration changed; create a fresh run")
        sources = source_files(self.project_root)
        if canonical_hash(sources) != identity["source_hash"]:
            raise ValueError("Source or requirements changed; create a fresh run")
        snapshot = read_json(self.path / "source_snapshot" / "manifest.json")
        if canonical_hash(snapshot) != identity["source_hash"]:
            raise ValueError("Source snapshot manifest changed")
        for relative, digest in snapshot.items():
            if sha256_file(self.path / "source_snapshot" / relative) != digest:
                raise ValueError(f"Source snapshot changed: {relative}")
        if (
            canonical_hash(read_json(self.path / "environment.json"))
            != identity["environment_hash"]
            or canonical_hash(self.environment) != identity["environment_hash"]
            or canonical_hash(environment_record()) != identity["environment_hash"]
        ):
            raise ValueError(
                "Environment/backend changed; use the recorded environment or start a new run"
            )

    def register_artifacts(self, paths: list[Path], *, replace: bool = False) -> None:
        """Immutable by default; mutable checkpoint/log indexes explicitly opt in."""
        with self.writer_lock():
            manifest = read_json(self.path / "manifest.json")
            for candidate in paths:
                candidate = Path(candidate)
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError(f"Artifact must be a regular file: {candidate}")
                path = candidate.resolve()
                relative = str(path.relative_to(self.path.resolve()))
                if relative in {"manifest.json", "identity.json"}:
                    raise ValueError("Cannot register internal run identity files as artifacts")
                value = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                if (
                    relative in manifest["artifacts"]
                    and manifest["artifacts"][relative] != value
                    and not replace
                ):
                    raise ValueError(f"Immutable artifact changed: {relative}")
                manifest["artifacts"][relative] = value
            manifest["updated_at"] = utc_now()
            atomic_json(self.path / "manifest.json", manifest)

    def verify_artifacts(self, prefixes: list[str] | tuple[str, ...] | None = None) -> None:
        artifacts = read_json(self.path / "manifest.json")["artifacts"]
        for relative, record in artifacts.items():
            if prefixes is not None and not any(
                relative == str(p).rstrip("/") or relative.startswith(str(p).rstrip("/") + "/")
                for p in prefixes
            ):
                continue
            path = self.path / relative
            path.resolve().relative_to(self.path.resolve())
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"Missing or changed run artifact: {relative}")
        # Shared data stays outside the run, but its original hashes remain binding.
        dataset_reference = "inputs/dataset/manifest.json"
        if dataset_reference in artifacts:
            reference_path = self.path / dataset_reference
            if (
                reference_path.is_symlink()
                or not reference_path.is_file()
                or sha256_file(reference_path) != artifacts[dataset_reference]["sha256"]
            ):
                raise ValueError("Missing or changed dataset reference")
            reference = read_json(reference_path)
            if reference.get("storage") == "project_shared":
                from .data import verify_shared_dataset

                verify_shared_dataset(self)

    def require_stage(self, name: str) -> dict[str, Any]:
        path = self.path / "stages" / f"{name}.json"
        if not path.is_file() or (record := read_json(path)).get("status") != "complete":
            raise RuntimeError(f"Complete notebook stage {name!r} first")
        return record

    def assert_training_open(self) -> None:
        self.verify_identity()
        if (self.path / "evaluations" / "frozen_protocol.json").exists():
            raise RuntimeError("Evaluation is frozen; further training requires a fresh run")

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        with self._mutex:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            with (self.project_root / "runs" / ".execution.lock").open("a+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Another notebook/process is writing this project") from exc
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    fcntl.flock(stream, fcntl.LOCK_UN)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid stage name")
        with self.writer_lock():
            self.verify_identity()
            path = self.path / "stages" / f"{name}.json"
            started = utc_now()
            atomic_json(path, {"name": name, "status": "running", "started_at": started})
            self.logger.info("Stage %s started", name)
            try:
                yield
            except BaseException as exc:
                status = (
                    "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
                )
                atomic_json(
                    path,
                    {
                        "name": name,
                        "status": status,
                        "started_at": started,
                        "ended_at": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self.logger.exception("Stage %s %s", name, status)
                raise
            else:
                atomic_json(
                    path,
                    {
                        "name": name,
                        "status": "complete",
                        "started_at": started,
                        "ended_at": utc_now(),
                    },
                )
                self.logger.info("Stage %s complete", name)


def create_run(project_root: Path, config: ExperimentConfig) -> RunContext:
    root = Path(project_root).resolve()
    config.validate()
    if not (root / "src" / "train.py").is_file() or not (root / "requirements.txt").is_file():
        raise ValueError("Select the prepared structured-request-adaptation project")
    environment = environment_record()
    sources = source_files(root)
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    numbers = [
        int(match[1]) for path in runs.iterdir() if (match := re.fullmatch(r"run_(\d+)", path.name))
    ]
    number = max(numbers, default=0) + 1
    while True:
        path = runs / f"run_{number:03d}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            number += 1
    for folder in RUN_FOLDERS:
        (path / folder).mkdir(parents=True, exist_ok=True)
    for relative in sources:
        destination = path / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    documentary_snapshots = []
    for item in (root / "README.md", *sorted((root / "notebooks").glob("*.ipynb"))):
        if item.is_file():
            relative = item.relative_to(root)
            destination = path / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if item.suffix == ".ipynb":
                notebook = read_json(item)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        cell["outputs"], cell["execution_count"] = [], None
                atomic_json(destination, notebook)
            else:
                shutil.copyfile(item, destination)
            documentary_snapshots.append(destination)
    atomic_json(path / "config.json", config.to_dict())
    atomic_json(path / "environment.json", environment)
    atomic_json(path / "source_snapshot" / "manifest.json", sources)
    atomic_json(
        path / "identity.json",
        {
            "config_hash": canonical_hash(config.to_dict()),
            "source_hash": canonical_hash(sources),
            "environment_hash": canonical_hash(environment),
            "created_at": utc_now(),
        },
    )
    atomic_json(path / "manifest.json", {"artifacts": {}, "updated_at": utc_now()})
    context = RunContext(root, path, config, environment)
    context.register_artifacts(documentary_snapshots)
    return context


def open_run(project_root: Path, run_name: str) -> RunContext:
    if not re.fullmatch(r"run_\d{3,}", run_name):
        raise ValueError("Specify an existing name such as run_001")
    root = Path(project_root).resolve()
    path = root / "runs" / run_name
    if path.is_symlink():
        raise ValueError("Runs must be materialized local directories")
    context = RunContext(
        root,
        path,
        ExperimentConfig.from_dict(read_json(path / "config.json")),
        read_json(path / "environment.json"),
    )
    context.verify_identity()
    context.verify_artifacts(prefixes=["inputs/dataset"])
    return context
