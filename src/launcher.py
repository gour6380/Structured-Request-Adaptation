"""Standard-library notebook controller with isolated processes and serial queues."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WORKER_CODE = """
import json, os, traceback
from importlib import import_module
from pathlib import Path
request_path = Path(os.environ['SRA_REQUEST_PATH'])
result_path = Path(os.environ['SRA_RESULT_PATH'])
try:
    execute = import_module(os.environ['SRA_BACKEND_MODULE']).execute
    result = {'ok': True, 'result': execute(json.loads(request_path.read_text()))}
except BaseException as error:
    traceback.print_exc()
    result = {'ok': False, 'error': type(error).__name__, 'message': str(error)}
temporary = result_path.with_suffix('.tmp')
with temporary.open('w') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\\n')
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, result_path)
if not result['ok']:
    raise SystemExit(1)
"""


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Worker:
    """A durable job description with either a live child or a queued dependency."""

    label: str
    request_path: Path
    result_path: Path
    log_path: Path
    resource: str | None
    visible: str | None
    coordinator: bool = False
    predecessor: Worker | None = None
    process: subprocess.Popen | None = None
    cancelled: bool = False
    submitted: float = field(default_factory=time.monotonic)
    started: float | None = None

    def status(self) -> dict[str, Any]:
        code = self.process.poll() if self.process is not None else None
        state = (
            "cancelled"
            if self.cancelled
            else "queued"
            if self.process is None
            else "running"
            if code is None
            else "complete"
            if code == 0
            else "failed"
        )
        return {
            "worker": self.label,
            "pid": self.process.pid if self.process else None,
            "status": state,
            "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - self.started, 1) if self.started else 0,
            "queued_seconds": round((self.started or time.monotonic()) - self.submitted, 1),
            "resource": self.resource,
            "log": str(self.log_path),
            "result": str(self.result_path),
        }


class ExperimentLauncher:
    """Use separate CUDA devices concurrently; queue CPU/MPS/single-CUDA arms.

    Models never enter the notebook process. OS leases survive notebook object
    resets while children run. A queued request starts only through this
    controller's wait(), and is never silently replayed after a kernel restart.
    """

    backend_module = "src.execution"

    def __init__(self, project_root: str | Path, python_executable: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.python = Path(python_executable or sys.executable)
        if not self.python.is_file():
            raise FileNotFoundError(f"Python interpreter not found: {self.python}")
        self.jobs = self.root / "jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.leases = self.root / ".device-locks"
        self.leases.mkdir(exist_ok=True)
        self.hardware: dict[str, Any] | None = None
        self._workers: list[Worker] = []
        atexit.register(self._shutdown)

    def _new_worker(
        self,
        request: dict,
        *,
        label: str,
        resource: str | None,
        visible: str | None,
        coordinator: bool = False,
        predecessor: Worker | None = None,
    ) -> Worker:
        job_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{label}-{uuid.uuid4().hex[:8]}"
        directory = self.jobs / job_id
        directory.mkdir()
        worker = Worker(
            label,
            directory / "request.json",
            directory / "result.json",
            directory / "worker.log",
            resource,
            visible,
            coordinator,
            predecessor,
        )
        _atomic_json(worker.request_path, {**request, "project_root": str(self.root)})
        worker.log_path.touch()
        self._workers.append(worker)
        if predecessor is None:
            self._spawn(worker)
        else:
            _atomic_json(directory / "process.json", worker.status())
            print(f"Queued {label} behind {predecessor.label}; device resources will not overlap.")
        return worker

    def _spawn(self, worker: Worker) -> None:
        leases = []
        try:
            gate = (self.leases / "execution.lock").open("a+")
            leases.append(gate)
            mode = fcntl.LOCK_EX if worker.coordinator else fcntl.LOCK_SH
            try:
                fcntl.flock(gate, mode | fcntl.LOCK_NB)
                if worker.resource is not None:
                    # Resource identifiers come from the validated hardware plan.
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", worker.resource)
                    lease = (self.leases / f"{safe_name}.lock").open("a+")
                    leases.append(lease)
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Device resources are busy ({worker.resource or 'coordinator'}); wait or cancel the existing worker"
                ) from error
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env.update(
                PYTHONPATH=str(self.root),
                PYTHONUNBUFFERED="1",
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                MPLCONFIGDIR=str(
                    Path(os.environ.get("TMPDIR", "/tmp")) / "structured-request-matplotlib"
                ),
                SRA_BACKEND_MODULE=self.backend_module,
                SRA_REQUEST_PATH=str(worker.request_path),
                SRA_RESULT_PATH=str(worker.result_path),
            )
            if worker.visible is not None:
                env["CUDA_VISIBLE_DEVICES"] = worker.visible
            with worker.log_path.open("ab", buffering=0) as log:
                worker.process = subprocess.Popen(
                    [str(self.python), "-u", "-c", _WORKER_CODE],
                    cwd=self.root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(lease.fileno() for lease in leases),
                )
            worker.started = time.monotonic()
            _atomic_json(worker.result_path.parent / "process.json", worker.status())
            print(f"Started {worker.label} (PID {worker.process.pid}). Log: {worker.log_path}")
        except BaseException:
            worker.cancelled = True
            if worker.process is not None and worker.process.poll() is None:
                try:
                    self.cancel(worker)
                except Exception:
                    pass  # Cancellation happens before its best-effort status write.
            else:
                _atomic_json(worker.result_path.parent / "process.json", worker.status())
            raise
        finally:
            for lease in leases:
                lease.close()  # The child keeps inherited leases until it exits.

    def inspect_hardware(self) -> dict[str, Any]:
        """Discover devices in a fresh project interpreter; no models are loaded."""
        return self.run_action("inspect_hardware")

    def _plan(self) -> dict[str, Any]:
        if self.hardware is None:
            raise RuntimeError("Run inspect_hardware() before creating or opening an experiment")
        return self.hardware

    def _assignment(self, arm: str) -> tuple[str, str]:
        plan = self._plan()
        if plan["device"] == "cuda":
            mask = plan["arm_masks"][arm]
            return f"cuda_{mask}", mask
        return plan["device"], ""

    def run_action(self, action: str, run_name: str | None = None, **kwargs: Any) -> dict:
        """Execute one explicit preparation, inspection, freeze or saved-report action."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", action):
            raise ValueError("Invalid action name")
        if any(worker.status()["status"] in {"running", "queued"} for worker in self._workers):
            raise RuntimeError("Wait for or cancel pending arm workers before coordinator actions")
        request = {"action": action, **kwargs}
        if run_name is not None:
            request["run_name"] = run_name
        if action == "inspect_hardware":
            resource, visible = None, None
        else:
            request["hardware"] = self._plan()
            arm = "qlora" if action in {"prepare_quantized_base", "preflight_qlora"} else "lora"
            if action == "predict":
                arm = kwargs.get("arm", "qlora")
                if arm not in {"lora", "qlora"}:
                    raise ValueError("Choose a named adapter arm")
            resource, visible = self._assignment(arm)
            if action in {"create_run", "resume_session"}:
                resource, visible = None, None  # Retain the host's allocation visibility.
        worker = self._new_worker(
            request, label=action, resource=resource, visible=visible, coordinator=True
        )
        result = self.wait([worker])[action]
        if action == "inspect_hardware":
            from src.hardware import validate_hardware_plan

            validate_hardware_plan(result)
            self.hardware = result
        return result

    def start_arm(self, run_name: str, arm: str, phase: str = "training") -> Worker:
        if arm not in {"lora", "qlora"}:
            raise ValueError("arm must be 'lora' or 'qlora'")
        if phase not in {"baseline", "training", "test", "benchmark", "evaluation", "pilot"}:
            raise ValueError("Unknown arm phase")
        if not re.fullmatch(r"run_[0-9]{3,}", run_name):
            raise ValueError("Select an explicitly named run_XXX")
        plan = self._plan()
        pending = [
            worker for worker in self._workers if worker.status()["status"] in {"running", "queued"}
        ]
        if any(worker.label == arm for worker in pending):
            raise RuntimeError(f"The {arm} arm already has a pending worker")
        if any(worker.coordinator for worker in pending):
            raise RuntimeError("A coordinator action is already running")
        predecessor = pending[-1] if pending and not plan["parallel"] else None
        resource, visible = self._assignment(arm)
        return self._new_worker(
            {
                "action": "run_arm",
                "run_name": run_name,
                "arm": arm,
                "phase": phase,
                "hardware": plan,
            },
            label=arm,
            resource=resource,
            visible=visible,
            predecessor=predecessor,
        )

    @staticmethod
    def _tail(path: Path, size: int = 1500) -> str:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - size))
            lines = stream.read().decode(errors="replace").replace("\r", "\n").splitlines()
        return "\n".join(lines[-4:])

    def _result(self, worker: Worker) -> dict:
        if worker.cancelled or worker.process is None or worker.process.poll() != 0:
            raise RuntimeError(
                f"{worker.label} failed or was cancelled. See {worker.log_path}\n{self._tail(worker.log_path)}"
            )
        if not worker.result_path.is_file():
            raise RuntimeError(
                f"{worker.label} exited without a committed result: {worker.log_path}"
            )
        value = json.loads(worker.result_path.read_text())
        if not value.get("ok"):
            raise RuntimeError(f"{worker.label}: {value}")
        return value["result"]

    def wait(self, workers: Sequence[Worker], poll_seconds: float = 1.0) -> dict[str, Any]:
        if not workers or not 0 < poll_seconds <= 30:
            raise ValueError("Supply workers and a polling interval in (0, 30] seconds")
        if len({worker.label for worker in workers}) != len(workers):
            raise ValueError("Worker labels must be distinct")
        if any(not any(worker is owned for owned in self._workers) for worker in workers):
            raise ValueError("Wait accepts only this controller's workers")
        watched = list(workers)
        for worker in list(watched):
            predecessor = worker.predecessor
            while predecessor is not None:
                if not any(predecessor is item for item in watched):
                    watched.append(predecessor)
                predecessor = predecessor.predecessor
        next_update = 0.0
        try:
            while True:
                for worker in watched:
                    if worker.status()["status"] in {"failed", "cancelled"}:
                        self._result(worker)
                    if worker.process is None and not worker.cancelled:
                        predecessor = worker.predecessor
                        if predecessor is None or predecessor.status()["status"] == "complete":
                            if predecessor is not None:
                                self._result(predecessor)  # Require its committed success first.
                            self._spawn(worker)
                states = [worker.status() for worker in watched]
                if all(state["status"] == "complete" for state in states):
                    break
                if time.monotonic() >= next_update:
                    for worker, state in zip(watched, states, strict=True):
                        print(f"{worker.label}: {state['status']}, {state['elapsed_seconds']:.0f}s")
                        tail = self._tail(worker.log_path)
                        if tail:
                            print(tail)
                    next_update = time.monotonic() + 30
                time.sleep(poll_seconds)
            result = {worker.label: self._result(worker) for worker in workers}
            for worker in watched:
                _atomic_json(worker.result_path.parent / "process.json", worker.status())
            return result
        except BaseException:
            self._cancel_many(watched, suppress_errors=True)
            raise

    def status(self) -> list[dict[str, Any]]:
        """Read state only; this never starts queued work."""
        return [worker.status() for worker in self._workers]

    def cancel(self, worker: Worker) -> None:
        if not any(worker is owned for owned in self._workers):
            raise ValueError("Cannot cancel another controller's worker")
        if worker.process is None:
            worker.cancelled = True
        elif worker.process.poll() is None:
            worker.cancelled = True
            for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 4)):
                try:
                    os.killpg(worker.process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    worker.process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
        _atomic_json(worker.result_path.parent / "process.json", worker.status())

    def _cancel_many(self, workers: Sequence[Worker], *, suppress_errors: bool) -> None:
        first_error = None
        for worker in workers:
            try:
                self.cancel(worker)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not suppress_errors:
            raise first_error

    def _shutdown(self) -> None:
        self._cancel_many(self._workers, suppress_errors=True)

    def cancel_all(self) -> None:
        """Attempt every cancellation, then report any bookkeeping or process error."""
        self._cancel_many(self._workers, suppress_errors=False)
