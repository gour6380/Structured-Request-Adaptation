"""Notebook contracts without installing dependencies or running GPU actions."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import nbformat

from src.config import ExperimentConfig
from src.hardware import _build_plan

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/lora_qlora.ipynb"


def _codes():
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _calls(source, name):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_executed_notebook_has_documented_action_cells_and_valid_saved_outputs():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert any(cell.get("outputs") for cell in notebook.cells)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        # A trailing blank cell and unset counters are preserved from the export.
        # Recorded outputs are evidence, not a requirement to rerun this notebook.
        if not cell.source.strip():
            assert cell.outputs == []
            continue
        assert all(output.output_type != "error" for output in cell.outputs)
        assert notebook.cells[index - 1].cell_type == "markdown"
        ast.parse(cell.source)
        assert "/Users/" not in cell.source
        assert "stage_project" not in cell.source
        assert not re.search(r"\bhf_[A-Za-z0-9]{20,}\b", cell.source)
        assert len(_calls(cell.source, "start_arm")) <= 1
        assert len(_calls(cell.source, "run_action")) <= 1 or "create_run" in cell.source
        assert not (_calls(cell.source, "start_arm") and _calls(cell.source, "wait"))


def test_install_uses_current_kernel_pip_without_custom_environment(tmp_path):
    codes = _codes()
    installer = next(code for code in codes if _calls(code, "check_call"))
    calls = []
    namespace = {
        "subprocess": SimpleNamespace(check_call=calls.append),
        "sys": SimpleNamespace(executable="/fixture/current-python"),
        "PROJECT_ROOT": tmp_path,
    }
    exec(compile(installer, str(NOTEBOOK), "exec"), namespace)
    assert calls == [
        [
            "/fixture/current-python",
            "-m",
            "pip",
            "install",
            "-r",
            str(tmp_path / "requirements.txt"),
        ]
    ]
    imports = next(i for i, code in enumerate(codes) if "from src.launcher import" in code)
    assert codes.index(installer) < imports
    source = "\n".join(codes)
    for obsolete in (
        "setup_environment",
        "src.environment",
        ".venv",
        "_check_storage",
        "ipykernel",
    ):
        assert obsolete not in source


def test_explicit_project_path_recovers_without_reading_deleted_working_directory(monkeypatch):
    def missing_directory(_cls):
        raise FileNotFoundError("kernel working directory was deleted")

    monkeypatch.setattr(Path, "cwd", classmethod(missing_directory))
    codes = _codes()
    location = next(code for code in codes if "PROJECT_ROOT = " in code)
    namespace = {"Path": Path}
    exec(compile(location, str(NOTEBOOK), "exec"), namespace)
    assert namespace["PROJECT_ROOT"].is_absolute()
    changed_directories = []
    namespace["os"] = SimpleNamespace(chdir=changed_directories.append)
    change = next(code for code in codes if _calls(code, "chdir"))
    exec(compile(change, str(NOTEBOOK), "exec"), namespace)
    assert changed_directories == [namespace["PROJECT_ROOT"]]
    assert codes.index(location) < codes.index(change)


def test_recorded_configuration_is_full_pinned_and_paired():
    code = next(c for c in _codes() if "config = ExperimentConfig.for_hardware(" in c)
    gpu = {
        "index": 0,
        "name": "CUDA fixture",
        "uuid": "GPU-test",
        "total_memory": 48 * 1024**3,
        "capability": [8, 9],
    }
    hardware = _build_plan(
        "cuda",
        {"0": gpu},
        {"0": True},
        {"system": "Linux", "machine": "x86_64", "cpu_model": "test", "memory_bytes": 64 * 1024**3},
    )
    namespace = {"ExperimentConfig": ExperimentConfig, "hardware": hardware}
    exec(compile(code, str(NOTEBOOK), "exec"), namespace)
    config = namespace["config"]
    assert config.preset == "full" and config.epochs == 5
    assert config.microbatch_size == 16 and config.gradient_accumulation == 1
    assert config.evaluation_batch_size == 32
    assert config.train_example_limit is None and config.validation_example_limit is None
    assert re.fullmatch(r"[0-9a-f]{40}", config.model_revision)


def test_notebook_documents_development_settings_without_replacing_recorded_code():
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert 'preset="development"' in markdown
    assert "128 training requests" in markdown
    assert "32 validation requests" in markdown
    assert "../results/run_001/report.md" in markdown
    assert "original source snapshot" in markdown


def test_pilot_and_training_barriers_precede_frozen_test():
    codes = _codes()
    positions = {}
    for phase in ("pilot", "baseline", "training", "test", "benchmark"):
        starts = []
        for i, code in enumerate(codes):
            for call in _calls(code, "start_arm"):
                values = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
                if values.get("phase") == phase:
                    starts.append((i, ast.literal_eval(call.args[1])))
        assert [arm for _, arm in starts] == ["lora", "qlora"]
        positions[phase] = [i for i, _ in starts]
    waits = [i for i, code in enumerate(codes) if _calls(code, "wait")]
    freeze = next(i for i, code in enumerate(codes) if '"freeze_evaluation"' in code)
    for before, after in (("pilot", "baseline"), ("baseline", "training")):
        assert any(max(positions[before]) < i < min(positions[after]) for i in waits)
    assert any(max(positions["training"]) < i < freeze for i in waits)
    assert freeze < min(positions["test"]) < min(positions["benchmark"])
