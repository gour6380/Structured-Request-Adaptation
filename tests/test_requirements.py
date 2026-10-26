"""Direct dependencies preserve compatible host CUDA PyTorch and native MPS support."""

from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def _active_requirements(system, version):
    environment = {**default_environment(), "sys_platform": system, "python_version": version}
    active = {}
    for line in REQUIREMENTS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            assert requirement.name not in active, "Ambiguous platform-specific dependencies"
            active[requirement.name] = requirement
    return active


@pytest.mark.parametrize("version", ["3.12", "3.13"])
@pytest.mark.parametrize("system", ["darwin", "linux"])
def test_platform_markers_keep_core_dependencies_and_native_backend(system, version):
    active = _active_requirements(system, version)
    expected = {
        "transformers": "5.17.0",
        "peft": "0.21.0",
        "bitsandbytes": "0.50.2",
        "accelerate": "1.15.0",
        "huggingface-hub": "1.32.0",
        "safetensors": "0.8.0",
        "tensorboard": "2.21.0",
    }
    for name, pin in expected.items():
        assert str(active[name].specifier) == f"=={pin}"
    assert {"tqdm", "numpy", "matplotlib", "psutil", "ipykernel", "nbformat"} <= active.keys()
    assert ("kernels" in active) == (system == "darwin")
    if system == "darwin":
        assert str(active["torch"].specifier) == "==2.14.0"
        assert str(active["kernels"].specifier) == "==0.17.1"


def test_linux_keeps_compatible_preinstalled_cuda_builds():
    torch = _active_requirements("linux", "3.12")["torch"]
    assert torch.url is None
    for version in ("2.8.0+cu128", "2.14.0+cu126"):
        assert version in torch.specifier
    assert "2.7.1" not in torch.specifier
    assert "3.0.0" not in torch.specifier
