"""Explicit notebook cache setup avoids a dependency on an external drive."""

import sys
from types import SimpleNamespace

import pytest

from src.runtime import configure_hub_cache


def test_configures_project_cache_before_hub_import(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.setenv(key, "/unavailable/external/cache")
    paths = configure_hub_cache(tmp_path)
    assert paths["HF_HOME"] == str(tmp_path / ".hf-cache")
    assert paths["HF_HUB_CACHE"] == str(tmp_path / ".hf-cache/hub")
    assert paths["HF_XET_CACHE"] == str(tmp_path / ".hf-cache/xet")


def test_refuses_to_pretend_loaded_hub_cache_changed(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "huggingface_hub.constants", SimpleNamespace(HF_HOME="/external")
    )
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        configure_hub_cache(tmp_path)
