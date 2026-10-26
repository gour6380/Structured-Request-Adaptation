"""Shared dataset acquisition contracts using local tar fixtures, never a network."""

import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from src import data
from src.config import ExperimentConfig
from src.io import atomic_json, read_json, sha256_file
from src.runs import create_run


def archive_bytes(*, include_english=True):
    english = (
        json.dumps(
            {
                "id": "0",
                "locale": "en-US",
                "partition": "train",
                "intent": "general_greet",
                "utt": "hello",
                "annot_utt": "hello",
            }
        ).encode()
        + b"\n"
    )
    members = {"1.1/LICENSE": b"Dataset fixture license\n", "1.1/data/de-DE.jsonl": b"{}\n"}
    if include_english:
        members["1.1/data/en-US.jsonl"] = english
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return stream.getvalue()


class ArchiveResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text('"""Test fixture."""\n')
    (tmp_path / "requirements.txt").write_text("test-fixture==1\n")
    monkeypatch.setattr(
        "src.runs.environment_record", lambda: {"device": "cpu", "compute_dtype": "float32"}
    )
    return tmp_path


@pytest.fixture
def download(monkeypatch):
    calls = []
    payload = archive_bytes()

    def open_archive(request, *, timeout):
        assert request.full_url == ExperimentConfig().dataset_url
        assert timeout > 0
        calls.append(request.full_url)
        return ArchiveResponse(payload)

    monkeypatch.setattr(data, "urlopen", open_archive)
    return calls


def new_run(project):
    return create_run(project, ExperimentConfig(model_revision="a" * 40))


def test_two_runs_reuse_one_download_and_store_only_references(project, download):
    first, second = new_run(project), new_run(project)
    first_summary = data.acquire_dataset(first)
    second_summary = data.acquire_dataset(second)
    cache = project / "data" / "massive-1.1" / "en-US"
    assert len(download) == 1
    assert data.dataset_cache_dir(first) == data.dataset_cache_dir(second) == cache
    assert set(path.name for path in cache.iterdir()) == {
        "archive.tar.gz",
        "en-US.jsonl",
        "LICENSE",
        "manifest.json",
    }
    for run, summary in ((first, first_summary), (second, second_summary)):
        assert summary["storage"] == "project_shared"
        assert summary["project_relative_path"] == "data/massive-1.1/en-US"
        assert summary["cache_manifest_sha256"] == sha256_file(cache / "manifest.json")
        assert summary["files"] == {
            name: sha256_file(cache / name) for name in ("archive.tar.gz", "en-US.jsonl", "LICENSE")
        }
        assert set(path.name for path in (run.path / "inputs" / "dataset").iterdir()) == {
            "manifest.json"
        }
        assert data.verify_shared_dataset(run) == cache
        assert run.require_stage("dataset")["status"] == "complete"
        run.verify_artifacts()
    assert data.acquire_dataset(first) == first_summary
    assert len(download) == 1


@pytest.mark.parametrize("filename", ["archive.tar.gz", "en-US.jsonl", "LICENSE", "manifest.json"])
def test_corrupt_cache_is_rejected_without_redownload(project, download, filename):
    first = new_run(project)
    data.acquire_dataset(first)
    (data.dataset_cache_dir(first) / filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        data.verify_shared_dataset(first)
    with pytest.raises(ValueError):
        first.verify_artifacts(prefixes=["preparation"])
    second = new_run(project)
    with pytest.raises(ValueError):
        data.acquire_dataset(second)
    assert len(download) == 1


def test_missing_committed_payload_is_rejected_without_redownload(project, download):
    first = new_run(project)
    data.acquire_dataset(first)
    (data.dataset_cache_dir(first) / "en-US.jsonl").unlink()
    with pytest.raises(ValueError):
        data.acquire_dataset(new_run(project))
    assert len(download) == 1


def test_modified_run_reference_is_rejected(project, download):
    run = new_run(project)
    data.acquire_dataset(run)
    reference_path = run.path / "inputs" / "dataset" / "manifest.json"
    reference = read_json(reference_path)
    reference["cache_manifest_sha256"] = "0" * 64
    atomic_json(reference_path, reference)
    with pytest.raises(ValueError):
        data.verify_shared_dataset(run)
    with pytest.raises(ValueError):
        run.verify_artifacts(prefixes=["preparation"])
    assert len(download) == 1


def test_changed_storage_marker_cannot_bypass_shared_verification(project, download):
    run = new_run(project)
    data.acquire_dataset(run)
    reference_path = run.path / "inputs" / "dataset" / "manifest.json"
    reference = read_json(reference_path)
    reference.pop("storage")
    atomic_json(reference_path, reference)
    with pytest.raises(ValueError):
        run.verify_artifacts(prefixes=["preparation"])
    assert len(download) == 1


def test_rehashed_cache_drift_still_breaks_existing_run_reference(project, download):
    run = new_run(project)
    data.acquire_dataset(run)
    cache = data.dataset_cache_dir(run)
    (cache / "en-US.jsonl").write_text('{"different": true}\n')
    manifest = read_json(cache / "manifest.json")
    manifest["files"]["en-US.jsonl"] = sha256_file(cache / "en-US.jsonl")
    atomic_json(cache / "manifest.json", manifest)
    with pytest.raises(ValueError):
        data.verify_shared_dataset(run)
    with pytest.raises(ValueError):
        run.verify_artifacts(prefixes=["preparation"])
    assert len(download) == 1


def test_shared_payload_symlink_is_rejected(project, download):
    run = new_run(project)
    data.acquire_dataset(run)
    payload = data.dataset_cache_dir(run) / "LICENSE"
    outside = project / "other-license"
    outside.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(outside)
    with pytest.raises(ValueError):
        data.verify_shared_dataset(run)


def test_invalid_archive_never_publishes_shared_cache(project, monkeypatch):
    payload = archive_bytes(include_english=False)
    monkeypatch.setattr(data, "urlopen", lambda *_, **__: ArchiveResponse(payload))
    run = new_run(project)
    with pytest.raises(ValueError, match="English"):
        data.acquire_dataset(run)
    assert not data.dataset_cache_dir(run).exists()
    assert "inputs/dataset/manifest.json" not in read_json(run.path / "manifest.json")["artifacts"]


def test_project_lock_prevents_simultaneous_cache_writers(project, monkeypatch):
    first, second = new_run(project), new_run(project)
    entered, release = Event(), Event()
    calls = []

    def open_archive(*_, **__):
        calls.append(True)
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("Test did not release the mocked archive response")
        return ArchiveResponse(archive_bytes())

    monkeypatch.setattr(data, "urlopen", open_archive)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(data.acquire_dataset, first)
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match="Another notebook/process"):
                data.acquire_dataset(second)
        finally:
            release.set()
        future.result(timeout=10)
    data.acquire_dataset(second)
    assert len(calls) == 1
    assert data.verify_shared_dataset(first) == data.verify_shared_dataset(second)
