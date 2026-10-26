"""Explicit MASSIVE acquisition, lossless targets and local-only tokenization."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.request import Request, urlopen

from .config import SPLITS
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

SOURCE_PARTITIONS = {"train": "train", "dev": "validation", "test": "test"}
ANNOTATION = re.compile(r"\[([A-Za-z_][A-Za-z_0-9]*) : ([^\[\]]+)\]")
DATA_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "ontology.json",
    "duplicate_audit.json",
)
TOKEN_FILES = (
    "system_prompt.txt",
    "tokenized_train.jsonl",
    "tokenized_validation.jsonl",
    "tokenized_test.jsonl",
)
DEVELOPMENT_SELECTION = "development_selection.json"
DEVELOPMENT_SEED_OFFSETS = {"train": 104729, "validation": 130363}


def _prepared_file_names(run: RunContext) -> tuple[str, ...]:
    if run.config.preset == "development":
        return (*DATA_FILES, DEVELOPMENT_SELECTION)
    return DATA_FILES


def development_selection(config, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Choose by ID position alone; keep every official on-disk row unchanged."""
    config.validate()
    if config.preset != "development":
        raise ValueError("A development selection requires the development preset")
    selection = {
        "schema_version": 1,
        "preset": "development",
        "config_sha256": canonical_hash(config.to_dict()),
        "seed": config.seed,
        "policy": "Uniform sampling without replacement by split seed; retain official row order",
        "splits": {},
    }
    for split, offset in DEVELOPMENT_SEED_OFFSETS.items():
        ids = [row["id"] for row in splits[split]]
        official_count = config.expected_split_counts[SPLITS.index(split)]
        limit = getattr(config, f"{split}_example_limit")
        if len(ids) != official_count or len(set(ids)) != official_count:
            raise ValueError(f"Development sampling requires the intact official {split} IDs")
        positions = sorted(random.Random(config.seed + offset).sample(range(len(ids)), limit))
        selected_ids = [ids[position] for position in positions]
        selection["splits"][split] = {
            "official_count": official_count,
            "source_ids_sha256": canonical_hash(ids),
            "split_seed": config.seed + offset,
            "selected_count": limit,
            "ids": selected_ids,
            "ids_sha256": canonical_hash(selected_ids),
        }
    return selection


def _registered(run: RunContext) -> dict[str, Any]:
    return read_json(run.path / "manifest.json")["artifacts"]


def _verify_registered(run: RunContext, paths: list[Path]) -> None:
    registered = _registered(run)
    relative = [str(path.relative_to(run.path)) for path in paths]
    missing = [name for name in relative if name not in registered]
    if missing:
        raise ValueError(f"Required data artifacts were not committed: {missing}")
    run.verify_artifacts(prefixes=relative)


def _resume_completed_stage(
    run: RunContext, stage: str, summary_path: Path, files: list[Path]
) -> dict[str, Any] | None:
    relative = str(summary_path.relative_to(run.path))
    if relative not in _registered(run):
        # An unregistered summary is only a partial write, never an identity record.
        return None
    _verify_registered(run, [*files, summary_path])
    summary = read_json(summary_path)
    if summary.get("files") != {path.name: sha256_file(path) for path in files}:
        raise ValueError("Completed data-stage summary disagrees with its artifacts")
    stage_path = run.path / "stages" / f"{stage}.json"
    if not stage_path.is_file() or read_json(stage_path).get("status") != "complete":
        # The artifact commit succeeded but the process stopped before status commit.
        with run.stage(stage):
            pass
    return summary


def normalized_utterance(text: str) -> str:
    """Audit key only: never apply this normalization to inputs or target values."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_annotated_utterance(utterance: str, annotated: str) -> list[dict[str, str]]:
    """Remove markup's single separator spaces, preserving the actual span value."""
    slots: list[dict[str, str]] = []
    reconstructed: list[str] = []
    cursor = 0
    for match in ANNOTATION.finditer(annotated):
        preceding = annotated[cursor : match.start()]
        if "[" in preceding or "]" in preceding:
            raise ValueError("Unparsed or nested slot annotation")
        reconstructed.extend((preceding, match.group(2)))
        slots.append({"type": match.group(1), "value": match.group(2)})
        cursor = match.end()
    tail = annotated[cursor:]
    if "[" in tail or "]" in tail:
        raise ValueError("Unparsed or nested slot annotation")
    reconstructed.append(tail)
    if "".join(reconstructed) != utterance:
        raise ValueError("Annotation removal does not exactly reconstruct the source utterance")
    return slots


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or "\\" in member.name:
        raise ValueError(f"Unsafe archive path: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"Unsupported archive member type: {member.name}")
    if member.size < 0 or member.size > 512 * 1024 * 1024:
        raise ValueError(f"Unexpected archive member size: {member.name}")


def _extract_english(archive: Path, destination: Path, locale: str) -> list[Path]:
    """Inspect every archive entry and copy only English plus its data license."""
    with tarfile.open(archive, mode="r:gz") as source:
        members = source.getmembers()
        for member in members:
            _validate_member(member)
        selected: dict[str, list[tarfile.TarInfo]] = {f"{locale}.jsonl": [], "LICENSE": []}
        for member in members:
            basename = PurePosixPath(member.name).name
            if member.isfile() and basename in selected:
                selected[basename].append(member)
        if any(len(matches) != 1 for matches in selected.values()):
            raise ValueError("Expected exactly one English JSONL file and one dataset LICENSE")
        written = []
        for name, matches in selected.items():
            member = matches[0]
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            target = destination / name
            with stream, target.open("wb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            if target.stat().st_size != member.size:
                raise ValueError(f"Truncated archive member: {member.name}")
            written.append(target)
    return written


def dataset_cache_dir(run: RunContext) -> Path:
    """One versioned raw dataset shared by all runs in this project."""
    root = run.project_root.resolve()
    directory = root / "data" / "massive-1.1" / run.config.locale
    for path in (root / "data", directory.parent, directory):
        if path.is_symlink():
            raise ValueError("Shared dataset paths must be ordinary project directories")
    return directory


def _verify_cache(run: RunContext, directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Shared dataset is incomplete: its manifest is missing")
    summary = read_json(manifest_path)
    expected = {
        "dataset": "MASSIVE",
        "version": "1.1",
        "locale": run.config.locale,
        "source_url": run.config.dataset_url,
        "license": "CC-BY-4.0",
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Shared dataset provenance differs from this run's configuration")
    names = {"archive.tar.gz", f"{run.config.locale}.jsonl", "LICENSE"}
    if set(summary.get("files", {})) != names:
        raise ValueError("Shared dataset manifest must describe exactly the required files")
    for name, digest in summary["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Missing or changed shared dataset artifact: {name}")
    return summary


def _dataset_reference(run: RunContext, directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        **summary,
        "storage": "project_shared",
        "project_relative_path": str(directory.relative_to(run.project_root.resolve())),
        "cache_manifest_sha256": sha256_file(directory / "manifest.json"),
    }


def verify_shared_dataset(run: RunContext) -> Path:
    """Verify the immutable run reference and every shared payload, without recursion."""
    reference_path = run.path / "inputs" / "dataset" / "manifest.json"
    relative = str(reference_path.relative_to(run.path))
    registered = _registered(run).get(relative)
    if (
        registered is None
        or reference_path.is_symlink()
        or not reference_path.is_file()
        or sha256_file(reference_path) != registered["sha256"]
    ):
        raise ValueError("Shared dataset reference is missing, changed or uncommitted")
    reference = read_json(reference_path)
    directory = dataset_cache_dir(run)
    summary = _verify_cache(run, directory)
    if reference != _dataset_reference(run, directory, summary):
        raise ValueError("Shared dataset identity no longer matches the run reference")
    return directory


def _download_shared_dataset(run: RunContext, directory: Path) -> None:
    """Publish a complete cache directory atomically; caller holds the project lock."""
    from tqdm.auto import tqdm

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run.config.locale}-pending-", dir=directory.parent)
    )
    try:
        archive = temporary / "archive.tar.gz"
        request = Request(
            run.config.dataset_url, headers={"User-Agent": "structured-request-adaptation/1.0"}
        )
        with archive.open("wb") as output, urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
            with tqdm(total=total, unit="B", unit_scale=True, desc="MASSIVE archive") as progress:
                received = 0
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > 2 * 1024**3:
                        raise ValueError("Dataset archive exceeded the 2 GiB safety bound")
                    output.write(block)
                    progress.update(len(block))
            if total is not None and received != total:
                raise ValueError("Incomplete dataset download")
            output.flush()
            os.fsync(output.fileno())
        _extract_english(archive, temporary, run.config.locale)
        files = [
            temporary / name for name in ("archive.tar.gz", f"{run.config.locale}.jsonl", "LICENSE")
        ]
        summary = {
            "dataset": "MASSIVE",
            "version": "1.1",
            "locale": run.config.locale,
            "source_url": run.config.dataset_url,
            "license": "CC-BY-4.0",
            "license_source": "https://github.com/alexa/massive/blob/main/NOTICE.md",
            "checksum_policy": "SHA256 recorded at first HTTPS acquisition; no upstream checksum claimed",
            "files": {path.name: sha256_file(path) for path in files},
        }
        atomic_json(temporary / "manifest.json", summary)
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def acquire_dataset(run: RunContext) -> dict[str, Any]:
    """Download once per project; each run pins and verifies the shared raw files."""
    directory = dataset_cache_dir(run)
    reference_path = run.path / "inputs" / "dataset" / "manifest.json"
    relative = str(reference_path.relative_to(run.path))
    with run.writer_lock():
        run.verify_identity()
        if relative in _registered(run):
            verify_shared_dataset(run)
            stage_path = run.path / "stages" / "dataset.json"
            if not stage_path.is_file() or read_json(stage_path).get("status") != "complete":
                with run.stage("dataset"):
                    pass
            return read_json(reference_path)
        with run.stage("dataset"):
            if not directory.exists():
                _download_shared_dataset(run, directory)
            summary = _verify_cache(run, directory)
            reference = _dataset_reference(run, directory, summary)
            atomic_json(reference_path, reference)
            run.register_artifacts([reference_path])
            return reference


def _duplicate_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for name, normalize in (("exact", lambda text: text), ("normalized", normalized_utterance)):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for split, rows in splits.items():
            for row in rows:
                groups[normalize(row["utterance"])].append(
                    {
                        "split": split,
                        "id": row["id"],
                        "target_hash": canonical_hash(row["target"]),
                    }
                )
        duplicates = [
            {
                "utterance_hash": canonical_hash(text),
                "examples": entries,
                "cross_split": len({entry["split"] for entry in entries}) > 1,
                "conflicting_annotations": len({entry["target_hash"] for entry in entries}) > 1,
            }
            for text, entries in sorted(groups.items())
            if len(entries) > 1
        ]
        audit[name] = {
            "duplicate_group_count": len(duplicates),
            "cross_split_group_count": sum(group["cross_split"] for group in duplicates),
            "conflicting_group_count": sum(
                group["conflicting_annotations"] for group in duplicates
            ),
            "groups": duplicates,
        }
    return audit


def build_prepared_rows(
    records: list[dict[str, Any]], expected_counts: tuple[int, int, int] = (11514, 2033, 2974)
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], dict[str, Any]]:
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    for record in records:
        if record.get("locale") != "en-US" or record.get("partition") not in SOURCE_PARTITIONS:
            raise ValueError("Unexpected locale or partition in English dataset")
        identifier = str(record["id"])
        if identifier in seen:
            raise ValueError(f"Repeated source ID: {identifier}")
        seen.add(identifier)
        utterance, annotated, intent = record["utt"], record["annot_utt"], record["intent"]
        if not all(isinstance(value, str) and value for value in (utterance, annotated, intent)):
            raise ValueError(f"Invalid source fields for {identifier}")
        try:
            target = {"intent": intent, "slots": parse_annotated_utterance(utterance, annotated)}
        except ValueError as exc:
            raise ValueError(f"Invalid annotation for source ID {identifier}: {exc}") from exc
        splits[SOURCE_PARTITIONS[record["partition"]]].append(
            {
                "id": identifier,
                "utterance": utterance,
                "target": target,
                "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                "novel_input": False,
            }
        )
    observed = tuple(len(splits[split]) for split in SPLITS)
    if observed != tuple(expected_counts):
        raise ValueError(f"Unexpected split counts: {observed}; expected {expected_counts}")
    ontology = {
        "intents": sorted({row["target"]["intent"] for row in splits["train"]}),
        "slots": sorted(
            {slot["type"] for row in splits["train"] for slot in row["target"]["slots"]}
        ),
    }
    train_keys = {normalized_utterance(row["utterance"]) for row in splits["train"]}
    dev_keys = {normalized_utterance(row["utterance"]) for row in splits["validation"]}
    for split, rows in splits.items():
        for row in rows:
            if row["target"]["intent"] not in ontology["intents"] or any(
                slot["type"] not in ontology["slots"] for slot in row["target"]["slots"]
            ):
                raise ValueError(f"Held-out label not present in training ontology: {row['id']}")
            seen_keys = train_keys | dev_keys if split == "test" else train_keys
            row["novel_input"] = (
                split != "train" and normalized_utterance(row["utterance"]) not in seen_keys
            )
    return splits, ontology, _duplicate_audit(splits)


def prepare_data(run: RunContext) -> dict[str, Any]:
    directory = run.path / "preparation"
    summary_path = directory / "data.json"
    data_paths = [directory / name for name in _prepared_file_names(run)]
    with run.writer_lock():
        run.verify_identity()
        run.require_stage("dataset")
        dataset_directory = verify_shared_dataset(run)
        run.verify_artifacts(prefixes=[str(path.relative_to(run.path)) for path in data_paths])
        completed = _resume_completed_stage(run, "data", summary_path, data_paths)
        if completed is not None:
            return completed
        with run.stage("data"):
            records = read_jsonl(dataset_directory / f"{run.config.locale}.jsonl")
            splits, ontology, duplicates = build_prepared_rows(
                records, run.config.expected_split_counts
            )
            if len(ontology["intents"]) != 60 or len(ontology["slots"]) != 55:
                raise ValueError(
                    "Training-derived ontology must contain exactly 60 intents and 55 slot types"
                )
            directory.mkdir(parents=True, exist_ok=True)
            paths = []
            for split, rows in splits.items():
                path = directory / f"{split}.jsonl"
                write_jsonl(path, rows)
                paths.append(path)
            ontology_path = directory / "ontology.json"
            duplicate_path = directory / "duplicate_audit.json"
            atomic_json(ontology_path, ontology)
            atomic_json(duplicate_path, duplicates)
            paths.extend((ontology_path, duplicate_path))
            if run.config.preset == "development":
                selection_path = directory / DEVELOPMENT_SELECTION
                atomic_json(selection_path, development_selection(run.config, splits))
                paths.append(selection_path)
            summary = {
                "split_counts": {split: len(rows) for split, rows in splits.items()},
                "novel_input_counts": {
                    split: sum(row["novel_input"] for row in rows) for split, rows in splits.items()
                },
                "ontology_source": "Training annotations only; held-out labels checked for membership",
                "split_policy": "All official partitions and rows retained; no resplitting or deduplication",
                "preset": run.config.preset,
                "consumption_policy": (
                    "Registered development train/validation selection; test consumption forbidden"
                    if run.config.preset == "development"
                    else "Complete official partitions"
                ),
                "novel_input_policy": "Test normalized utterance absent from both train and validation; dev absent from train",
                "audit_normalization": "NFKC, casefold, collapse whitespace; audit only",
                "duplicate_audit_counts": {
                    mode: {key: value for key, value in audit.items() if key != "groups"}
                    for mode, audit in duplicates.items()
                },
                "files": {path.name: sha256_file(path) for path in paths},
            }
            run.register_artifacts(paths)
            atomic_json(summary_path, summary)
            run.register_artifacts([summary_path])
            return summary


def build_system_prompt(ontology: dict[str, list[str]]) -> str:
    return (
        'Parse the user\'s English request. Return exactly one JSON object with keys "intent" and "slots". '
        'The intent must be one allowed intent. Slots must be a list of objects with exactly "type" and "value". '
        "Use only allowed slot types. Copy each value exactly from the request, preserving case and whitespace. "
        "Keep every occurrence, including repeated slot types or values, in request order. Use [] when no slots occur. "
        "Do not add keys, explanations, markdown, or code fences.\n"
        "Allowed intents: " + ", ".join(ontology["intents"]) + "\n"
        "Allowed slot types: " + ", ".join(ontology["slots"])
    )


def prompt_messages(
    utterance: str, intents: list[str], slot_types: list[str]
) -> list[dict[str, str]]:
    """One prompt definition for tokenization and standalone inference."""
    return [
        {
            "role": "system",
            "content": build_system_prompt({"intents": intents, "slots": slot_types}),
        },
        {"role": "user", "content": utterance},
    ]


def chat_token_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool
) -> list[int]:
    """Normalize Transformers' list and BatchEncoding chat-template return forms."""
    from collections.abc import Mapping

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    ids = list(encoded)
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Expected one flat nonempty sequence of integer chat token IDs")
    return ids


def tokenize_example(
    row: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_sequence_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["utterance"]},
    ]
    prompt_ids = chat_token_ids(tokenizer, messages, add_generation_prompt=True)
    full_ids = chat_token_ids(
        tokenizer,
        [*messages, {"role": "assistant", "content": row["target_text"]}],
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Chat-template training/prompt prefix mismatch for {row['id']}")
    suffix = full_ids[len(prompt_ids) :]
    eos = tokenizer.eos_token_id
    if eos is None or suffix.count(eos) != 1:
        raise ValueError(f"Expected exactly one assistant EOS for {row['id']}")
    supervised_length = suffix.index(eos) + 1
    trailing = suffix[supervised_length:]
    if tokenizer.decode(
        trailing, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ).strip():
        raise ValueError(f"Unexpected non-whitespace chat-template suffix for {row['id']}")
    target_ids = suffix[: supervised_length - 1]
    if (
        tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        != row["target_text"]
    ):
        raise ValueError(f"Target token decoding is not lossless for {row['id']}")
    if (
        len(full_ids) > max_sequence_length
        or len(prompt_ids) + max_new_tokens > max_sequence_length
        or supervised_length > max_new_tokens
    ):
        raise ValueError(
            f"Token budget exceeded for {row['id']}: prompt={len(prompt_ids)}, "
            f"target_with_eos={supervised_length}, full={len(full_ids)}; "
            "nothing was truncated; change configuration in a fresh run"
        )
    labels = [-100] * len(prompt_ids) + suffix[:supervised_length] + [-100] * len(trailing)
    if (
        len(labels) != len(full_ids)
        or sum(value != -100 for value in labels[1:]) != supervised_length
    ):
        raise ValueError("Assistant-only causal loss masking invariant failed")
    return {
        **row,
        "input_ids": full_ids,
        "labels": labels,
        "prompt_ids": prompt_ids,
        "target_tokens": supervised_length,
    }


def _length_summary(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def tokenize_data(run: RunContext) -> dict[str, Any]:
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    directory = run.path / "preparation"
    summary_path = directory / "tokenization.json"
    token_paths = [directory / name for name in TOKEN_FILES]
    with run.writer_lock():
        run.verify_identity()
        run.require_stage("data")
        run.require_stage("acquire_model")
        _verify_registered(
            run, [directory / name for name in (*_prepared_file_names(run), "data.json")]
        )
        tokenizer_files = [
            path for path in (run.path / "inputs" / "tokenizer").rglob("*") if path.is_file()
        ]
        if not tokenizer_files:
            raise ValueError("No committed tokenizer snapshot is available")
        _verify_registered(run, [*tokenizer_files, run.path / "inputs" / "model_snapshot.json"])
        run.verify_artifacts(
            prefixes=[
                "inputs/tokenizer",
                *[str(path.relative_to(run.path)) for path in token_paths],
            ]
        )
        completed = _resume_completed_stage(run, "tokenization", summary_path, token_paths)
        if completed is not None:
            return completed
        with run.stage("tokenization"):
            tokenizer = AutoTokenizer.from_pretrained(
                run.path / "inputs" / "tokenizer", local_files_only=True, trust_remote_code=False
            )
            ontology = load_ontology(run)
            system_prompt = build_system_prompt(ontology)
            prompt_path = directory / "system_prompt.txt"
            atomic_text(prompt_path, system_prompt)
            paths = [prompt_path]
            summary: dict[str, Any] = {
                "max_sequence_length": run.config.max_sequence_length,
                "max_new_tokens": run.config.max_new_tokens,
                "truncation": False,
                "loss_mask": "Assistant JSON and EOS only; prompt/padding/template whitespace ignored",
                "splits": {},
                "chat_template_sha256": canonical_hash(tokenizer.chat_template),
            }
            for split in SPLITS:
                rows = read_jsonl(directory / f"{split}.jsonl")
                tokenized = [
                    tokenize_example(
                        row,
                        tokenizer,
                        system_prompt,
                        run.config.max_sequence_length,
                        run.config.max_new_tokens,
                    )
                    for row in tqdm(rows, desc=f"Tokenize {split}", unit="request")
                ]
                path = directory / f"tokenized_{split}.jsonl"
                write_jsonl(path, tokenized)
                paths.append(path)
                summary["splits"][split] = {
                    "examples": len(tokenized),
                    "prompt_tokens": _length_summary([len(row["prompt_ids"]) for row in tokenized]),
                    "target_tokens": _length_summary([row["target_tokens"] for row in tokenized]),
                    "full_tokens": _length_summary([len(row["input_ids"]) for row in tokenized]),
                    "total_target_tokens": sum(row["target_tokens"] for row in tokenized),
                    "total_input_tokens": sum(len(row["input_ids"]) for row in tokenized),
                }
            summary["files"] = {path.name: sha256_file(path) for path in paths}
            run.register_artifacts(paths)
            atomic_json(summary_path, summary)
            run.register_artifacts([summary_path])
            return summary


def load_examples(run: RunContext, split: str, tokenized: bool = True) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if run.config.preset == "development" and split == "test":
        raise RuntimeError("The development preset cannot consume the held-out test partition")
    run.verify_identity()
    run.require_stage("tokenization" if tokenized else "data")
    prefix = "tokenized_" if tokenized else ""
    path = run.path / "preparation" / f"{prefix}{split}.jsonl"
    _verify_registered(
        run, [path, run.path / "preparation" / ("tokenization.json" if tokenized else "data.json")]
    )
    rows = read_jsonl(path)
    if run.config.preset != "development":
        return rows
    selection_path = run.path / "preparation" / DEVELOPMENT_SELECTION
    raw_paths = {
        name: run.path / "preparation" / f"{name}.jsonl" for name in DEVELOPMENT_SEED_OFFSETS
    }
    _verify_registered(
        run,
        [selection_path, run.path / "preparation" / "data.json", *raw_paths.values()],
    )
    prepared = {name: read_jsonl(raw_path) for name, raw_path in raw_paths.items()}
    selection = read_json(selection_path)
    if selection != development_selection(run.config, prepared):
        raise ValueError("Development selection disagrees with the frozen configuration/source IDs")
    if [row["id"] for row in rows] != [row["id"] for row in prepared[split]]:
        raise ValueError("Tokenized development source IDs differ from the complete official split")
    chosen_ids = set(selection["splits"][split]["ids"])
    return [row for row in rows if row["id"] in chosen_ids]


def load_ontology(run: RunContext) -> dict[str, list[str]]:
    run.verify_identity()
    run.require_stage("data")
    path = run.path / "preparation" / "ontology.json"
    _verify_registered(run, [path, run.path / "preparation" / "data.json"])
    return read_json(path)


def inspect_example(run: RunContext, index: int = 0) -> dict[str, Any]:
    rows = load_examples(run, "train")
    if not 0 <= index < len(rows):
        raise IndexError("Training example index is out of bounds")
    row = rows[index]
    return {
        "id": row["id"],
        "utterance": row["utterance"],
        "target": row["target"],
        "prompt_tokens": len(row["prompt_ids"]),
        "full_tokens": len(row["input_ids"]),
        "supervised_tokens_including_eos": row["target_tokens"],
        "ignored_tokens": Counter(row["labels"])[-100],
    }
