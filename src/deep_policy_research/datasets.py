from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .examples import ExamplePool, ExampleRecord
from .policy import SourcePointer
from .spec import TaskSpec


SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
SPLIT_PRECEDENCE = {"train": 0, "validation": 1, "test": 2}


def load_example_pool(
    spec: TaskSpec,
    *,
    version: str,
    data_format: str,
    train_path: Path | None = None,
    validation_path: Path | None = None,
    test_path: Path | None = None,
    validation_split_seed: str,
) -> ExamplePool:
    train_examples = _load_split_examples(
        train_path,
        split="train",
        data_format=data_format,
        label_space=spec.evaluate.label_space,
    )
    validation_examples = _load_split_examples(
        validation_path,
        split="validation",
        data_format=data_format,
        label_space=spec.evaluate.label_space,
    )
    test_examples = _load_split_examples(
        test_path,
        split="test",
        data_format=data_format,
        label_space=spec.evaluate.label_space,
    )

    if train_examples and not validation_examples:
        train_examples, validation_examples = _derive_validation_split(
            train_examples,
            train_ratio=spec.inputs.data.bootstrap_train_ratio,
            split_seed=validation_split_seed,
        )

    deduped = _deduplicate_examples(train_examples + validation_examples + test_examples)
    deduped.sort(key=lambda record: (SPLIT_ORDER[record.split], record.id))
    return ExamplePool(version=version, examples=deduped)


def merge_generated_examples(
    existing_pool: ExamplePool,
    generated_examples: list[ExampleRecord],
    *,
    version: str,
    train_ratio: float,
    split_seed: str,
) -> tuple[ExamplePool, list[ExampleRecord], list[ExampleRecord]]:
    existing_by_input = {_normalize_input(record.input): record for record in existing_pool.examples}
    deduped_new: dict[str, ExampleRecord] = {}
    rejected: list[ExampleRecord] = []
    for record in generated_examples:
        key = _normalize_input(record.input)
        existing = existing_by_input.get(key)
        if existing is not None:
            rejected.append(record)
            continue

        current = deduped_new.get(key)
        if current is None:
            deduped_new[key] = record
            continue
        if current.label != record.label:
            rejected.append(record)
            continue
        rejected.append(record)

    accepted = list(deduped_new.values())
    accepted = _assign_generated_splits(accepted, train_ratio=train_ratio, split_seed=split_seed)
    merged_examples = [*existing_pool.examples, *accepted]
    merged_examples.sort(key=lambda record: (SPLIT_ORDER[record.split], record.id))
    return ExamplePool(version=version, examples=merged_examples), accepted, rejected


def has_train_or_validation_examples(example_pool: ExamplePool) -> bool:
    return any(record.split in {"train", "validation"} for record in example_pool.examples)


def _load_split_examples(
    path: Path | None,
    *,
    split: str,
    data_format: str,
    label_space: list[str],
) -> list[ExampleRecord]:
    if path is None:
        return []

    raw_items = _read_records(path, data_format)
    records = []
    for index, item in enumerate(raw_items, start=1):
        records.append(
            _normalize_record(
                item,
                split=split,
                label_space=label_space,
                index=index,
                path=path,
            )
        )
    return records


def _read_records(path: Path, data_format: str) -> list[dict]:
    if data_format == "json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        return payload

    if data_format == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        return rows

    raise ValueError(f"unsupported data format: {data_format}")


def _normalize_record(
    item: dict,
    *,
    split: str,
    label_space: list[str],
    index: int,
    path: Path,
) -> ExampleRecord:
    if not isinstance(item, dict):
        raise ValueError(f"expected an example object in {path}, got {type(item).__name__}")
    if "input" not in item or "label" not in item:
        raise ValueError(f"examples in {path} must contain input and label")

    row_split = item.get("split")
    if row_split is not None and row_split != split:
        raise ValueError(f"example split {row_split!r} does not match expected split {split!r} in {path}")

    label = item["label"]
    if not isinstance(label, str):
        raise ValueError(f"example label must be a string in {path}")
    if label not in label_space:
        raise ValueError(f"label {label!r} is not in evaluate.label_space for {path}")

    keyphrases = item.get("keyphrases", [])
    if not isinstance(keyphrases, list) or any(not isinstance(value, str) for value in keyphrases):
        raise ValueError(f"example keyphrases must be a list of strings in {path}")

    provenance = item.get("provenance", [])
    if not isinstance(provenance, list):
        raise ValueError(f"example provenance must be a list in {path}")

    example_id = item.get("id") or _generated_example_id(item["input"], label, index)
    return ExampleRecord(
        id=str(example_id),
        input=str(item["input"]),
        label=label,
        split=split,
        source_type=item.get("source_type", "seed"),
        provenance=[SourcePointer.from_dict(source) for source in provenance],
        keyphrases=copy.deepcopy(keyphrases),
        policy_version=item.get("policy_version"),
    )


def _derive_validation_split(
    train_examples: list[ExampleRecord],
    *,
    train_ratio: float,
    split_seed: str,
) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    n_examples = len(train_examples)
    if n_examples < 2:
        return train_examples, []

    n_validation = round(n_examples * (1 - train_ratio))
    n_validation = max(1, min(n_examples - 1, n_validation))

    ranked = sorted(
        train_examples,
        key=lambda record: _stable_hash(f"{split_seed}:{record.id}:{_normalize_input(record.input)}"),
    )
    validation_ids = {record.id for record in ranked[:n_validation]}

    train_records = []
    validation_records = []
    for record in train_examples:
        if record.id in validation_ids:
            validation_records.append(replace(record, split="validation"))
        else:
            train_records.append(replace(record, split="train"))

    return train_records, validation_records


def _assign_generated_splits(
    records: list[ExampleRecord],
    *,
    train_ratio: float,
    split_seed: str,
) -> list[ExampleRecord]:
    if not records:
        return []
    seed_records = [replace(record, split="train") for record in records]
    train_records, validation_records = _derive_validation_split(
        seed_records,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    combined = [*train_records, *validation_records]
    combined.sort(key=lambda record: (SPLIT_ORDER[record.split], record.id))
    return combined


def _deduplicate_examples(records: list[ExampleRecord]) -> list[ExampleRecord]:
    deduped: dict[str, ExampleRecord] = {}
    for record in records:
        key = _normalize_input(record.input)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = record
            continue

        if existing.label != record.label:
            raise ValueError(
                "conflicting labels for duplicate example text: "
                f"{existing.label!r} vs {record.label!r} for input {record.input[:80]!r}"
            )

        if SPLIT_PRECEDENCE[record.split] > SPLIT_PRECEDENCE[existing.split]:
            deduped[key] = record

    return list(deduped.values())


def _generated_example_id(text: str, label: str, index: int) -> str:
    digest = hashlib.sha256(f"{label}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"ex_{index:04d}_{digest}"


def _normalize_input(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
