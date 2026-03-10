from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .spec import TaskSpec


SUPPORTED_DATA_FORMATS = {"json", "jsonl"}
SUPPORTED_METRICS = {"f1"}


@dataclass(slots=True)
class ResolvedTaskSpec:
    spec: TaskSpec
    source_spec_path: Path
    source_spec_dir: Path
    train_path: Path | None = None
    validation_path: Path | None = None
    test_path: Path | None = None
    initial_policy_doc_path: Path | None = None
    validation_split_seed: str = ""

    def to_run_config_dict(self) -> dict[str, Any]:
        return {
            "version": self.spec.version,
            "task_id": self.spec.task_id,
            "source_spec_path": str(self.source_spec_path),
            "domain": self.spec.domain.to_dict(),
            "initial_policy": {
                "text": self.spec.inputs.initial_policy.text,
                "policy_doc_path": str(self.initial_policy_doc_path) if self.initial_policy_doc_path else None,
            },
            "data": {
                "format": self.spec.inputs.data.format,
                "bootstrap_train_ratio": self.spec.inputs.data.bootstrap_train_ratio,
                "validation_split_seed": self.validation_split_seed,
                "train_path": str(self.train_path) if self.train_path else None,
                "validation_path": str(self.validation_path) if self.validation_path else None,
                "test_path": str(self.test_path) if self.test_path else None,
            },
            "models": {
                "research": self.spec.research.model.to_dict(),
                "redteam": self.spec.redteam.model.to_dict(),
                "reader": self.spec.optimize.reader_model.to_dict(),
            },
            "research": self.spec.research.to_dict(),
            "redteam": self.spec.redteam.to_dict(),
            "optimize": self.spec.optimize.to_dict(),
            "evaluate": self.spec.evaluate.to_dict(),
        }


def load_task_spec(path: str | Path) -> ResolvedTaskSpec:
    source_spec_path = Path(path).expanduser().resolve()
    with source_spec_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    spec = TaskSpec.from_dict(data)
    spec.validate()
    _validate_task_spec(spec)

    source_spec_dir = source_spec_path.parent
    return ResolvedTaskSpec(
        spec=spec,
        source_spec_path=source_spec_path,
        source_spec_dir=source_spec_dir,
        train_path=_resolve_optional_path(spec.inputs.data.train_path, source_spec_dir, "train_path"),
        validation_path=_resolve_optional_path(spec.inputs.data.validation_path, source_spec_dir, "validation_path"),
        test_path=_resolve_optional_path(spec.inputs.data.test_path, source_spec_dir, "test_path"),
        initial_policy_doc_path=_resolve_optional_path(
            spec.inputs.initial_policy.policy_doc_path,
            source_spec_dir,
            "policy_doc_path",
        ),
        validation_split_seed=spec.task_id,
    )


def write_task_spec(path: str | Path, spec: TaskSpec) -> None:
    destination = Path(path)
    destination.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _validate_task_spec(spec: TaskSpec) -> None:
    label_space = spec.evaluate.label_space
    if not label_space:
        raise ValueError("evaluate.label_space must contain at least one label")
    if any(not isinstance(label, str) or not label.strip() for label in label_space):
        raise ValueError("evaluate.label_space must contain non-empty strings")
    if len(label_space) != len(set(label_space)):
        raise ValueError("evaluate.label_space must not contain duplicates")

    metric_names = [metric.name for metric in spec.evaluate.metrics]
    unsupported_metrics = sorted(set(metric_names) - SUPPORTED_METRICS)
    if unsupported_metrics:
        joined = ", ".join(unsupported_metrics)
        raise ValueError(f"unsupported evaluate.metrics values: {joined}")
    if spec.optimize.acceptance_metric not in metric_names:
        raise ValueError(
            "optimize.acceptance_metric must be present in evaluate.metrics; "
            f"got {spec.optimize.acceptance_metric!r}"
        )
    if spec.optimize.acceptance_split not in {"train", "validation", "test"}:
        raise ValueError(
            "optimize.acceptance_split must be one of 'train', 'validation', or 'test'; "
            f"got {spec.optimize.acceptance_split!r}"
        )

    data_format = spec.inputs.data.format
    if data_format not in SUPPORTED_DATA_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_DATA_FORMATS))
        raise ValueError(f"unsupported inputs.data.format: {data_format!r}; expected one of {supported}")

    train_ratio = spec.inputs.data.bootstrap_train_ratio
    if not 0 < train_ratio < 1:
        raise ValueError("inputs.data.bootstrap_train_ratio must be between 0 and 1")


def _resolve_optional_path(value: str | None, base_dir: Path, field_name: str) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{field_name} must point to a file: {path}")
    return path
