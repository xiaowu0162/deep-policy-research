from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec import (
    DataConfig,
    DomainSpec,
    EvaluateStageConfig,
    InitialPolicyInput,
    ModelConfig,
    OptimizeStageConfig,
    RedteamStageConfig,
    ResearchStageConfig,
    TaskInputs,
    TaskSpec,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "openai_moderation_resplit"


@dataclass(slots=True)
class OpenAIModerationTask:
    task_id: str
    domain_name: str
    description: str
    train_path: Path
    validation_path: Path
    test_path: Path
    label_space: list[str]
    stats: dict[str, dict[str, int]]
    positive_target_per_train_validation_split: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain_name": self.domain_name,
            "description": self.description,
            "train_path": str(self.train_path),
            "validation_path": str(self.validation_path),
            "test_path": str(self.test_path),
            "label_space": list(self.label_space),
            "stats": self.stats,
            "positive_target_per_train_validation_split": self.positive_target_per_train_validation_split,
        }


def load_manifest(root_dir: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = _resolve_root_dir(root_dir)
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_tasks(root_dir: str | Path = DEFAULT_ROOT) -> list[OpenAIModerationTask]:
    manifest = load_manifest(root_dir)
    root = _resolve_root_dir(root_dir)
    label_space = list(manifest["label_space"])
    tasks = []
    for item in manifest["tasks"]:
        tasks.append(
            OpenAIModerationTask(
                task_id=item["task_id"],
                domain_name=item["domain_name"],
                description=item["description"],
                train_path=root / item["train_path"],
                validation_path=root / item["validation_path"],
                test_path=root / item["test_path"],
                label_space=label_space,
                stats=item["stats"],
                positive_target_per_train_validation_split=item.get(
                    "positive_target_per_train_validation_split"
                ),
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    return tasks


def get_task(task_id: str, root_dir: str | Path = DEFAULT_ROOT) -> OpenAIModerationTask:
    for task in list_tasks(root_dir):
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown OpenAI moderation task_id: {task_id!r}")


def build_task_spec(
    *,
    task_id: str,
    model: ModelConfig,
    initial_policy_text: str | None = None,
    initial_policy_doc_path: str | None = None,
    root_dir: str | Path = DEFAULT_ROOT,
) -> TaskSpec:
    task = get_task(task_id, root_dir=root_dir)
    return TaskSpec(
        task_id=task.task_id,
        domain=DomainSpec(name=task.domain_name, description=task.description),
        inputs=TaskInputs(
            initial_policy=InitialPolicyInput(
                text=initial_policy_text,
                policy_doc_path=initial_policy_doc_path,
            ),
            data=DataConfig(
                train_path=str(task.train_path),
                validation_path=str(task.validation_path),
                test_path=str(task.test_path),
                format="jsonl",
            ),
        ),
        research=ResearchStageConfig(model=model),
        redteam=RedteamStageConfig(model=model),
        optimize=OptimizeStageConfig(reader_model=model),
        evaluate=EvaluateStageConfig(label_space=task.label_space),
    )


def _resolve_root_dir(root_dir: str | Path) -> Path:
    root = Path(root_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "OpenAI moderation resplit data was not found at "
            f"{root}. Pass an explicit root_dir or use --root."
        )
    return root
