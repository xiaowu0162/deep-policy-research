from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import MetricsArtifact, ReaderPrompts, SplitMetrics
from .config import ResolvedTaskSpec, load_task_spec
from .datasets import SPLIT_ORDER, load_example_pool
from .examples import ExamplePool, ExampleRecord
from .openai_client import (
    create_chat_completion,
    create_sync_client,
    extract_text_from_chat_response,
    probe_model,
)
from .policy import PolicyDoc, PolicyRule, PolicySection
from .prompts import default_reader_prompts, render_policy_doc, render_reader_messages
from .run_manager import RunManager
from .spec import ModelConfig, TaskSpec

DEFAULT_READER_MAX_TOKENS = 8192


@dataclass(slots=True)
class EvaluationRecord:
    example_id: str
    split: str
    label: str
    prediction: str | None
    raw_output: str

    @property
    def correct(self) -> bool:
        return self.prediction == self.label


def run_eval_command(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    probe_all_models: bool = False,
) -> tuple[RunManager, MetricsArtifact]:
    resolved = load_task_spec(spec_path)
    run_config = resolved.to_run_config_dict()
    run_config["entrypoint"] = "eval"
    manager = RunManager.create(output_dir=output_dir, spec=resolved.spec, run_config=run_config)
    return _run_or_resume_eval(manager, resolved, probe_all_models=probe_all_models)


def resume_eval_command(run_dir: str | Path) -> tuple[RunManager, MetricsArtifact | None]:
    manager = RunManager.load(run_dir)
    if manager.manifest.status == "completed":
        current_version = manager.manifest.current_metrics_version
        if current_version is None:
            return manager, None
        return manager, manager.load_artifact("metrics", MetricsArtifact, current_version)

    with manager.spec_path.open("r", encoding="utf-8") as handle:
        spec = TaskSpec.from_dict(json.load(handle))

    resolved = ResolvedTaskSpec(
        spec=spec,
        source_spec_path=manager.spec_path,
        source_spec_dir=manager.spec_path.parent,
        train_path=_optional_path_from_run_config(manager.run_config_path, "train_path"),
        validation_path=_optional_path_from_run_config(manager.run_config_path, "validation_path"),
        test_path=_optional_path_from_run_config(manager.run_config_path, "test_path"),
        initial_policy_doc_path=_optional_path_from_run_config(manager.run_config_path, "policy_doc_path"),
        research_search_fixture_path=_optional_path_from_run_config(manager.run_config_path, "research.search.fixture_path"),
        redteam_search_fixture_path=_optional_path_from_run_config(manager.run_config_path, "redteam.search.fixture_path"),
        validation_split_seed=_read_validation_split_seed(manager.run_config_path, default=spec.task_id),
    )
    return _run_or_resume_eval(manager, resolved, probe_all_models=False)


def evaluate_policy(
    *,
    client: Any,
    model_config: ModelConfig,
    policy: PolicyDoc,
    example_pool: ExamplePool,
    prompts: ReaderPrompts,
    label_space: list[str],
    metric_names: list[str],
) -> tuple[dict[str, SplitMetrics], dict[str, list[EvaluationRecord]]]:
    policy_text = render_policy_doc(policy)
    grouped_records: dict[str, list[ExampleRecord]] = defaultdict(list)
    for record in example_pool.examples:
        grouped_records[record.split].append(record)

    split_metrics: dict[str, SplitMetrics] = {}
    predictions: dict[str, list[EvaluationRecord]] = {}

    for split in sorted(grouped_records, key=lambda value: SPLIT_ORDER[value]):
        records = grouped_records[split]
        split_predictions: list[EvaluationRecord] = []
        for record in records:
            messages = render_reader_messages(
                prompts,
                policy_text=policy_text,
                input_text=record.input,
                labels=label_space,
            )
            completion_overrides: dict[str, Any] = {"temperature": 0}
            if (
                "max_tokens" not in model_config.request_defaults
                and "max_completion_tokens" not in model_config.request_defaults
            ):
                completion_overrides["max_tokens"] = DEFAULT_READER_MAX_TOKENS
            response = create_chat_completion(
                client,
                model_config,
                messages,
                **completion_overrides,
            )
            raw_output = extract_text_from_chat_response(response).strip()
            prediction = parse_label(raw_output, label_space)
            split_predictions.append(
                EvaluationRecord(
                    example_id=record.id,
                    split=split,
                    label=record.label,
                    prediction=prediction,
                    raw_output=raw_output,
                )
            )

        split_metrics[split] = SplitMetrics(
            n_examples=len(records),
            metrics=_calculate_metrics(split_predictions, label_space, metric_names),
        )
        predictions[split] = split_predictions

    return split_metrics, predictions


def parse_label(text: str, label_space: list[str]) -> str | None:
    lookup = {_normalize_label(label): label for label in label_space}
    normalized_lines = [_normalize_label(line) for line in text.splitlines() if line.strip()]
    for line in reversed(normalized_lines):
        if line in lookup:
            return lookup[line]

    normalized_text = _normalize_label(text)
    matches = [label for normalized, label in lookup.items() if _contains_label(normalized_text, normalized)]
    if len(matches) == 1:
        return matches[0]
    return None


def coerce_initial_policy(spec: TaskSpec, *, version: str, policy_doc_path: Path | None = None) -> PolicyDoc:
    if policy_doc_path is not None:
        with policy_doc_path.open("r", encoding="utf-8") as handle:
            loaded = PolicyDoc.from_dict(json.load(handle))
        return PolicyDoc(version=version, sections=loaded.sections)

    text = spec.inputs.initial_policy.text
    if not text or not text.strip():
        raise ValueError("evaluation requires inputs.initial_policy.text or inputs.initial_policy.policy_doc_path")

    blocks = [block.strip(" -*\t\r\n") for block in re.split(r"\n\s*\n", text) if block.strip()]
    rules = [PolicyRule(text=block) for block in blocks]
    return PolicyDoc(
        version=version,
        sections=[
            PolicySection(
                title=spec.domain.name,
                content_type="rules",
                rules=rules,
            )
        ],
    )


def summarize_metrics(metrics: MetricsArtifact) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split, split_metrics in metrics.splits.items():
        summary[split] = dict(split_metrics.metrics)
    return summary


def _run_or_resume_eval(
    manager: RunManager,
    resolved: ResolvedTaskSpec,
    *,
    probe_all_models: bool,
) -> tuple[RunManager, MetricsArtifact]:
    try:
        manager.update_status("running", current_step="eval")

        policy = _load_or_write_policy(manager, resolved)
        example_pool = _load_or_write_example_pool(manager, resolved)
        if not example_pool.examples:
            raise ValueError("evaluation requires at least one example across train, validation, or test")
        prompts = _load_or_write_prompts(manager)

        _probe_models(resolved.spec, probe_all_models=probe_all_models)
        client = create_sync_client(resolved.spec.optimize.reader_model)
        split_metrics, _ = evaluate_policy(
            client=client,
            model_config=resolved.spec.optimize.reader_model,
            policy=policy,
            example_pool=example_pool,
            prompts=prompts,
            label_space=resolved.spec.evaluate.label_space,
            metric_names=[metric.name for metric in resolved.spec.evaluate.metrics],
        )

        metrics = MetricsArtifact(
            version=manager.new_version("metrics"),
            policy_version=policy.version,
            example_pool_version=example_pool.version,
            prompt_version=prompts.version,
            splits=split_metrics,
        )
        manager.write_artifact("metrics", metrics)
        manager.update_status("completed", current_step="eval")
        return manager, metrics
    except Exception:
        manager.update_status("failed", current_step="eval")
        raise


def _load_or_write_policy(manager: RunManager, resolved: ResolvedTaskSpec) -> PolicyDoc:
    if manager.manifest.current_policy_version is not None:
        return manager.load_artifact("policy", PolicyDoc)

    policy = coerce_initial_policy(
        resolved.spec,
        version=manager.new_version("policy"),
        policy_doc_path=resolved.initial_policy_doc_path,
    )
    manager.write_artifact("policy", policy)
    return policy


def _load_or_write_example_pool(manager: RunManager, resolved: ResolvedTaskSpec) -> ExamplePool:
    if manager.manifest.current_example_pool_version is not None:
        return manager.load_artifact("example_pool", ExamplePool)

    example_pool = load_example_pool(
        resolved.spec,
        version=manager.new_version("example_pool"),
        data_format=resolved.spec.inputs.data.format,
        train_path=resolved.train_path,
        validation_path=resolved.validation_path,
        test_path=resolved.test_path,
        validation_split_seed=resolved.validation_split_seed,
    )
    manager.write_artifact("example_pool", example_pool)
    return example_pool


def _load_or_write_prompts(manager: RunManager) -> ReaderPrompts:
    if manager.manifest.current_prompt_version is not None:
        return manager.load_artifact("prompt", ReaderPrompts)

    prompts = default_reader_prompts(version=manager.new_version("prompt"))
    manager.write_artifact("prompt", prompts)
    return prompts


def _probe_models(spec: TaskSpec, *, probe_all_models: bool) -> None:
    if probe_all_models:
        configs = [spec.research.model, spec.redteam.model, spec.optimize.reader_model]
    else:
        configs = [spec.optimize.reader_model]

    for config in configs:
        client = create_sync_client(config)
        probe_model(client, config)


def _calculate_metrics(
    records: list[EvaluationRecord],
    label_space: list[str],
    metric_names: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in metric_names:
        if name == "f1":
            metrics[name] = _macro_f1(records, label_space)
        else:  # pragma: no cover - guarded at config-load time
            raise ValueError(f"unsupported metric: {name}")
    return metrics


def _macro_f1(records: list[EvaluationRecord], label_space: list[str]) -> float:
    if not records:
        return 0.0

    f1_values = []
    for label in label_space:
        tp = sum(1 for record in records if record.label == label and record.prediction == label)
        fp = sum(1 for record in records if record.label != label and record.prediction == label)
        fn = sum(1 for record in records if record.label == label and record.prediction != label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append((2 * precision * recall) / (precision + recall))

    return sum(f1_values) / len(label_space)


def _normalize_label(text: str) -> str:
    stripped = text.strip().upper()
    stripped = re.sub(r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", stripped)
    return " ".join(stripped.split())


def _contains_label(text: str, label: str) -> bool:
    pattern = r"(?<![A-Z0-9])" + re.escape(label) + r"(?![A-Z0-9])"
    return re.search(pattern, text) is not None


def _optional_path_from_run_config(run_config_path: Path, key: str) -> Path | None:
    with run_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if key == "policy_doc_path":
        value = config["initial_policy"].get("policy_doc_path")
    elif key.startswith("research."):
        value = _get_nested_value(config.get("research", {}), key.split(".")[1:])
    elif key.startswith("redteam."):
        value = _get_nested_value(config.get("redteam", {}), key.split(".")[1:])
    else:
        value = config["data"].get(key)
    return Path(value) if value else None


def _read_validation_split_seed(run_config_path: Path, *, default: str) -> str:
    with run_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("data", {}).get("validation_split_seed", default)


def _get_nested_value(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
