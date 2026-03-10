from __future__ import annotations

import json
from pathlib import Path

from .artifacts import MetricsArtifact, PromptOptimizationArtifact, ReaderPrompts
from .config import ResolvedTaskSpec, load_task_spec
from .eval import (
    _load_or_write_example_pool,
    _load_or_write_policy,
    _load_or_write_prompts,
    _probe_models,
    _read_validation_split_seed,
    _optional_path_from_run_config,
    evaluate_policy,
)
from .openai_client import create_sync_client
from .run_manager import RunManager
from .spec import TaskSpec
from .prompts import generate_prompt_variants


PromptOptimizationResult = PromptOptimizationArtifact


def optimize_reader_command(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    probe_all_models: bool = False,
) -> tuple[RunManager, PromptOptimizationResult, MetricsArtifact]:
    resolved = load_task_spec(spec_path)
    run_config = resolved.to_run_config_dict()
    run_config["entrypoint"] = "optimize-reader"
    manager = RunManager.create(
        output_dir=output_dir,
        spec=resolved.spec,
        run_config=run_config,
        current_step="optimize",
    )
    return _run_or_resume_optimize(manager, resolved, probe_all_models=probe_all_models)


def resume_optimize_reader_command(
    run_dir: str | Path,
) -> tuple[RunManager, PromptOptimizationResult, MetricsArtifact]:
    manager = RunManager.load(run_dir)
    with manager.spec_path.open("r", encoding="utf-8") as handle:
        spec = TaskSpec.from_dict(json.load(handle))

    if manager.manifest.status == "completed":
        if manager.manifest.current_prompt_optimization_version is not None:
            result = manager.load_artifact("prompt_optimization", PromptOptimizationArtifact)
            metrics = manager.load_artifact("metrics", MetricsArtifact, version=result.best_metrics_version)
            return manager, result, metrics
        result, metrics = _reconstruct_completed_result_from_current_aliases(manager, spec)
        return manager, result, metrics

    resolved = ResolvedTaskSpec(
        spec=spec,
        source_spec_path=manager.spec_path,
        source_spec_dir=manager.spec_path.parent,
        train_path=_optional_path_from_run_config(manager.run_config_path, "train_path"),
        validation_path=_optional_path_from_run_config(manager.run_config_path, "validation_path"),
        test_path=_optional_path_from_run_config(manager.run_config_path, "test_path"),
        initial_policy_doc_path=_optional_path_from_run_config(manager.run_config_path, "policy_doc_path"),
        validation_split_seed=_read_validation_split_seed(manager.run_config_path, default=spec.task_id),
    )
    return _run_or_resume_optimize(manager, resolved, probe_all_models=False)


def _run_or_resume_optimize(
    manager: RunManager,
    resolved: ResolvedTaskSpec,
    *,
    probe_all_models: bool,
) -> tuple[RunManager, PromptOptimizationResult, MetricsArtifact]:
    try:
        manager.update_status("running", current_step="optimize")

        policy = _load_or_write_policy(manager, resolved)
        example_pool = _load_or_write_example_pool(manager, resolved)
        if not example_pool.examples:
            raise ValueError("prompt optimization requires at least one example across train, validation, or test")
        prompts = _load_or_write_prompts(manager)

        _probe_models(resolved.spec, probe_all_models=probe_all_models)
        client = create_sync_client(resolved.spec.optimize.reader_model)

        baseline_metrics = _ensure_current_metrics(
            manager=manager,
            client=client,
            resolved=resolved,
            prompts=prompts,
            accept=True,
        )
        acceptance_split = resolved.spec.optimize.acceptance_split
        acceptance_metric = resolved.spec.optimize.acceptance_metric
        baseline_score = _get_metric_score(baseline_metrics, acceptance_split, acceptance_metric)

        best_prompt_version = prompts.version
        best_metrics = baseline_metrics
        best_metrics_version = baseline_metrics.version
        best_score = baseline_score
        candidate_prompt_versions: list[str] = []
        next_prompt_seq = len(manager.manifest.prompt_versions) + 1

        def next_prompt_version() -> str:
            nonlocal next_prompt_seq
            version = f"{manager.manifest.run_id}__reader_prompts__{next_prompt_seq:03d}"
            next_prompt_seq += 1
            return version

        for candidate in generate_prompt_variants(
            prompts,
            version_factory=next_prompt_version,
        ):
            manager.record_artifact("prompt", candidate, accept=False)
            candidate_prompt_versions.append(candidate.version)
            metrics = _evaluate_metrics_artifact(
                manager=manager,
                client=client,
                resolved=resolved,
                prompts=candidate,
                accept=False,
            )
            score = _get_metric_score(metrics, acceptance_split, acceptance_metric)
            if score > best_score:
                best_score = score
                best_prompt_version = candidate.version
                best_metrics = metrics
                best_metrics_version = metrics.version

        improved = best_prompt_version != prompts.version
        if improved:
            manager.accept_artifact_version("prompt", best_prompt_version)
            manager.accept_artifact_version("metrics", best_metrics_version)

        manager.update_status("completed", current_step="optimize")
        result = PromptOptimizationResult(
            version=manager.new_version("prompt_optimization"),
            acceptance_split=acceptance_split,
            acceptance_metric=acceptance_metric,
            baseline_score=baseline_score,
            best_score=best_score,
            improved=improved,
            baseline_prompt_version=prompts.version,
            best_prompt_version=best_prompt_version,
            best_metrics_version=best_metrics_version,
            candidate_prompt_versions=candidate_prompt_versions,
        )
        manager.record_artifact("prompt_optimization", result, accept=True)
        return manager, result, best_metrics
    except Exception:
        manager.update_status("failed", current_step="optimize")
        raise


def _ensure_current_metrics(
    *,
    manager: RunManager,
    client,
    resolved: ResolvedTaskSpec,
    prompts: ReaderPrompts,
    accept: bool,
) -> MetricsArtifact:
    if manager.manifest.current_metrics_version is not None:
        metrics = manager.load_artifact("metrics", MetricsArtifact)
        if _metrics_match_current_state(
            metrics,
            policy_version=manager.manifest.current_policy_version,
            example_pool_version=manager.manifest.current_example_pool_version,
            prompt_version=prompts.version,
        ):
            return metrics

    return _evaluate_metrics_artifact(
        manager=manager,
        client=client,
        resolved=resolved,
        prompts=prompts,
        accept=accept,
    )


def _evaluate_metrics_artifact(
    *,
    manager: RunManager,
    client,
    resolved: ResolvedTaskSpec,
    prompts: ReaderPrompts,
    accept: bool,
) -> MetricsArtifact:
    policy = _load_or_write_policy(manager, resolved)
    example_pool = _load_or_write_example_pool(manager, resolved)
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
    manager.record_artifact("metrics", metrics, accept=accept)
    return metrics


def _metrics_match_current_state(
    metrics: MetricsArtifact,
    *,
    policy_version: str | None,
    example_pool_version: str | None,
    prompt_version: str | None,
) -> bool:
    return (
        metrics.policy_version == policy_version
        and metrics.example_pool_version == example_pool_version
        and metrics.prompt_version == prompt_version
    )


def _get_metric_score(metrics: MetricsArtifact, split: str, metric_name: str) -> float:
    if split not in metrics.splits:
        raise ValueError(f"acceptance split {split!r} is not available in metrics artifact {metrics.version}")
    metric_values = metrics.splits[split].metrics
    if metric_name not in metric_values:
        raise ValueError(f"acceptance metric {metric_name!r} is not available in split {split!r}")
    return metric_values[metric_name]


def _reconstruct_completed_result_from_current_aliases(
    manager: RunManager,
    spec: TaskSpec,
) -> tuple[PromptOptimizationResult, MetricsArtifact]:
    prompts = manager.load_artifact("prompt", ReaderPrompts)
    metrics = manager.load_artifact("metrics", MetricsArtifact)
    score = _get_metric_score(
        metrics,
        split=spec.optimize.acceptance_split,
        metric_name=spec.optimize.acceptance_metric,
    )
    return (
        PromptOptimizationResult(
            version="legacy_completed_run",
            acceptance_split=spec.optimize.acceptance_split,
            acceptance_metric=spec.optimize.acceptance_metric,
            baseline_score=score,
            best_score=score,
            improved=False,
            baseline_prompt_version=prompts.version,
            best_prompt_version=prompts.version,
            best_metrics_version=metrics.version,
            candidate_prompt_versions=[],
        ),
        metrics,
    )
