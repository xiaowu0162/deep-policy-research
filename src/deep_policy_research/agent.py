from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import MetricsArtifact, PromptOptimizationArtifact, RedteamArtifact
from .config import ResolvedTaskSpec, load_task_spec
from .eval import _optional_path_from_run_config, _read_validation_split_seed, _run_or_resume_eval
from .optimize import _run_or_resume_optimize
from .research import _load_current_research_result, _run_or_resume_research
from .redteam import _run_or_resume_redteam
from .policy import PolicyDoc
from .run_manager import RunManager
from .spec import TaskSpec


@dataclass(slots=True)
class RunResult:
    research: dict[str, object]
    redteam: dict[str, object] | None
    optimization: dict[str, object] | None
    metrics: dict[str, dict[str, float]] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "research": self.research,
            "redteam": self.redteam,
            "optimization": self.optimization,
            "metrics": self.metrics,
        }


def run_task_command(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    probe_all_models: bool = False,
) -> tuple[RunManager, RunResult]:
    resolved = load_task_spec(spec_path)
    run_config = resolved.to_run_config_dict()
    run_config["entrypoint"] = "run"
    manager = RunManager.create(
        output_dir=output_dir,
        spec=resolved.spec,
        run_config=run_config,
        current_step="research",
    )
    return _resume_full_run(manager, resolved, probe_all_models=probe_all_models)


def resume_task_command(run_dir: str | Path) -> tuple[RunManager, RunResult]:
    manager = RunManager.load(run_dir)
    resolved = _load_resolved_spec_from_run(manager)
    return _resume_full_run(manager, resolved, probe_all_models=False)


def _resume_full_run(
    manager: RunManager,
    resolved: ResolvedTaskSpec,
    *,
    probe_all_models: bool,
) -> tuple[RunManager, RunResult]:
    current_policy = None
    if manager.manifest.current_policy_version is not None:
        current_policy = manager.load_artifact("policy", PolicyDoc)

    if manager.manifest.current_step == "research":
        if manager.manifest.status == "completed":
            if current_policy is None:
                raise ValueError("completed research step is missing its accepted policy artifact")
            research_result = _load_current_research_result(manager, current_policy, resolved.spec)
        else:
            manager, research_result, current_policy = _run_or_resume_research(
                manager,
                resolved,
                probe_all_models=probe_all_models,
            )
    else:
        if current_policy is None:
            raise ValueError("full run resume requires an accepted policy artifact")
        research_result = _load_current_research_result(manager, current_policy, resolved.spec)

    if manager.manifest.current_step in {"research", "redteam"}:
        if manager.manifest.current_step == "redteam" and manager.manifest.status == "completed":
            redteam_result = manager.load_artifact("redteam", RedteamArtifact)
        else:
            manager, redteam_result, _ = _run_or_resume_redteam(
                manager,
                resolved,
                probe_all_models=False,
            )
    else:
        if manager.manifest.current_redteam_version is not None:
            redteam_result = manager.load_artifact("redteam", RedteamArtifact)
        else:
            redteam_result = None

    if manager.manifest.current_step in {"research", "redteam", "optimize"}:
        if manager.manifest.current_step == "optimize" and manager.manifest.status == "completed":
            optimization = manager.load_artifact("prompt_optimization", PromptOptimizationArtifact)
        else:
            manager, optimization, _ = _run_or_resume_optimize(
                manager,
                resolved,
                probe_all_models=False,
            )
    else:
        if manager.manifest.current_prompt_optimization_version is not None:
            optimization = manager.load_artifact("prompt_optimization", PromptOptimizationArtifact)
        else:
            optimization = None

    if manager.manifest.current_step == "eval" and manager.manifest.status == "completed":
        metrics = manager.load_artifact("metrics", MetricsArtifact)
    else:
        manager, metrics = _run_or_resume_eval(manager, resolved, probe_all_models=False)
    return manager, RunResult(
        research=research_result.to_dict(),
        redteam=redteam_result.to_dict() if redteam_result is not None else None,
        optimization=optimization.to_dict() if optimization is not None else None,
        metrics=_summarize_metrics(metrics),
    )


def _load_resolved_spec_from_run(manager: RunManager) -> ResolvedTaskSpec:
    with manager.spec_path.open("r", encoding="utf-8") as handle:
        spec = TaskSpec.from_dict(json.load(handle))
    return ResolvedTaskSpec(
        spec=spec,
        source_spec_path=manager.spec_path,
        source_spec_dir=manager.spec_path.parent,
        train_path=_optional_path_from_run_config(manager.run_config_path, "train_path"),
        validation_path=_optional_path_from_run_config(manager.run_config_path, "validation_path"),
        test_path=_optional_path_from_run_config(manager.run_config_path, "test_path"),
        initial_policy_doc_path=_optional_path_from_run_config(manager.run_config_path, "policy_doc_path"),
        validation_split_seed=_read_validation_split_seed(manager.run_config_path, default=spec.task_id),
    )


def _summarize_metrics(metrics: MetricsArtifact) -> dict[str, dict[str, float]]:
    return {split: dict(split_metrics.metrics) for split, split_metrics in metrics.splits.items()}
