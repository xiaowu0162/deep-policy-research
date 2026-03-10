from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from .agent import resume_task_command, run_task_command
from .benchmarks.openai_moderation import get_task as get_openai_moderation_task
from .benchmarks.openai_moderation import list_tasks as list_openai_moderation_tasks
from .benchmarks.openai_moderation import DEFAULT_ROOT as OPENAI_MODERATION_DEFAULT_ROOT
from .eval import resume_eval_command, run_eval_command, summarize_metrics
from .optimize import optimize_reader_command, resume_optimize_reader_command
from .redteam import run_redteam_command, resume_redteam_command
from .research import resume_research_command, run_research_command
from .artifacts import (
    CandidateRuleRecord,
    ChunkRecord,
    FilteredRuleRecord,
    MetricsArtifact,
    PromptOptimizationArtifact,
    ResearchArtifact,
    RedteamArtifact,
    ReaderPrompts,
    RunManifest,
    SourceRecord,
)
from .examples import ExamplePool
from .policy import PolicyDoc
from .spec import TaskSpec


SampleFactory = Callable[[], Any]


SCHEMA_SAMPLES: dict[str, SampleFactory] = {
    "task-spec": TaskSpec.sample,
    "policy-doc": PolicyDoc.sample,
    "example-pool": ExamplePool.sample,
    "run-manifest": RunManifest.sample,
    "reader-prompts": ReaderPrompts.sample,
    "metrics": MetricsArtifact.sample,
    "research": ResearchArtifact.sample,
    "prompt-optimization": PromptOptimizationArtifact.sample,
    "redteam": RedteamArtifact.sample,
    "source-row": SourceRecord.sample,
    "chunk-row": ChunkRecord.sample,
    "candidate-rule-row": CandidateRuleRecord.sample,
    "filtered-rule-row": FilteredRuleRecord.sample,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dpr", description="Deep policy research scaffold CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Print a sample schema artifact.")
    schema_parser.add_argument("name", choices=sorted(SCHEMA_SAMPLES))

    run_parser = subparsers.add_parser(
        "run",
        help="Run the current research -> redteam -> optimize -> eval slice in one run directory.",
    )
    run_parser.add_argument("--spec", required=True, help="Path to a TaskSpec JSON file.")
    run_parser.add_argument("--output-dir", default="runs", help="Directory that will contain the run folder.")

    research_parser = subparsers.add_parser(
        "research",
        help="Run the research stage and write sources, chunks, candidate rules, filtered rules, and a policy revision.",
    )
    research_parser.add_argument("--spec", required=True, help="Path to a TaskSpec JSON file.")
    research_parser.add_argument("--output-dir", default="runs", help="Directory that will contain the run folder.")

    redteam_parser = subparsers.add_parser(
        "redteam",
        help="Run the red-team stage and update the example pool with accepted examples.",
    )
    redteam_parser.add_argument("--spec", required=True, help="Path to a TaskSpec JSON file.")
    redteam_parser.add_argument("--output-dir", default="runs", help="Directory that will contain the run folder.")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a fixed policy against the configured dataset.")
    eval_parser.add_argument("--spec", required=True, help="Path to a TaskSpec JSON file.")
    eval_parser.add_argument("--output-dir", default="runs", help="Directory that will contain the run folder.")

    optimize_parser = subparsers.add_parser(
        "optimize-reader",
        help="Search over prompt variants and accept a new reader prompt only when validation improves.",
    )
    optimize_parser.add_argument("--spec", required=True, help="Path to a TaskSpec JSON file.")
    optimize_parser.add_argument("--output-dir", default="runs", help="Directory that will contain the run folder.")

    resume_parser = subparsers.add_parser("resume", help="Resume a run from the last accepted artifacts.")
    resume_parser.add_argument("--run-dir", required=True, help="Existing run directory.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a run or the OpenAI moderation adapter.")
    inspect_subparsers = inspect_parser.add_subparsers(dest="inspect_target", required=True)

    inspect_run_parser = inspect_subparsers.add_parser("run", help="Print the run manifest and directory.")
    inspect_run_parser.add_argument("run_dir", help="Run directory to inspect.")

    inspect_moderation_parser = inspect_subparsers.add_parser(
        "openai-moderation",
        help="List or inspect OpenAI moderation tasks from the resplit dataset.",
    )
    inspect_moderation_parser.add_argument(
        "--root",
        default=str(OPENAI_MODERATION_DEFAULT_ROOT),
        help="Root directory containing manifest.json and domain folders.",
    )
    inspect_moderation_parser.add_argument("--task-id", help="Optional task identifier to inspect.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "schema":
        sample = SCHEMA_SAMPLES[args.name]()
        print(json.dumps(sample.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        manager, result = run_task_command(
            args.spec,
            output_dir=args.output_dir,
            probe_all_models=True,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(manager.root_dir),
                    **result.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "research":
        manager, research, policy = run_research_command(
            args.spec,
            output_dir=args.output_dir,
            probe_all_models=True,
        )
        print(
            json.dumps(
                {
                    "research": research.to_dict(),
                    "run_dir": str(manager.root_dir),
                    "policy_version": policy.version,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "redteam":
        manager, redteam, example_pool = run_redteam_command(
            args.spec,
            output_dir=args.output_dir,
            probe_all_models=True,
        )
        print(
            json.dumps(
                {
                    "redteam": redteam.to_dict(),
                    "run_dir": str(manager.root_dir),
                    "example_pool_version": example_pool.version,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "optimize-reader":
        manager, optimization, metrics = optimize_reader_command(
            args.spec,
            output_dir=args.output_dir,
            probe_all_models=False,
        )
        print(
            json.dumps(
                {
                    "optimization": optimization.to_dict(),
                    "run_dir": str(manager.root_dir),
                    "metrics": summarize_metrics(metrics),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "eval":
        manager, metrics = run_eval_command(args.spec, output_dir=args.output_dir, probe_all_models=False)
        print(
            json.dumps(
                {
                    "run_dir": str(manager.root_dir),
                    "metrics": summarize_metrics(metrics),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "resume":
        from .run_manager import RunManager

        current_manager = RunManager.load(args.run_dir)
        entrypoint = _read_entrypoint(current_manager.run_config_path)
        if entrypoint == "run":
            manager, result = resume_task_command(args.run_dir)
            payload = {"run_dir": str(manager.root_dir), **result.to_dict()}
        elif entrypoint == "research" or current_manager.manifest.current_step == "research":
            manager, research, policy = resume_research_command(args.run_dir)
            payload = {
                "research": research.to_dict(),
                "run_dir": str(manager.root_dir),
                "policy_version": policy.version,
            }
        elif entrypoint == "redteam" or current_manager.manifest.current_step == "redteam":
            manager, redteam, example_pool = resume_redteam_command(args.run_dir)
            payload = {
                "redteam": redteam.to_dict(),
                "run_dir": str(manager.root_dir),
                "example_pool_version": example_pool.version,
            }
        elif entrypoint == "optimize-reader" or current_manager.manifest.current_step == "optimize":
            manager, optimization, metrics = resume_optimize_reader_command(args.run_dir)
            payload = {
                "optimization": optimization.to_dict(),
                "run_dir": str(manager.root_dir),
                "metrics": summarize_metrics(metrics),
            }
        else:
            manager, metrics = resume_eval_command(args.run_dir)
            payload = {"run_dir": str(manager.root_dir)}
            if metrics is not None:
                payload["metrics"] = summarize_metrics(metrics)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "inspect":
        if args.inspect_target == "run":
            from .run_manager import RunManager

            manager = RunManager.load(args.run_dir)
            print(json.dumps(manager.describe(), indent=2, sort_keys=True))
            return 0

        if args.inspect_target == "openai-moderation":
            if args.task_id:
                task = get_openai_moderation_task(args.task_id, root_dir=args.root)
                print(json.dumps(task.to_dict(), indent=2, sort_keys=True))
            else:
                tasks = [task.to_dict() for task in list_openai_moderation_tasks(args.root)]
                print(json.dumps({"tasks": tasks}, indent=2, sort_keys=True))
            return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


def _read_entrypoint(run_config_path) -> str | None:
    with run_config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("entrypoint")
