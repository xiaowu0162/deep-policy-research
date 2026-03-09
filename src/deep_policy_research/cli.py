from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from .artifacts import (
    CandidateRuleRecord,
    ChunkRecord,
    FilteredRuleRecord,
    MetricsArtifact,
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "schema":
        sample = SCHEMA_SAMPLES[args.name]()
        print(json.dumps(sample.to_dict(), indent=2, sort_keys=True))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
