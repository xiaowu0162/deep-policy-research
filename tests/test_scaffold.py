from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_policy_research.artifacts import (  # noqa: E402
    CandidateRuleRecord,
    ChunkRecord,
    FilteredRuleRecord,
    MetricsArtifact,
    ReaderPrompts,
    RunManifest,
    SourceRecord,
)
from deep_policy_research.examples import ExamplePool  # noqa: E402
from deep_policy_research.policy import PolicyDoc  # noqa: E402
from deep_policy_research.spec import TaskSpec  # noqa: E402


class SchemaRoundTripTests(unittest.TestCase):
    def test_top_level_round_trips(self) -> None:
        for cls in [TaskSpec, PolicyDoc, ExamplePool, RunManifest, ReaderPrompts, MetricsArtifact]:
            sample = cls.sample()
            rebuilt = cls.from_dict(sample.to_dict())
            self.assertEqual(rebuilt.to_dict(), sample.to_dict())

    def test_row_round_trips(self) -> None:
        for cls in [SourceRecord, ChunkRecord, CandidateRuleRecord, FilteredRuleRecord]:
            sample = cls.sample()
            rebuilt = cls.from_dict(sample.to_dict())
            self.assertEqual(rebuilt.to_dict(), sample.to_dict())


class CliTests(unittest.TestCase):
    def test_schema_command_outputs_json(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "deep_policy_research", "schema", "policy-doc"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        self.assertIn("sections", data)


if __name__ == "__main__":
    unittest.main()
