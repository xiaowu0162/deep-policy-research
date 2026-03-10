from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_policy_research.artifacts import (  # noqa: E402
    CandidateRuleRecord,
    ChunkRecord,
    FilteredRuleRecord,
    MetricsArtifact,
    PromptOptimizationArtifact,
    ResearchArtifact,
    ReaderPromptMessage,
    ReaderPrompts,
    RedteamArtifact,
    RunManifest,
    SourceRecord,
)
from deep_policy_research.agent import resume_task_command, run_task_command  # noqa: E402
from deep_policy_research.benchmarks.openai_moderation import (  # noqa: E402
    build_task_spec,
    get_task as get_openai_moderation_task,
    list_tasks as list_openai_moderation_tasks,
)
from deep_policy_research.cli import main as cli_main  # noqa: E402
from deep_policy_research.config import load_task_spec  # noqa: E402
from deep_policy_research.datasets import load_example_pool, merge_generated_examples  # noqa: E402
from deep_policy_research.eval import evaluate_policy, parse_label  # noqa: E402
from deep_policy_research.examples import ExamplePool  # noqa: E402
from deep_policy_research.fetch import (  # noqa: E402
    FetchedDocument,
    _extract_text_from_html,
    merge_sources,
    split_text_into_chunks,
)
from deep_policy_research.openai_client import chat_completion_request, probe_model  # noqa: E402
from deep_policy_research.optimize import optimize_reader_command, resume_optimize_reader_command  # noqa: E402
from deep_policy_research.policy import PolicyDoc, SourcePointer  # noqa: E402
from deep_policy_research.prompts import default_reader_prompts, render_reader_messages  # noqa: E402
from deep_policy_research.rank import filter_candidate_rules  # noqa: E402
from deep_policy_research.redteam import resume_redteam_command, run_redteam_command  # noqa: E402
from deep_policy_research.research import resume_research_command, run_research_command  # noqa: E402
from deep_policy_research.run_manager import RunManager  # noqa: E402
from deep_policy_research.search import SearchResult  # noqa: E402
from deep_policy_research.spec import ModelConfig  # noqa: E402
from deep_policy_research.spec import TaskSpec  # noqa: E402


class SchemaRoundTripTests(unittest.TestCase):
    def test_top_level_round_trips(self) -> None:
        for cls in [
            TaskSpec,
            PolicyDoc,
            ExamplePool,
            RunManifest,
            ReaderPrompts,
            MetricsArtifact,
            PromptOptimizationArtifact,
            ResearchArtifact,
            RedteamArtifact,
        ]:
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


class ConfigAndDatasetTests(unittest.TestCase):
    def test_load_task_spec_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"input": "hello", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )
            spec_path = root / "spec.json"
            spec = TaskSpec.sample().to_dict()
            spec["inputs"]["data"] = {
                "train_path": "train.jsonl",
                "format": "jsonl",
                "bootstrap_train_ratio": 0.8,
            }
            _write_json(spec_path, spec)

            resolved = load_task_spec(spec_path)

            self.assertEqual(resolved.train_path, train_path.resolve())
            self.assertIsNone(resolved.validation_path)
            self.assertEqual(resolved.validation_split_seed, spec["task_id"])

    def test_load_task_spec_resolves_fixture_search_paths(self) -> None:
        resolved = load_task_spec(ROOT / "examples" / "cached_demo" / "task_spec.json")
        self.assertEqual(
            resolved.research_search_fixture_path,
            (ROOT / "examples" / "cached_demo" / "search_fixture.json").resolve(),
        )
        self.assertEqual(
            resolved.redteam_search_fixture_path,
            (ROOT / "examples" / "cached_demo" / "search_fixture.json").resolve(),
        )

    def test_dataset_loader_derives_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"input": "hello", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                    {"input": "nice to meet you", "label": "APPROVED"},
                    {"input": "I will kill you", "label": "REJECTED"},
                ],
            )

            spec = TaskSpec.sample()
            spec.inputs.data.train_path = str(train_path)
            spec.inputs.data.validation_path = None
            spec.inputs.data.test_path = None
            spec.inputs.data.bootstrap_train_ratio = 0.75

            pool = load_example_pool(
                spec,
                version="run_x__example_pool__001",
                data_format="jsonl",
                train_path=train_path,
                validation_path=None,
                test_path=None,
                validation_split_seed=spec.task_id,
            )

            split_counts = {split: 0 for split in ["train", "validation", "test"]}
            for example in pool.examples:
                split_counts[example.split] += 1
            self.assertEqual(split_counts["train"], 3)
            self.assertEqual(split_counts["validation"], 1)
            self.assertEqual(split_counts["test"], 0)

    def test_dataset_loader_uses_split_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            _write_jsonl(train_path, [{"input": "duplicate", "label": "APPROVED"}])
            _write_jsonl(
                test_path,
                [
                    {"input": "duplicate", "label": "APPROVED"},
                    {"input": "unique test", "label": "REJECTED"},
                ],
            )

            spec = TaskSpec.sample()
            spec.inputs.data.train_path = str(train_path)
            spec.inputs.data.validation_path = None
            spec.inputs.data.test_path = str(test_path)

            pool = load_example_pool(
                spec,
                version="run_x__example_pool__001",
                data_format="jsonl",
                train_path=train_path,
                validation_path=None,
                test_path=test_path,
                validation_split_seed=spec.task_id,
            )

            duplicates = [example for example in pool.examples if example.input == "duplicate"]
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(duplicates[0].split, "test")

    def test_merge_generated_examples_bootstraps_train_and_validation(self) -> None:
        pool = ExamplePool(version="run_x__example_pool__001", examples=[])
        merged_pool, accepted, rejected = merge_generated_examples(
            pool,
            [
                _make_generated_example("rt1", "I will hurt you", "REJECTED"),
                _make_generated_example("rt2", "I will kill you", "REJECTED"),
            ],
            version="run_x__example_pool__002",
            train_ratio=0.5,
            split_seed="toy-task",
        )

        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(accepted), 2)
        self.assertEqual({example.split for example in accepted}, {"train", "validation"})
        self.assertEqual(len(merged_pool.examples), 2)

    def test_dataset_loader_supports_json_with_metadata_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.json"
            _write_json(
                train_path,
                [
                    {
                        "id": "seed_001",
                        "input": "I will hurt you.",
                        "label": "REJECTED",
                        "source_type": "research",
                        "policy_version": "run_x__policy__003",
                        "keyphrases": ["threat"],
                        "provenance": [
                            {
                                "url": "https://example.com/threats",
                                "supporting_excerpt": "Threatening violence toward a person is abusive conduct.",
                            }
                        ],
                        "ignored_extra_field": "kept out of normalization",
                    }
                ],
            )

            spec = TaskSpec.sample()
            spec.inputs.data.train_path = str(train_path)
            spec.inputs.data.validation_path = None
            spec.inputs.data.test_path = None
            spec.inputs.data.format = "json"

            pool = load_example_pool(
                spec,
                version="run_x__example_pool__001",
                data_format="json",
                train_path=train_path,
                validation_path=None,
                test_path=None,
                validation_split_seed=spec.task_id,
            )

            self.assertEqual(len(pool.examples), 1)
            example = pool.examples[0]
            self.assertEqual(example.id, "seed_001")
            self.assertEqual(example.source_type, "research")
            self.assertEqual(example.policy_version, "run_x__policy__003")
            self.assertEqual(example.keyphrases, ["threat"])
            self.assertEqual(example.provenance[0].url, "https://example.com/threats")

    def test_dataset_loader_fails_on_conflicting_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            _write_jsonl(train_path, [{"input": "duplicate example", "label": "APPROVED"}])
            _write_jsonl(validation_path, [{"input": "duplicate example", "label": "REJECTED"}])

            spec = TaskSpec.sample()
            spec.inputs.data.train_path = str(train_path)
            spec.inputs.data.validation_path = str(validation_path)
            spec.inputs.data.test_path = None

            with self.assertRaisesRegex(ValueError, "conflicting labels for duplicate example text"):
                load_example_pool(
                    spec,
                    version="run_x__example_pool__001",
                    data_format="jsonl",
                    train_path=train_path,
                    validation_path=validation_path,
                    test_path=None,
                    validation_split_seed=spec.task_id,
                )


class FetchTests(unittest.TestCase):
    def test_split_text_into_chunks_preserves_paragraph_boundaries(self) -> None:
        text = ("a" * 1500) + "\n\n" + ("b" * 1500)
        chunks = split_text_into_chunks(text, chunk_size=2000, chunk_overlap=0)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(set(chunks[0]) <= {"a"})
        self.assertTrue(set(chunks[1]) <= {"b"})

    def test_extract_text_from_html_strips_scripts(self) -> None:
        text = _extract_text_from_html("<script>bad()</script><p>good</p>")
        self.assertNotIn("bad()", text)
        self.assertIn("good", text)

    def test_merge_sources_updates_queries_for_existing_urls(self) -> None:
        merged = merge_sources(
            [
                SourceRecord(
                    source_id="src_0001",
                    url="https://example.com/a",
                    retrieved_at="2026-03-09T12:00:00Z",
                    queries=["first query"],
                    text="Original source text.",
                )
            ],
            [],
            query_by_url={"https://example.com/a": ["second query"]},
        )
        self.assertEqual(merged[0].queries, ["first query", "second query"])


class PromptAndEvaluationTests(unittest.TestCase):
    def test_parse_label_accepts_embedded_label(self) -> None:
        label = parse_label("Decision: REJECTED.", ["APPROVED", "REJECTED"])
        self.assertEqual(label, "REJECTED")

    def test_render_reader_messages_replaces_supported_placeholders(self) -> None:
        prompts = ReaderPrompts(
            version="run_x__reader_prompts__001",
            messages=[
                ReaderPromptMessage(role="system", content="Use policy:\n{{policy}}"),
                ReaderPromptMessage(
                    role="user",
                    content="Labels:\n{{labels}}\n\nInput:\n{{input}}",
                ),
            ],
        )

        rendered = render_reader_messages(
            prompts,
            policy_text="  Rule 1\nRule 2  ",
            input_text="hello there",
            labels=["APPROVED", "REJECTED"],
        )

        self.assertEqual(rendered[0]["content"], "Use policy:\nRule 1\nRule 2")
        self.assertIn("- APPROVED", rendered[1]["content"])
        self.assertIn("- REJECTED", rendered[1]["content"])
        self.assertIn("hello there", rendered[1]["content"])
        self.assertNotIn("{{policy}}", rendered[0]["content"])
        self.assertNotIn("{{labels}}", rendered[1]["content"])
        self.assertNotIn("{{input}}", rendered[1]["content"])

    def test_render_reader_messages_rejects_unknown_placeholders(self) -> None:
        prompts = ReaderPrompts(
            version="run_x__reader_prompts__001",
            messages=[ReaderPromptMessage(role="user", content="Unsupported {{foobar}} placeholder")],
        )

        with self.assertRaisesRegex(ValueError, "unsupported prompt placeholder"):
            render_reader_messages(
                prompts,
                policy_text="rule",
                input_text="hello",
                labels=["APPROVED", "REJECTED"],
            )

    def test_evaluate_policy_smoke(self) -> None:
        spec = TaskSpec.sample()
        policy = PolicyDoc.from_dict(
            {
                "version": "run_x__policy__001",
                "sections": [
                    {
                        "title": "Toy Moderation",
                        "content_type": "rules",
                        "rules": [{"text": "Reject threats. Approve harmless content."}],
                    }
                ],
            }
        )
        pool = ExamplePool.from_dict(
            {
                "version": "run_x__example_pool__001",
                "examples": [
                    {
                        "id": "ex1",
                        "input": "hello there",
                        "label": "APPROVED",
                        "split": "train",
                        "source_type": "seed",
                    },
                    {
                        "id": "ex2",
                        "input": "I will hurt you",
                        "label": "REJECTED",
                        "split": "train",
                        "source_type": "seed",
                    },
                    {
                        "id": "ex3",
                        "input": "good morning",
                        "label": "APPROVED",
                        "split": "validation",
                        "source_type": "seed",
                    },
                    {
                        "id": "ex4",
                        "input": "I will hurt you",
                        "label": "REJECTED",
                        "split": "validation",
                        "source_type": "seed",
                    },
                ],
            }
        )
        prompts = default_reader_prompts(version="run_x__reader_prompts__001")

        split_metrics, predictions = evaluate_policy(
            client=FakeClient(),
            model_config=spec.optimize.reader_model,
            policy=policy,
            example_pool=pool,
            prompts=prompts,
            label_space=spec.evaluate.label_space,
            metric_names=["f1"],
        )

        self.assertEqual(split_metrics["train"].metrics["f1"], 1.0)
        self.assertEqual(split_metrics["validation"].metrics["f1"], 1.0)
        self.assertEqual(predictions["validation"][1].prediction, "REJECTED")

    def test_evaluate_policy_uses_default_completion_budget(self) -> None:
        spec = TaskSpec.sample()
        policy = PolicyDoc.from_dict(
            {
                "version": "run_x__policy__001",
                "sections": [
                    {
                        "title": "Toy Moderation",
                        "content_type": "rules",
                        "rules": [{"text": "Reject threats. Approve harmless content."}],
                    }
                ],
            }
        )
        pool = ExamplePool.from_dict(
            {
                "version": "run_x__example_pool__001",
                "examples": [
                    {
                        "id": "ex1",
                        "input": "hello there",
                        "label": "APPROVED",
                        "split": "validation",
                        "source_type": "seed",
                    }
                ],
            }
        )
        prompts = default_reader_prompts(version="run_x__reader_prompts__001")
        captured_kwargs: list[dict[str, object]] = []

        class CapturingChatCompletions:
            def create(self, *, messages, **kwargs):
                captured_kwargs.append(dict(kwargs))
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="APPROVED"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=CapturingChatCompletions()))

        split_metrics, predictions = evaluate_policy(
            client=client,
            model_config=spec.optimize.reader_model,
            policy=policy,
            example_pool=pool,
            prompts=prompts,
            label_space=spec.evaluate.label_space,
            metric_names=["f1"],
        )

        self.assertEqual(split_metrics["validation"].metrics["f1"], 0.5)
        self.assertEqual(predictions["validation"][0].prediction, "APPROVED")
        self.assertEqual(captured_kwargs[0]["max_tokens"], 8192)
        self.assertEqual(captured_kwargs[0]["temperature"], 0)

    def test_macro_f1_uses_full_label_space(self) -> None:
        spec = TaskSpec.sample()
        policy = PolicyDoc.from_dict(
            {
                "version": "run_x__policy__001",
                "sections": [
                    {
                        "title": "Toy Moderation",
                        "content_type": "rules",
                        "rules": [{"text": "Reject threats. Approve harmless content."}],
                    }
                ],
            }
        )
        pool = ExamplePool.from_dict(
            {
                "version": "run_x__example_pool__001",
                "examples": [
                    {
                        "id": "ex1",
                        "input": "hello there",
                        "label": "APPROVED",
                        "split": "validation",
                        "source_type": "seed",
                    }
                ],
            }
        )
        prompts = default_reader_prompts(version="run_x__reader_prompts__001")

        split_metrics, _ = evaluate_policy(
            client=FakeClient(),
            model_config=spec.optimize.reader_model,
            policy=policy,
            example_pool=pool,
            prompts=prompts,
            label_space=spec.evaluate.label_space,
            metric_names=["f1"],
        )

        self.assertEqual(split_metrics["validation"].metrics["f1"], 0.5)

    def test_filter_candidate_rules_deduplicates_and_unions_provenance(self) -> None:
        filtered = filter_candidate_rules(
            [
                CandidateRuleRecord(
                    candidate_id="cand_001",
                    text="Reject direct threats of violence.",
                    keyphrases=["threats", "violence"],
                    sources=[SourcePointer(url="https://example.com/a", supporting_excerpt="Threats are abusive.")],
                    source_chunk_ids=["chunk_001"],
                ),
                CandidateRuleRecord(
                    candidate_id="cand_002",
                    text="Reject direct threats of violence!",
                    keyphrases=["direct threats"],
                    sources=[
                        SourcePointer(
                            url="https://example.com/b",
                            supporting_excerpt="Violent threats are prohibited.",
                        )
                    ],
                    source_chunk_ids=["chunk_002"],
                ),
            ]
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].source_candidate_ids, ["cand_001", "cand_002"])
        self.assertGreaterEqual(filtered[0].relevance_score, 0.6)
        self.assertEqual(len(filtered[0].sources), 2)


class ResearchAndRunTests(unittest.TestCase):
    def test_research_command_writes_research_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                    with patch(
                        "deep_policy_research.research.create_search_provider",
                        side_effect=lambda config, cache_dir: FakeSearchProvider(),
                    ):
                        with patch(
                            "deep_policy_research.research.create_url_fetcher",
                            side_effect=lambda cache_dir: FakeUrlFetcher(),
                        ):
                            manager, research, policy = run_research_command(
                                spec_path,
                                output_dir=root / "runs",
                                probe_all_models=True,
                            )

            self.assertEqual(manager.manifest.status, "completed")
            self.assertEqual(manager.manifest.current_step, "research")
            self.assertIsNotNone(manager.manifest.current_sources_version)
            self.assertIsNotNone(manager.manifest.current_chunks_version)
            self.assertIsNotNone(manager.manifest.current_candidate_rules_version)
            self.assertIsNotNone(manager.manifest.current_filtered_rules_version)
            self.assertIsNotNone(manager.manifest.current_research_version)
            self.assertTrue((manager.root_dir / "sources.jsonl").exists())
            self.assertTrue((manager.root_dir / "chunks.jsonl").exists())
            self.assertTrue((manager.root_dir / "candidate_rules.jsonl").exists())
            self.assertTrue((manager.root_dir / "filtered_rules.jsonl").exists())
            self.assertTrue((manager.root_dir / "research.json").exists())
            self.assertGreaterEqual(research.source_count, 2)
            self.assertGreaterEqual(research.filtered_rule_count, 2)
            self.assertGreaterEqual(len(policy.sections[0].rules), 1)

            resumed_manager, resumed_research, resumed_policy = resume_research_command(manager.root_dir)
            self.assertEqual(resumed_manager.root_dir, manager.root_dir)
            self.assertEqual(resumed_research.source_count, research.source_count)
            self.assertEqual(resumed_research.filtered_rule_count, research.filtered_rule_count)
            self.assertEqual(resumed_policy.version, policy.version)

    def test_resume_research_command_preserves_actual_query_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                with patch(
                    "deep_policy_research.research.create_search_provider",
                    side_effect=lambda config, cache_dir: FakeSearchProvider(),
                ):
                    with patch(
                        "deep_policy_research.research.create_url_fetcher",
                        side_effect=lambda cache_dir: FakeUrlFetcher(),
                    ):
                        with patch(
                            "deep_policy_research.research.generate_search_queries",
                            side_effect=lambda **kwargs: ["single query"],
                        ):
                            manager, research, _ = run_research_command(
                                spec_path,
                                output_dir=root / "runs",
                                probe_all_models=False,
                            )

            _, resumed_research, _ = resume_research_command(manager.root_dir)
            self.assertEqual(research.query_count, 1)
            self.assertEqual(resumed_research.query_count, research.query_count)

    def test_run_task_command_executes_research_optimize_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.optimize.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                    with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                        with patch("deep_policy_research.redteam.create_sync_client", side_effect=lambda config: RedteamAwareFakeClient()):
                            with patch(
                                "deep_policy_research.research.create_search_provider",
                                side_effect=lambda config, cache_dir: FakeSearchProvider(),
                            ):
                                with patch(
                                    "deep_policy_research.research.create_url_fetcher",
                                    side_effect=lambda cache_dir: FakeUrlFetcher(),
                                ):
                                    manager, result = run_task_command(
                                        spec_path,
                                        output_dir=root / "runs",
                                        probe_all_models=True,
                                    )

            self.assertEqual(manager.manifest.status, "completed")
            self.assertEqual(manager.manifest.current_step, "eval")
            self.assertIn("research", result.to_dict())
            self.assertIn("redteam", result.to_dict())
            self.assertIn("optimization", result.to_dict())
            self.assertIn("metrics", result.to_dict())
            self.assertIsNotNone(result.redteam)
            self.assertGreater(result.redteam["accepted_example_count"], 0)
            self.assertTrue(result.redteam["bootstrap_applied"])
            accepted_pool = ExamplePool.from_dict(json.loads((manager.root_dir / "example_pool.json").read_text()))
            self.assertGreaterEqual(len(accepted_pool.examples), 2)
            self.assertIsNotNone(manager.manifest.current_prompt_optimization_version)
            self.assertTrue((manager.root_dir / "prompt_optimization.json").exists())

            resumed_manager, resumed_result = resume_task_command(manager.root_dir)
            self.assertEqual(resumed_manager.root_dir, manager.root_dir)
            self.assertEqual(resumed_result.metrics, result.metrics)
            self.assertEqual(resumed_result.redteam, result.redteam)

    def test_cached_demo_spec_runs_end_to_end_with_fixture_search(self) -> None:
        spec_path = ROOT / "examples" / "cached_demo" / "task_spec.json"
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.optimize.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                    with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                        with patch("deep_policy_research.redteam.create_sync_client", side_effect=lambda config: RedteamAwareFakeClient()):
                            manager, result = run_task_command(
                                spec_path,
                                output_dir=root / "runs",
                                probe_all_models=False,
                            )

        self.assertEqual(manager.manifest.status, "completed")
        self.assertGreater(result.research["source_count"], 0)
        self.assertIsNotNone(result.redteam)
        self.assertIn("validation", result.metrics)

    def test_run_redteam_command_bootstraps_research_when_started_from_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                with patch("deep_policy_research.redteam.create_sync_client", side_effect=lambda config: RedteamAwareFakeClient()):
                    with patch(
                        "deep_policy_research.research.create_search_provider",
                        side_effect=lambda config, cache_dir: FakeSearchProvider(),
                    ):
                        with patch(
                            "deep_policy_research.research.create_url_fetcher",
                            side_effect=lambda cache_dir: FakeUrlFetcher(),
                        ):
                            manager, redteam, example_pool = run_redteam_command(
                                spec_path,
                                output_dir=root / "runs",
                                probe_all_models=False,
                            )

            self.assertEqual(manager.manifest.current_step, "redteam")
            self.assertEqual(manager.manifest.status, "completed")
            self.assertIsNotNone(manager.manifest.current_research_version)
            self.assertIsNotNone(manager.manifest.current_redteam_version)
            self.assertGreaterEqual(redteam.accepted_example_count, 2)
            self.assertTrue(any(example.split == "validation" for example in example_pool.examples))

    def test_redteam_command_bootstraps_examples_without_seed_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                with patch("deep_policy_research.redteam.create_sync_client", side_effect=lambda config: RedteamAwareFakeClient()):
                    with patch(
                        "deep_policy_research.research.create_search_provider",
                        side_effect=lambda config, cache_dir: FakeSearchProvider(),
                    ):
                        with patch(
                            "deep_policy_research.research.create_url_fetcher",
                            side_effect=lambda cache_dir: FakeUrlFetcher(),
                        ):
                            research_manager, _, _ = run_research_command(
                                spec_path,
                                output_dir=root / "runs",
                                probe_all_models=False,
                            )
                            manager, redteam, example_pool = resume_redteam_command(research_manager.root_dir)

            self.assertEqual(manager.manifest.current_step, "redteam")
            self.assertTrue(redteam.bootstrap_applied)
            self.assertGreaterEqual(redteam.accepted_example_count, 2)
            self.assertTrue(any(example.split == "validation" for example in example_pool.examples))

    def test_redteam_bootstrap_requires_acceptance_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                with patch(
                    "deep_policy_research.research.create_search_provider",
                    side_effect=lambda config, cache_dir: FakeSearchProvider(),
                ):
                    with patch(
                        "deep_policy_research.research.create_url_fetcher",
                        side_effect=lambda cache_dir: FakeUrlFetcher(),
                    ):
                        research_manager, _, _ = run_research_command(
                            spec_path,
                            output_dir=root / "runs",
                            probe_all_models=False,
                        )

            with patch("deep_policy_research.redteam.create_sync_client", side_effect=lambda config: OneExampleRedteamAwareFakeClient()):
                with self.assertRaisesRegex(ValueError, "required splits: validation"):
                    resume_redteam_command(research_manager.root_dir)

    def test_cli_run_and_resume_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = _write_research_ready_spec(root, include_examples=False)

            stdout = io.StringIO()
            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.optimize.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                    with patch("deep_policy_research.research.create_sync_client", side_effect=lambda config: ResearchAwareFakeClient()):
                        with patch("deep_policy_research.redteam.create_sync_client", side_effect=lambda config: RedteamAwareFakeClient()):
                            with patch(
                                "deep_policy_research.research.create_search_provider",
                                side_effect=lambda config, cache_dir: FakeSearchProvider(),
                            ):
                                with patch(
                                    "deep_policy_research.research.create_url_fetcher",
                                    side_effect=lambda cache_dir: FakeUrlFetcher(),
                                ):
                                    with contextlib.redirect_stdout(stdout):
                                        exit_code = cli_main(["run", "--spec", str(spec_path), "--output-dir", str(root / "runs")])
            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertIn("research", payload)
            self.assertIn("redteam", payload)
            self.assertIn("optimization", payload)
            self.assertIn("metrics", payload)

            resume_stdout = io.StringIO()
            with contextlib.redirect_stdout(resume_stdout):
                exit_code = cli_main(["resume", "--run-dir", payload["run_dir"]])
            self.assertEqual(exit_code, 0)
            resumed = json.loads(resume_stdout.getvalue())
            self.assertIn("research", resumed)
            self.assertIn("redteam", resumed)
            self.assertIn("optimization", resumed)
            self.assertIn("metrics", resumed)


class BenchmarkAndCliTests(unittest.TestCase):
    def test_openai_moderation_adapter_lists_expected_task(self) -> None:
        task_ids = {task.task_id for task in list_openai_moderation_tasks(ROOT / "data" / "openai_moderation_resplit")}
        self.assertIn("openai_moderation__harassment", task_ids)

        task = get_openai_moderation_task(
            "openai_moderation__harassment",
            root_dir=ROOT / "data" / "openai_moderation_resplit",
        )
        self.assertEqual(task.domain_name, "Harassment")

    def test_openai_moderation_adapter_default_root_is_cwd_independent(self) -> None:
        original_cwd = Path.cwd()
        try:
            os.chdir("/tmp")
            task_ids = {task.task_id for task in list_openai_moderation_tasks()}
        finally:
            os.chdir(original_cwd)
        self.assertIn("openai_moderation__harassment", task_ids)

    def test_build_task_spec_uses_manifest_paths(self) -> None:
        model = ModelConfig(model="fake-model", base_url="http://localhost:8223/v1", api_key_env="FAKE_KEY")
        spec = build_task_spec(
            task_id="openai_moderation__harassment",
            model=model,
            initial_policy_text="Reject harassment.",
            root_dir=ROOT / "data" / "openai_moderation_resplit",
        )
        self.assertEqual(spec.task_id, "openai_moderation__harassment")
        self.assertEqual(spec.evaluate.label_space, ["APPROVED", "REJECTED"])
        self.assertTrue(spec.inputs.data.train_path.endswith("harassment/train.jsonl"))

    def test_cli_eval_resume_and_inspect_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"input": "hello there", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )
            _write_jsonl(validation_path, [{"input": "good morning", "label": "APPROVED"}])

            spec = TaskSpec.sample().to_dict()
            spec["research"]["model"] = {
                "model": "fake-model",
                "base_url": "http://localhost:8223/v1",
                "api_key_env": "FAKE_KEY",
                "client_kwargs": {},
                "request_defaults": {},
            }
            spec["redteam"]["model"] = dict(spec["research"]["model"])
            spec["optimize"]["reader_model"] = dict(spec["research"]["model"])
            spec["inputs"]["data"] = {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "format": "jsonl",
                "bootstrap_train_ratio": 0.8,
            }
            spec_path = root / "spec.json"
            _write_json(spec_path, spec)

            stdout = io.StringIO()
            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: FakeClient()):
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli_main(["eval", "--spec", str(spec_path), "--output-dir", str(root / "runs")])
            self.assertEqual(exit_code, 0)

            payload = json.loads(stdout.getvalue())
            run_dir = Path(payload["run_dir"])
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "metrics.json").exists())

            inspect_stdout = io.StringIO()
            with contextlib.redirect_stdout(inspect_stdout):
                exit_code = cli_main(["inspect", "run", str(run_dir)])
            self.assertEqual(exit_code, 0)
            inspected = json.loads(inspect_stdout.getvalue())
            self.assertEqual(inspected["manifest"]["status"], "completed")

            resume_stdout = io.StringIO()
            with contextlib.redirect_stdout(resume_stdout):
                exit_code = cli_main(["resume", "--run-dir", str(run_dir)])
            self.assertEqual(exit_code, 0)
            resumed = json.loads(resume_stdout.getvalue())
            self.assertIn("metrics", resumed)

    def test_optimize_reader_accepts_better_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"input": "hello there", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )
            _write_jsonl(
                validation_path,
                [
                    {"input": "good morning", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )

            spec = TaskSpec.sample().to_dict()
            spec["research"]["model"] = {
                "model": "fake-model",
                "base_url": "http://localhost:8223/v1",
                "api_key_env": "FAKE_KEY",
                "client_kwargs": {},
                "request_defaults": {},
            }
            spec["redteam"]["model"] = dict(spec["research"]["model"])
            spec["optimize"]["reader_model"] = dict(spec["research"]["model"])
            spec["inputs"]["data"] = {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "format": "jsonl",
                "bootstrap_train_ratio": 0.8,
            }
            spec_path = root / "spec.json"
            _write_json(spec_path, spec)

            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.optimize.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                    manager, optimization, metrics = optimize_reader_command(
                        spec_path,
                        output_dir=root / "runs",
                        probe_all_models=False,
                    )

            self.assertTrue(optimization.improved)
            self.assertGreater(len(optimization.candidate_prompt_versions), 0)
            self.assertGreater(optimization.best_score, optimization.baseline_score)
            self.assertEqual(manager.manifest.current_step, "optimize")
            self.assertEqual(manager.manifest.current_prompt_version, optimization.best_prompt_version)
            self.assertEqual(manager.manifest.current_metrics_version, optimization.best_metrics_version)
            self.assertEqual(manager.manifest.current_prompt_optimization_version, optimization.version)

            accepted_prompts = ReaderPrompts.from_dict(json.loads((manager.root_dir / "reader_prompts.json").read_text()))
            self.assertEqual(accepted_prompts.version, optimization.best_prompt_version)
            self.assertEqual(metrics.prompt_version, optimization.best_prompt_version)

    def test_resume_completed_optimize_run_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"input": "hello there", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )
            _write_jsonl(
                validation_path,
                [
                    {"input": "good morning", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )

            spec = TaskSpec.sample().to_dict()
            spec["research"]["model"] = {
                "model": "fake-model",
                "base_url": "http://localhost:8223/v1",
                "api_key_env": "FAKE_KEY",
                "client_kwargs": {},
                "request_defaults": {},
            }
            spec["redteam"]["model"] = dict(spec["research"]["model"])
            spec["optimize"]["reader_model"] = dict(spec["research"]["model"])
            spec["inputs"]["data"] = {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "format": "jsonl",
                "bootstrap_train_ratio": 0.8,
            }
            spec_path = root / "spec.json"
            _write_json(spec_path, spec)

            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.optimize.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                    manager, optimization, metrics = optimize_reader_command(
                        spec_path,
                        output_dir=root / "runs",
                        probe_all_models=False,
                    )

            resumed_manager, resumed_optimization, resumed_metrics = resume_optimize_reader_command(manager.root_dir)

            self.assertEqual(resumed_manager.root_dir, manager.root_dir)
            self.assertEqual(resumed_optimization.to_dict(), optimization.to_dict())
            self.assertEqual(resumed_metrics.version, optimization.best_metrics_version)
            self.assertEqual(resumed_metrics.to_dict(), metrics.to_dict())
            self.assertTrue(resumed_optimization.improved)
            self.assertGreater(len(resumed_optimization.candidate_prompt_versions), 0)

    def test_cli_optimize_reader_and_resume_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            _write_jsonl(
                train_path,
                [
                    {"input": "hello there", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )
            _write_jsonl(
                validation_path,
                [
                    {"input": "good morning", "label": "APPROVED"},
                    {"input": "I will hurt you", "label": "REJECTED"},
                ],
            )

            spec = TaskSpec.sample().to_dict()
            spec["research"]["model"] = {
                "model": "fake-model",
                "base_url": "http://localhost:8223/v1",
                "api_key_env": "FAKE_KEY",
                "client_kwargs": {},
                "request_defaults": {},
            }
            spec["redteam"]["model"] = dict(spec["research"]["model"])
            spec["optimize"]["reader_model"] = dict(spec["research"]["model"])
            spec["inputs"]["data"] = {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "format": "jsonl",
                "bootstrap_train_ratio": 0.8,
            }
            spec_path = root / "spec.json"
            _write_json(spec_path, spec)

            stdout = io.StringIO()
            with patch("deep_policy_research.eval.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                with patch("deep_policy_research.optimize.create_sync_client", side_effect=lambda config: PromptAwareFakeClient()):
                    with contextlib.redirect_stdout(stdout):
                        exit_code = cli_main(
                            ["optimize-reader", "--spec", str(spec_path), "--output-dir", str(root / "runs")]
                        )
            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["optimization"]["improved"])

            resume_stdout = io.StringIO()
            with contextlib.redirect_stdout(resume_stdout):
                exit_code = cli_main(["resume", "--run-dir", payload["run_dir"]])
            self.assertEqual(exit_code, 0)
            resumed = json.loads(resume_stdout.getvalue())
            self.assertIn("optimization", resumed)
            self.assertIn("metrics", resumed)
            self.assertEqual(resumed["optimization"], payload["optimization"])


class OpenAIClientTests(unittest.TestCase):
    def test_probe_model_works_with_fake_client(self) -> None:
        probe_model(FakeClient(), ModelConfig(model="fake-model", base_url="http://localhost:8223/v1", api_key_env="FAKE_KEY"))

    def test_qwen_requests_do_not_override_thinking_by_default(self) -> None:
        request = chat_completion_request(
            ModelConfig(model="Qwen/Qwen3.5-9B", base_url="http://localhost:8223/v1", api_key_env="FAKE_KEY"),
            [{"role": "user", "content": "hello"}],
        )
        self.assertNotIn("extra_body", request)


class FakeChatCompletions:
    def create(self, *, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Reply with OK." in prompt:
            content = "OK"
        else:
            input_text = _extract_input_text(prompt)
            lowered = input_text.lower()
            if any(token in lowered for token in ["hurt", "kill", "threat"]):
                content = "REJECTED"
            else:
                content = "APPROVED"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class PromptAwareFakeChatCompletions:
    def create(self, *, messages, **kwargs):
        combined = "\n\n".join(message["content"] for message in messages)
        if "Reply with OK." in combined:
            content = "OK"
        else:
            input_text = _extract_input_text(combined)
            lowered_input = input_text.lower()
            lowered_prompt = combined.lower()
            if any(token in lowered_input for token in ["hurt", "kill", "threat"]):
                if "final line" in lowered_prompt or "answer in two lines" in lowered_prompt:
                    content = "REJECTED"
                else:
                    content = "APPROVED"
            else:
                content = "APPROVED"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class PromptAwareFakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=PromptAwareFakeChatCompletions())


class ResearchAwareFakeChatCompletions:
    def create(self, *, messages, **kwargs):
        combined = "\n\n".join(message["content"] for message in messages)
        if "Reply with OK." in combined:
            content = "OK"
        elif "Generate exactly" in combined and "web search queries" in combined:
            count = 3
            match = re.search(r"Generate exactly (\d+) web search queries", combined)
            if match:
                count = int(match.group(1))
            content = json.dumps(
                [
                    "targeted harassment official policy guidance",
                    "violent threats safety definitions",
                    "abusive conduct examples policy",
                ][:count]
            )
        elif 'Each item must have:\n- "text": a concise policy rule directly supported by the chunk' in combined:
            if "Threatening violence toward a person is abusive conduct." in combined:
                content = json.dumps(
                    [
                        {
                            "text": "Reject direct threats of violence toward a person.",
                            "keyphrases": ["direct threats", "violence"],
                            "supporting_excerpt": "Threatening violence toward a person is abusive conduct.",
                        }
                    ]
                )
            elif "Targeted abusive insults and demeaning slurs are harassment." in combined:
                content = json.dumps(
                    [
                        {
                            "text": "Reject targeted abusive insults and demeaning slurs.",
                            "keyphrases": ["targeted abuse", "slurs"],
                            "supporting_excerpt": "Targeted abusive insults and demeaning slurs are harassment.",
                        }
                    ]
                )
            else:
                content = "[]"
        elif "You are organizing policy rules into a readable policy document." in combined:
            rule_ids = re.findall(r"rule_\d{5}", combined)
            content = json.dumps(
                [
                    {
                        "title": "Abusive Conduct",
                        "summary": "Grounded rules for harmful and abusive user behavior.",
                        "rule_ids": list(dict.fromkeys(rule_ids)),
                    }
                ]
            )
        else:
            content = "[]"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class ResearchAwareFakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=ResearchAwareFakeChatCompletions())


class RedteamAwareFakeChatCompletions:
    def create(self, *, messages, **kwargs):
        combined = "\n\n".join(message["content"] for message in messages)
        if "Reply with OK." in combined:
            content = "OK"
        elif "Generate up to" in combined and "adversarial or coverage-expanding labeled examples" in combined:
            content = json.dumps(
                [
                    {
                        "input": "I will hurt you tonight.",
                        "label": "REJECTED",
                        "url": "https://example.com/threats",
                        "supporting_excerpt": "Threatening violence toward a person is abusive conduct.",
                        "keyphrases": ["threat", "violence"],
                    },
                    {
                        "input": "You disgusting idiot and everyone hates you.",
                        "label": "REJECTED",
                        "url": "https://example.com/harassment",
                        "supporting_excerpt": "Targeted abusive insults and demeaning slurs are harassment.",
                        "keyphrases": ["harassment", "slurs"],
                    },
                    {
                        "input": "Good morning, I hope you are well.",
                        "label": "APPROVED",
                        "url": "https://example.com/harassment",
                        "supporting_excerpt": "Harassing attacks aimed at a person are not allowed.",
                        "keyphrases": ["harmless"],
                    },
                ]
            )
        elif "You are checking whether a proposed label is clear and externally supported." in combined:
            label_match = re.search(r"Proposed label: ([A-Z_]+)", combined)
            label = label_match.group(1) if label_match else "APPROVED"
            content = json.dumps({"label": label, "clear": True})
        else:
            input_text = _extract_input_text(combined)
            lowered_input = input_text.lower()
            lowered_prompt = combined.lower()
            if any(token in lowered_input for token in ["hurt", "kill", "threat"]):
                if "final line" in lowered_prompt or "answer in two lines" in lowered_prompt:
                    content = "REJECTED"
                else:
                    content = "APPROVED"
            elif any(token in lowered_input for token in ["idiot", "hates you", "disgusting"]):
                content = "APPROVED"
            else:
                content = "APPROVED"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class RedteamAwareFakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=RedteamAwareFakeChatCompletions())


class OneExampleRedteamAwareFakeChatCompletions:
    def create(self, *, messages, **kwargs):
        combined = "\n\n".join(message["content"] for message in messages)
        if "Reply with OK." in combined:
            content = "OK"
        elif "Generate up to" in combined and "adversarial or coverage-expanding labeled examples" in combined:
            content = json.dumps(
                [
                    {
                        "input": "I will hurt you tonight.",
                        "label": "REJECTED",
                        "url": "https://example.com/threats",
                        "supporting_excerpt": "Threatening violence toward a person is abusive conduct.",
                        "keyphrases": ["threat"],
                    }
                ]
            )
        elif "You are checking whether a proposed label is clear and externally supported." in combined:
            content = json.dumps({"label": "REJECTED", "clear": True})
        else:
            content = "APPROVED"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class OneExampleRedteamAwareFakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=OneExampleRedteamAwareFakeChatCompletions())


class FakeSearchProvider:
    def search(self, query: str, *, num_results: int | None = None) -> list[SearchResult]:
        results = [
            SearchResult(
                query=query,
                url="https://example.com/threats",
                rank=1,
                title="Threats guidance",
                snippet="Threats should be rejected.",
            ),
            SearchResult(
                query=query,
                url="https://example.com/harassment",
                rank=2,
                title="Harassment guidance",
                snippet="Targeted abuse should be rejected.",
            ),
        ]
        if num_results is None:
            return results
        return results[:num_results]


class FakeUrlFetcher:
    def fetch(self, url: str) -> FetchedDocument | None:
        documents = {
            "https://example.com/threats": FetchedDocument(
                url="https://example.com/threats",
                retrieved_at="2026-03-09T12:00:00Z",
                text=(
                    "Threatening violence toward a person is abusive conduct.\n\n"
                    "Content that promises physical harm should be treated as a threat."
                ),
            ),
            "https://example.com/harassment": FetchedDocument(
                url="https://example.com/harassment",
                retrieved_at="2026-03-09T12:00:01Z",
                text=(
                    "Targeted abusive insults and demeaning slurs are harassment.\n\n"
                    "Harassing attacks aimed at a person are not allowed."
                ),
            ),
        }
        return documents.get(url)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _extract_input_text(prompt: str) -> str:
    if "Input:\n" not in prompt:
        return prompt
    return prompt.rsplit("Input:\n", 1)[1].split("\n\n", 1)[0]


def _write_research_ready_spec(root: Path, *, include_examples: bool) -> Path:
    spec = TaskSpec.sample().to_dict()
    fake_model = {
        "model": "fake-model",
        "base_url": "http://localhost:8223/v1",
        "api_key_env": "FAKE_KEY",
        "client_kwargs": {},
        "request_defaults": {},
    }
    spec["research"]["model"] = fake_model
    spec["research"]["max_rounds"] = 1
    spec["research"]["iterations_per_round"] = 1
    spec["research"]["queries_per_iteration"] = 2
    spec["research"]["pages_per_query"] = 2
    spec["redteam"]["model"] = dict(fake_model)
    spec["optimize"]["reader_model"] = dict(fake_model)
    spec["inputs"]["initial_policy"] = {"text": None, "policy_doc_path": None}
    if include_examples:
        train_path = root / "train.jsonl"
        validation_path = root / "validation.jsonl"
        _write_jsonl(
            train_path,
            [
                {"input": "hello there", "label": "APPROVED"},
                {"input": "I will hurt you", "label": "REJECTED"},
            ],
        )
        _write_jsonl(
            validation_path,
            [
                {"input": "good morning", "label": "APPROVED"},
                {"input": "You are worthless and I will hurt you", "label": "REJECTED"},
            ],
        )
        spec["inputs"]["data"] = {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "format": "jsonl",
            "bootstrap_train_ratio": 0.8,
        }
    else:
        spec["inputs"]["data"] = {
            "train_path": None,
            "validation_path": None,
            "test_path": None,
            "format": "jsonl",
            "bootstrap_train_ratio": 0.8,
        }
    spec_path = root / "spec.json"
    _write_json(spec_path, spec)
    return spec_path


def _make_generated_example(example_id: str, text: str, label: str):
    from deep_policy_research.examples import ExampleRecord

    return ExampleRecord(
        id=example_id,
        input=text,
        label=label,
        split="train",
        source_type="redteam",
        provenance=[SourcePointer(url="https://example.com/test", supporting_excerpt="support")],
        keyphrases=[],
        policy_version="run_x__policy__001",
    )


if __name__ == "__main__":
    unittest.main()
