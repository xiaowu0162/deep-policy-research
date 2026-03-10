from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ChunkRecord,
    FilteredRuleRecord,
    ReaderPrompts,
    RedteamArtifact,
    SourceRecord,
)
from .config import ResolvedTaskSpec, load_task_spec
from .datasets import has_train_or_validation_examples, merge_generated_examples
from .eval import (
    _load_or_write_example_pool,
    _load_or_write_policy,
    _load_or_write_prompts,
    _optional_path_from_run_config,
    _probe_models,
    _read_validation_split_seed,
    parse_label,
)
from .examples import ExamplePool, ExampleRecord
from .extract import _parse_json_payload, _strip_reasoning_text
from .openai_client import create_chat_completion, create_sync_client, extract_text_from_chat_response, probe_model
from .policy import PolicyDoc, SourcePointer
from .prompts import render_policy_doc, render_reader_messages
from .research import _run_or_resume_research
from .run_manager import RunManager
from .spec import ModelConfig, TaskSpec


DEFAULT_REDTEAM_CANDIDATE_COUNT = 8
DEFAULT_REDTEAM_MAX_TOKENS = 4096
LABEL_CHECK_PERSPECTIVES = (
    "literal policy reading",
    "safety-conservative reading",
    "adversarial skeptical reading",
)


@dataclass(slots=True)
class RedteamCandidate:
    input: str
    label: str
    url: str
    supporting_excerpt: str
    keyphrases: list[str]


RedteamResult = RedteamArtifact


def run_redteam_command(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    probe_all_models: bool = False,
) -> tuple[RunManager, RedteamResult, ExamplePool]:
    resolved = load_task_spec(spec_path)
    run_config = resolved.to_run_config_dict()
    run_config["entrypoint"] = "redteam"
    manager = RunManager.create(
        output_dir=output_dir,
        spec=resolved.spec,
        run_config=run_config,
        current_step="research",
    )
    return _run_or_resume_redteam_entrypoint(manager, resolved, probe_all_models=probe_all_models)


def resume_redteam_command(run_dir: str | Path) -> tuple[RunManager, RedteamResult, ExamplePool]:
    manager = RunManager.load(run_dir)
    with manager.spec_path.open("r", encoding="utf-8") as handle:
        spec = TaskSpec.from_dict(json.load(handle))
    if manager.manifest.status == "completed" and manager.manifest.current_redteam_version is not None:
        result = manager.load_artifact("redteam", RedteamArtifact)
        example_pool = manager.load_artifact("example_pool", ExamplePool, version=result.ending_example_pool_version)
        return manager, result, example_pool

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
    return _run_or_resume_redteam_entrypoint(manager, resolved, probe_all_models=False)


def _run_or_resume_redteam_entrypoint(
    manager: RunManager,
    resolved: ResolvedTaskSpec,
    *,
    probe_all_models: bool,
) -> tuple[RunManager, RedteamResult, ExamplePool]:
    if _redteam_requires_research_state(manager):
        manager, _, _ = _run_or_resume_research(manager, resolved, probe_all_models=probe_all_models)
        probe_all_models = False
    return _run_or_resume_redteam(manager, resolved, probe_all_models=probe_all_models)


def _run_or_resume_redteam(
    manager: RunManager,
    resolved: ResolvedTaskSpec,
    *,
    probe_all_models: bool,
) -> tuple[RunManager, RedteamResult, ExamplePool]:
    try:
        manager.update_status("running", current_step="redteam")
        if probe_all_models:
            _probe_models(resolved.spec, probe_all_models=True)
        else:
            redteam_probe_client = create_sync_client(resolved.spec.redteam.model)
            probe_model(redteam_probe_client, resolved.spec.redteam.model)
            reader_probe_client = create_sync_client(resolved.spec.optimize.reader_model)
            probe_model(reader_probe_client, resolved.spec.optimize.reader_model)

        policy = _load_or_write_policy(manager, resolved)
        example_pool = _load_or_write_example_pool(manager, resolved)
        prompts = _load_or_write_prompts(manager)
        sources = _load_rows_if_present(manager, "sources", SourceRecord)
        filtered_rules = _load_rows_if_present(manager, "filtered_rules", FilteredRuleRecord)

        redteam_client = create_sync_client(resolved.spec.redteam.model)
        reader_client = create_sync_client(resolved.spec.optimize.reader_model)

        candidates = _generate_redteam_candidates(
            client=redteam_client,
            model_config=resolved.spec.redteam.model,
            domain_name=resolved.spec.domain.name,
            domain_description=resolved.spec.domain.description,
            policy=policy,
            example_pool=example_pool,
            filtered_rules=filtered_rules,
            sources=sources,
            label_space=resolved.spec.evaluate.label_space,
        )
        accepted_examples, rejected_count = _accept_redteam_candidates(
            candidates,
            redteam_client=redteam_client,
            redteam_model=resolved.spec.redteam.model,
            reader_client=reader_client,
            reader_model=resolved.spec.optimize.reader_model,
            policy=policy,
            prompts=prompts,
            label_space=resolved.spec.evaluate.label_space,
            sources=sources,
            existing_example_pool=example_pool,
        )

        bootstrap_required = not has_train_or_validation_examples(example_pool)

        updated_example_pool = example_pool
        if accepted_examples:
            updated_example_pool, accepted_examples, merge_rejections = merge_generated_examples(
                example_pool,
                accepted_examples,
                version=manager.new_version("example_pool"),
                train_ratio=resolved.spec.inputs.data.bootstrap_train_ratio,
                split_seed=resolved.validation_split_seed,
            )
            rejected_count += len(merge_rejections)
            manager.write_artifact("example_pool", updated_example_pool)
        if bootstrap_required:
            missing_splits = _missing_bootstrap_splits(
                updated_example_pool,
                acceptance_split=resolved.spec.optimize.acceptance_split,
            )
            if missing_splits:
                joined = ", ".join(missing_splits)
                raise ValueError(f"redteam bootstrap did not produce the required splits: {joined}")

        result = RedteamResult(
            version=manager.new_version("redteam"),
            policy_version=policy.version,
            prompt_version=prompts.version,
            starting_example_pool_version=example_pool.version,
            ending_example_pool_version=updated_example_pool.version,
            generated_candidate_count=len(candidates),
            accepted_example_count=len(accepted_examples),
            rejected_example_count=rejected_count,
            accepted_example_ids=[example.id for example in accepted_examples],
            bootstrap_applied=bootstrap_required and bool(accepted_examples),
        )
        manager.record_artifact("redteam", result, accept=True)
        manager.update_status("completed", current_step="redteam")
        return manager, result, updated_example_pool
    except Exception:
        manager.update_status("failed", current_step="redteam")
        raise


def _generate_redteam_candidates(
    *,
    client: Any,
    model_config: ModelConfig,
    domain_name: str,
    domain_description: str,
    policy: PolicyDoc,
    example_pool: ExamplePool,
    filtered_rules: list[FilteredRuleRecord],
    sources: list[SourceRecord],
    label_space: list[str],
) -> list[RedteamCandidate]:
    evidence_context = _build_evidence_context(filtered_rules, sources)
    if not evidence_context:
        return []

    current_examples = _summarize_examples(example_pool)
    policy_text = render_policy_doc(policy)
    prompt = f"""You are a red-team data generator for a policy reader model.

Generate up to {DEFAULT_REDTEAM_CANDIDATE_COUNT} adversarial or coverage-expanding labeled examples that are grounded in the evidence.

Domain name: {domain_name}
Domain description: {domain_description}
Allowed labels: {json.dumps(label_space)}

Current policy:
{policy_text}

Existing examples summary:
{current_examples}

Evidence:
{evidence_context}

Return a JSON array. Each item must have:
- "input": example text
- "label": one allowed label
- "url": supporting source URL
- "supporting_excerpt": an exact supporting quote from the evidence
- "keyphrases": a short list of strings

Focus on cases likely to challenge the current reader setup. Avoid duplicates of the existing examples summary."""
    raw_output = _run_json_prompt(client, model_config, prompt, max_tokens=DEFAULT_REDTEAM_MAX_TOKENS)
    parsed = _parse_json_payload(raw_output)
    return _coerce_redteam_candidates(parsed, label_space)


def _accept_redteam_candidates(
    candidates: list[RedteamCandidate],
    *,
    redteam_client: Any,
    redteam_model: ModelConfig,
    reader_client: Any,
    reader_model: ModelConfig,
    policy: PolicyDoc,
    prompts: ReaderPrompts,
    label_space: list[str],
    sources: list[SourceRecord],
    existing_example_pool: ExamplePool,
) -> tuple[list[ExampleRecord], int]:
    accepted: list[ExampleRecord] = []
    rejected = 0
    source_lookup = {source.url: source for source in sources}
    seen_inputs = {_normalize_input(example.input) for example in existing_example_pool.examples}
    accepted_inputs: set[str] = set()

    for index, candidate in enumerate(candidates, start=1):
        normalized_input = _normalize_input(candidate.input)
        if not normalized_input or normalized_input in seen_inputs or normalized_input in accepted_inputs:
            rejected += 1
            continue
        if not _candidate_has_external_support(candidate, source_lookup):
            rejected += 1
            continue
        if not _label_is_clear(
            candidate,
            client=redteam_client,
            model_config=redteam_model,
            policy=policy,
            label_space=label_space,
        ):
            rejected += 1
            continue
        prediction = _predict_reader_label(
            reader_client,
            reader_model,
            policy=policy,
            prompts=prompts,
            input_text=candidate.input,
            label_space=label_space,
        )
        if prediction == candidate.label:
            rejected += 1
            continue

        example = ExampleRecord(
            id=_generated_redteam_id(candidate.input, candidate.label, index),
            input=candidate.input,
            label=candidate.label,
            split="train",
            source_type="redteam",
            provenance=[SourcePointer(url=candidate.url, supporting_excerpt=candidate.supporting_excerpt)],
            keyphrases=candidate.keyphrases,
            policy_version=policy.version,
        )
        accepted.append(example)
        accepted_inputs.add(normalized_input)

    return accepted, rejected


def _label_is_clear(
    candidate: RedteamCandidate,
    *,
    client: Any,
    model_config: ModelConfig,
    policy: PolicyDoc,
    label_space: list[str],
) -> bool:
    policy_text = render_policy_doc(policy)
    for perspective in LABEL_CHECK_PERSPECTIVES:
        prompt = f"""You are checking whether a proposed label is clear and externally supported.

Perspective: {perspective}
Allowed labels: {json.dumps(label_space)}
Proposed label: {candidate.label}
Source URL: {candidate.url}
Supporting excerpt: {candidate.supporting_excerpt}

Current policy:
{policy_text}

Candidate input:
{candidate.input}

Return a JSON object with keys:
- "label": the best label from the allowed labels
- "clear": true if the label is unambiguous and well supported, otherwise false"""
        raw_output = _run_json_prompt(client, model_config, prompt, max_tokens=512)
        parsed = _parse_json_payload(raw_output)
        if not isinstance(parsed, dict):
            return False
        label = str(parsed.get("label") or "").strip()
        clear = parsed.get("clear")
        if label != candidate.label or clear is not True:
            return False
    return True


def _predict_reader_label(
    client: Any,
    model_config: ModelConfig,
    *,
    policy: PolicyDoc,
    prompts: ReaderPrompts,
    input_text: str,
    label_space: list[str],
) -> str | None:
    messages = render_reader_messages(
        prompts,
        policy_text=render_policy_doc(policy),
        input_text=input_text,
        labels=label_space,
    )
    overrides: dict[str, Any] = {"temperature": 0}
    if (
        "max_tokens" not in model_config.request_defaults
        and "max_completion_tokens" not in model_config.request_defaults
    ):
        overrides["max_tokens"] = 8192
    response = create_chat_completion(client, model_config, messages, **overrides)
    return parse_label(extract_text_from_chat_response(response).strip(), label_space)


def _candidate_has_external_support(
    candidate: RedteamCandidate,
    source_lookup: dict[str, SourceRecord],
) -> bool:
    source = source_lookup.get(candidate.url)
    if source is None or not candidate.supporting_excerpt.strip():
        return False
    compact_excerpt = " ".join(candidate.supporting_excerpt.split()).lower()
    compact_source = " ".join(source.text.split()).lower()
    return compact_excerpt in compact_source


def _run_json_prompt(client: Any, model_config: ModelConfig, prompt: str, *, max_tokens: int) -> str:
    overrides: dict[str, Any] = {"temperature": 0}
    if (
        "max_tokens" not in model_config.request_defaults
        and "max_completion_tokens" not in model_config.request_defaults
    ):
        overrides["max_tokens"] = max_tokens
    response = create_chat_completion(
        client,
        model_config,
        [
            {"role": "system", "content": "Return only valid JSON that matches the requested schema."},
            {"role": "user", "content": prompt},
        ],
        **overrides,
    )
    return _strip_reasoning_text(extract_text_from_chat_response(response)).strip()


def _coerce_redteam_candidates(payload: Any, label_space: list[str]) -> list[RedteamCandidate]:
    if isinstance(payload, dict):
        for key in ["examples", "items", "data", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        return []

    candidates: list[RedteamCandidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = str(item.get("input") or "").strip()
        label = str(item.get("label") or "").strip()
        url = str(item.get("url") or "").strip()
        supporting_excerpt = str(item.get("supporting_excerpt") or "").strip()
        raw_keyphrases = item.get("keyphrases", [])
        if label not in label_space or not text or not url or not supporting_excerpt:
            continue
        if isinstance(raw_keyphrases, str):
            keyphrases = [raw_keyphrases.strip()] if raw_keyphrases.strip() else []
        elif isinstance(raw_keyphrases, list):
            keyphrases = [str(value).strip() for value in raw_keyphrases if str(value).strip()]
        else:
            keyphrases = []
        candidates.append(
            RedteamCandidate(
                input=text,
                label=label,
                url=url,
                supporting_excerpt=supporting_excerpt,
                keyphrases=list(dict.fromkeys(keyphrases)),
            )
        )
    return candidates


def _build_evidence_context(filtered_rules: list[FilteredRuleRecord], sources: list[SourceRecord]) -> str:
    evidence_lines: list[str] = []
    if filtered_rules:
        for rule in filtered_rules[:12]:
            excerpt = rule.sources[0].supporting_excerpt if rule.sources else None
            url = rule.sources[0].url if rule.sources else None
            evidence_lines.append(
                json.dumps(
                    {
                        "rule_id": rule.rule_id,
                        "rule": rule.text,
                        "url": url,
                        "supporting_excerpt": excerpt,
                    },
                    sort_keys=True,
                )
            )
    elif sources:
        for source in sources[:8]:
            evidence_lines.append(
                json.dumps(
                    {
                        "url": source.url,
                        "supporting_excerpt": source.text[:320],
                    },
                    sort_keys=True,
                )
            )
    return "\n".join(evidence_lines)


def _redteam_requires_research_state(manager: RunManager) -> bool:
    if manager.manifest.current_policy_version is None:
        return True
    return manager.manifest.current_filtered_rules_version is None and manager.manifest.current_sources_version is None


def _missing_bootstrap_splits(example_pool: ExamplePool, *, acceptance_split: str) -> list[str]:
    required_splits = ["train"]
    if acceptance_split not in required_splits:
        required_splits.append(acceptance_split)
    available_splits = {example.split for example in example_pool.examples}
    return [split for split in required_splits if split not in available_splits]


def _summarize_examples(example_pool: ExamplePool) -> str:
    if not example_pool.examples:
        return "No existing examples."
    lines = []
    for example in example_pool.examples[:20]:
        lines.append(f"- [{example.label}] {example.input}")
    return "\n".join(lines)


def _generated_redteam_id(text: str, label: str, index: int) -> str:
    digest = hashlib.sha256(f"redteam:{label}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"rt_{index:04d}_{digest}"


def _normalize_input(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _load_rows_if_present(manager: RunManager, kind: str, row_cls):
    descriptor_attr = f"current_{kind}_version"
    if getattr(manager.manifest, descriptor_attr) is None:
        return []
    return manager.load_rows_artifact(kind, row_cls)
