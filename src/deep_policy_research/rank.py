from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any

from .artifacts import CandidateRuleRecord, FilteredRuleRecord
from .extract import _parse_json_payload, _strip_reasoning_text
from .openai_client import create_chat_completion, extract_text_from_chat_response
from .policy import PolicyDoc, PolicyRule, PolicySection, SourcePointer
from .spec import DomainSpec, ModelConfig


DEFAULT_SYNTHESIS_MAX_TOKENS = 4096


def filter_candidate_rules(
    candidates: list[CandidateRuleRecord],
    *,
    starting_index: int = 1,
) -> list[FilteredRuleRecord]:
    grouped: OrderedDict[str, list[CandidateRuleRecord]] = OrderedDict()
    for candidate in candidates:
        key = _normalize_rule_text(candidate.text)
        if not key:
            continue
        grouped.setdefault(key, []).append(candidate)

    filtered: list[FilteredRuleRecord] = []
    next_index = starting_index
    for group in grouped.values():
        canonical = group[0]
        keyphrases = _union_in_order(candidate.keyphrases for candidate in group)
        sources = _merge_sources(candidate.sources for candidate in group)
        source_candidate_ids = [candidate.candidate_id for candidate in group]
        relevance_score = _support_score(
            n_candidates=len(group),
            n_sources=len(sources),
            n_keyphrases=len(keyphrases),
        )
        filtered.append(
            FilteredRuleRecord(
                rule_id=f"rule_{next_index:05d}",
                text=canonical.text,
                relevance_score=relevance_score,
                keyphrases=keyphrases,
                sources=sources,
                source_candidate_ids=source_candidate_ids,
            )
        )
        next_index += 1

    filtered.sort(key=lambda item: (-item.relevance_score, item.text.lower()))
    return filtered


def synthesize_policy_doc(
    *,
    client: Any,
    model_config: ModelConfig,
    domain: DomainSpec,
    filtered_rules: list[FilteredRuleRecord],
    version: str,
    current_policy: PolicyDoc | None = None,
) -> PolicyDoc:
    if not filtered_rules:
        if current_policy is None:
            raise ValueError("cannot synthesize a policy without any filtered rules")
        return PolicyDoc(version=version, sections=current_policy.sections)

    rule_lookup = {rule.rule_id: rule for rule in filtered_rules}
    structured_sections = _attempt_model_synthesis(
        client=client,
        model_config=model_config,
        domain=domain,
        filtered_rules=filtered_rules,
    )
    if structured_sections is None:
        return PolicyDoc(
            version=version,
            sections=[
                PolicySection(
                    title=domain.name,
                    summary=domain.description,
                    content_type="rules",
                    rules=[_policy_rule_from_filtered(rule) for rule in filtered_rules],
                )
            ],
        )

    sections: list[PolicySection] = []
    seen_rule_ids: set[str] = set()
    for item in structured_sections:
        rule_ids = [rule_id for rule_id in item["rule_ids"] if rule_id in rule_lookup and rule_id not in seen_rule_ids]
        if not rule_ids:
            continue
        seen_rule_ids.update(rule_ids)
        sections.append(
            PolicySection(
                title=item["title"],
                summary=item["summary"],
                content_type="rules",
                rules=[_policy_rule_from_filtered(rule_lookup[rule_id]) for rule_id in rule_ids],
            )
        )

    remaining_rule_ids = [rule.rule_id for rule in filtered_rules if rule.rule_id not in seen_rule_ids]
    if remaining_rule_ids:
        sections.append(
            PolicySection(
                title="Additional Coverage",
                content_type="rules",
                rules=[_policy_rule_from_filtered(rule_lookup[rule_id]) for rule_id in remaining_rule_ids],
            )
        )

    return PolicyDoc(version=version, sections=sections)


def _attempt_model_synthesis(
    *,
    client: Any,
    model_config: ModelConfig,
    domain: DomainSpec,
    filtered_rules: list[FilteredRuleRecord],
) -> list[dict[str, Any]] | None:
    prompt = f"""You are organizing policy rules into a readable policy document.

Domain name: {domain.name}
Domain description: {domain.description}

Rules:
{json.dumps([_filtered_rule_summary(rule) for rule in filtered_rules], indent=2)}

Return a JSON array where each item has:
- "title": a concise section title
- "summary": an optional short summary string or null
- "rule_ids": a non-empty list of rule ids

Cover every rule id exactly once. Use section titles that are specific to the domain. Do not invent new rules or modify rule text."""
    overrides: dict[str, Any] = {"temperature": 0}
    if (
        "max_tokens" not in model_config.request_defaults
        and "max_completion_tokens" not in model_config.request_defaults
    ):
        overrides["max_tokens"] = DEFAULT_SYNTHESIS_MAX_TOKENS
    try:
        response = create_chat_completion(
            client,
            model_config,
            [
                {"role": "system", "content": "Return only valid JSON that matches the requested schema."},
                {"role": "user", "content": prompt},
            ],
            **overrides,
        )
        parsed = _parse_json_payload(_strip_reasoning_text(extract_text_from_chat_response(response)).strip())
    except Exception:
        return None

    if not isinstance(parsed, list):
        return None
    sections: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        rule_ids = item.get("rule_ids", [])
        if not title or not isinstance(rule_ids, list):
            continue
        normalized_rule_ids = [str(rule_id).strip() for rule_id in rule_ids if str(rule_id).strip()]
        if not normalized_rule_ids:
            continue
        sections.append(
            {
                "title": title,
                "summary": _optional_str(item.get("summary")),
                "rule_ids": normalized_rule_ids,
            }
        )
    return sections or None


def _filtered_rule_summary(rule: FilteredRuleRecord) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "text": rule.text,
        "keyphrases": rule.keyphrases,
        "relevance_score": rule.relevance_score,
    }


def _policy_rule_from_filtered(rule: FilteredRuleRecord) -> PolicyRule:
    return PolicyRule(text=rule.text, keyphrases=rule.keyphrases, sources=rule.sources)


def _support_score(*, n_candidates: int, n_sources: int, n_keyphrases: int) -> float:
    score = 0.55
    score += min(0.2, 0.05 * max(0, n_candidates - 1))
    score += min(0.15, 0.05 * max(0, n_sources - 1))
    score += min(0.1, 0.025 * n_keyphrases)
    return round(min(score, 1.0), 4)


def _normalize_rule_text(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def _union_in_order(groups: list[list[str]] | Any) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for group in groups:
        for item in group:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(item)
    return values


def _merge_sources(groups: list[list[SourcePointer]] | Any) -> list[SourcePointer]:
    seen: set[tuple[str, str | None]] = set()
    merged: list[SourcePointer] = []
    for group in groups:
        for source in group:
            key = (source.url, source.supporting_excerpt)
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)
    return merged


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
