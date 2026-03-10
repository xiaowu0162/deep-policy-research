from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .artifacts import CandidateRuleRecord, ChunkRecord, SourceRecord
from .openai_client import create_chat_completion, extract_text_from_chat_response
from .policy import PolicyDoc, SourcePointer
from .spec import DomainSpec, ModelConfig


DEFAULT_QUERY_MAX_TOKENS = 2048
DEFAULT_RULE_EXTRACTION_MAX_TOKENS = 4096


def generate_search_queries(
    *,
    client: Any,
    model_config: ModelConfig,
    domain: DomainSpec,
    current_policy: PolicyDoc | None,
    query_count: int,
) -> list[str]:
    policy_summary = _summarize_policy(current_policy)
    prompt = f"""You are an expert researcher building a high-quality policy for a domain.

Generate exactly {query_count} web search queries that will improve policy coverage.

Domain name: {domain.name}
Domain description: {domain.description}

Current policy summary:
{policy_summary}

Return a JSON array of strings and nothing else. The queries should be specific, evidence-seeking, and biased toward authoritative source types such as policy guidance, legal definitions, clinical guidance, platform rules, or safety documentation when relevant."""
    raw_output = _run_json_prompt(
        client,
        model_config,
        system_prompt="Return only valid JSON that matches the requested schema.",
        user_prompt=prompt,
        max_tokens=DEFAULT_QUERY_MAX_TOKENS,
    )
    parsed = _parse_json_payload(raw_output)
    queries = _coerce_query_list(parsed)
    if len(queries) < query_count:
        queries.extend(_fallback_queries(domain, existing=queries, count=query_count - len(queries)))
    deduped = list(dict.fromkeys(query.strip() for query in queries if query and query.strip()))
    return deduped[:query_count]


def extract_candidate_rules(
    *,
    client: Any,
    model_config: ModelConfig,
    domain: DomainSpec,
    chunks: Sequence[ChunkRecord],
    source_lookup: dict[str, SourceRecord],
    starting_index: int = 1,
) -> list[CandidateRuleRecord]:
    candidates: list[CandidateRuleRecord] = []
    next_index = starting_index
    for chunk in chunks:
        source = source_lookup[chunk.source_id]
        prompt = f"""You are extracting grounded policy statements from source material.

Domain name: {domain.name}
Domain description: {domain.description}

Source URL: {source.url}
Chunk text:
{chunk.text}

Return a JSON array. Each item must have:
- "text": a concise policy rule directly supported by the chunk
- "keyphrases": a short list of distinct keyphrases
- "supporting_excerpt": a brief exact quote copied from the chunk

Do not include unsupported rules. Return an empty JSON array if the chunk does not contain useful policy evidence."""
        raw_output = _run_json_prompt(
            client,
            model_config,
            system_prompt="Return only valid JSON that matches the requested schema.",
            user_prompt=prompt,
            max_tokens=DEFAULT_RULE_EXTRACTION_MAX_TOKENS,
        )
        parsed = _parse_json_payload(raw_output)
        for item in _coerce_candidate_list(parsed):
            excerpt = _normalize_excerpt(item["supporting_excerpt"], chunk.text)
            candidates.append(
                CandidateRuleRecord(
                    candidate_id=f"cand_{next_index:05d}",
                    text=item["text"],
                    keyphrases=item["keyphrases"],
                    sources=[SourcePointer(url=source.url, supporting_excerpt=excerpt)],
                    source_chunk_ids=[chunk.chunk_id],
                )
            )
            next_index += 1
    return candidates


def _run_json_prompt(
    client: Any,
    model_config: ModelConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **overrides,
    )
    return _strip_reasoning_text(extract_text_from_chat_response(response)).strip()


def _parse_json_payload(raw_output: str) -> Any:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[\s*.*\s*\]", cleaned, re.DOTALL)
    if array_match:
        return json.loads(array_match.group(0))
    object_match = re.search(r"\{\s*.*\s*\}", cleaned, re.DOTALL)
    if object_match:
        return json.loads(object_match.group(0))
    raise ValueError("model output did not contain valid JSON")


def _coerce_query_list(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("query generation output must be a JSON array of strings")


def _coerce_candidate_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ["rules", "data", "results", "items", "candidates"]:
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        return []

    candidates: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("rule") or "").strip()
        if not text:
            continue
        raw_keyphrases = item.get("keyphrases", [])
        if isinstance(raw_keyphrases, str):
            keyphrases = [raw_keyphrases.strip()] if raw_keyphrases.strip() else []
        elif isinstance(raw_keyphrases, list):
            keyphrases = [str(value).strip() for value in raw_keyphrases if str(value).strip()]
        else:
            keyphrases = []
        supporting_excerpt = str(
            item.get("supporting_excerpt") or item.get("supporting_text") or ""
        ).strip()
        candidates.append(
            {
                "text": text,
                "keyphrases": list(dict.fromkeys(keyphrases)),
                "supporting_excerpt": supporting_excerpt,
            }
        )
    return candidates


def _normalize_excerpt(excerpt: str, chunk_text: str) -> str | None:
    if not excerpt:
        fallback = chunk_text[:280].strip()
        return fallback or None
    if excerpt in chunk_text:
        return excerpt
    compact_excerpt = " ".join(excerpt.split())
    compact_chunk = " ".join(chunk_text.split())
    if compact_excerpt and compact_excerpt in compact_chunk:
        return excerpt
    fallback = excerpt[:280].strip()
    return fallback or None


def _fallback_queries(domain: DomainSpec, *, existing: list[str], count: int) -> list[str]:
    templates = [
        f"{domain.name} official policy guidance",
        f"{domain.name} examples and edge cases",
        f"{domain.name} legal definitions",
        f"{domain.name} safety guidance",
        f"{domain.name} platform policy examples",
    ]
    result = []
    seen = set(existing)
    for query in templates:
        if query not in seen:
            result.append(query)
            seen.add(query)
        if len(result) >= count:
            break
    return result


def _summarize_policy(policy: PolicyDoc | None) -> str:
    if policy is None or not policy.sections:
        return "No accepted policy yet."
    lines: list[str] = []
    for section in policy.sections:
        lines.append(f"- {section.title}")
        for rule in section.rules:
            lines.append(f"  - {rule.text}")
        for subsection in section.subsections:
            lines.append(f"  - {subsection.title}")
            for rule in subsection.rules:
                lines.append(f"    - {rule.text}")
    return "\n".join(lines[:40])


def _strip_reasoning_text(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()
