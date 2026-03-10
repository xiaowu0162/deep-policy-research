from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .artifacts import CandidateRuleRecord, ChunkRecord, FilteredRuleRecord, ResearchArtifact, SourceRecord
from .config import ResolvedTaskSpec, load_task_spec
from .eval import _load_or_write_policy, _optional_path_from_run_config, _probe_models, _read_validation_split_seed
from .extract import extract_candidate_rules, generate_search_queries
from .fetch import chunk_sources, create_url_fetcher, merge_sources
from .openai_client import create_sync_client, probe_model
from .policy import PolicyDoc
from .rank import filter_candidate_rules, synthesize_policy_doc
from .run_manager import RunManager
from .search import create_search_provider
from .spec import TaskSpec


@dataclass(slots=True)
class ResearchResult:
    completed_iterations: int
    query_count: int
    source_count: int
    chunk_count: int
    candidate_rule_count: int
    filtered_rule_count: int
    policy_version: str
    sources_version: str | None
    chunks_version: str | None
    candidate_rules_version: str | None
    filtered_rules_version: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "completed_iterations": self.completed_iterations,
            "query_count": self.query_count,
            "source_count": self.source_count,
            "chunk_count": self.chunk_count,
            "candidate_rule_count": self.candidate_rule_count,
            "filtered_rule_count": self.filtered_rule_count,
            "policy_version": self.policy_version,
            "sources_version": self.sources_version,
            "chunks_version": self.chunks_version,
            "candidate_rules_version": self.candidate_rules_version,
            "filtered_rules_version": self.filtered_rules_version,
        }

    @classmethod
    def from_artifact(cls, artifact: ResearchArtifact) -> "ResearchResult":
        return cls(
            completed_iterations=artifact.completed_iterations,
            query_count=artifact.query_count,
            source_count=artifact.source_count,
            chunk_count=artifact.chunk_count,
            candidate_rule_count=artifact.candidate_rule_count,
            filtered_rule_count=artifact.filtered_rule_count,
            policy_version=artifact.policy_version,
            sources_version=artifact.sources_version,
            chunks_version=artifact.chunks_version,
            candidate_rules_version=artifact.candidate_rules_version,
            filtered_rules_version=artifact.filtered_rules_version,
        )


def run_research_command(
    spec_path: str | Path,
    *,
    output_dir: str | Path,
    probe_all_models: bool = False,
) -> tuple[RunManager, ResearchResult, PolicyDoc]:
    resolved = load_task_spec(spec_path)
    run_config = resolved.to_run_config_dict()
    run_config["entrypoint"] = "research"
    manager = RunManager.create(
        output_dir=output_dir,
        spec=resolved.spec,
        run_config=run_config,
        current_step="research",
    )
    return _run_or_resume_research(manager, resolved, probe_all_models=probe_all_models)


def resume_research_command(run_dir: str | Path) -> tuple[RunManager, ResearchResult, PolicyDoc]:
    manager = RunManager.load(run_dir)
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
    if manager.manifest.status == "completed":
        policy = manager.load_artifact("policy", PolicyDoc)
        return (manager, _load_current_research_result(manager, policy, spec), policy)
    return _run_or_resume_research(manager, resolved, probe_all_models=False)


def _run_or_resume_research(
    manager: RunManager,
    resolved: ResolvedTaskSpec,
    *,
    probe_all_models: bool,
) -> tuple[RunManager, ResearchResult, PolicyDoc]:
    try:
        manager.update_status(
            "running",
            current_step="research",
            current_round=manager.manifest.current_round,
            current_iteration=manager.manifest.current_iteration,
        )
        if probe_all_models:
            _probe_models(resolved.spec, probe_all_models=True)
        else:
            research_probe_client = create_sync_client(resolved.spec.research.model)
            probe_model(research_probe_client, resolved.spec.research.model)

        research_client = create_sync_client(resolved.spec.research.model)
        search_provider = create_search_provider(
            resolved.resolved_research_search(),
            cache_dir=manager.root_dir / "cache" / "search",
        )
        fetcher = create_url_fetcher(cache_dir=manager.root_dir / "cache" / "fetch")
        current_policy = _load_existing_policy(manager, resolved)

        total_iterations = resolved.spec.research.max_rounds * resolved.spec.research.iterations_per_round
        query_count = 0
        for iteration_index in range(manager.manifest.current_iteration, total_iterations):
            round_index = (iteration_index // resolved.spec.research.iterations_per_round) + 1
            manager.update_status(
                "running",
                current_step="research",
                current_round=round_index,
                current_iteration=iteration_index,
            )
            existing_sources = _load_rows_if_present(manager, "sources", SourceRecord)
            existing_chunks = _load_rows_if_present(manager, "chunks", ChunkRecord)
            existing_candidates = _load_rows_if_present(manager, "candidate_rules", CandidateRuleRecord)

            queries = generate_search_queries(
                client=research_client,
                model_config=resolved.spec.research.model,
                domain=resolved.spec.domain,
                current_policy=current_policy,
                query_count=resolved.spec.research.queries_per_iteration,
            )
            query_count += len(queries)

            existing_urls = {source.url for source in existing_sources}
            query_by_url: dict[str, list[str]] = defaultdict(list)
            urls_to_fetch: list[str] = []
            seen_urls_to_fetch: set[str] = set()
            for query in queries:
                for result in search_provider.search(query, num_results=resolved.spec.research.pages_per_query):
                    query_by_url[result.url].append(query)
                    if result.url in existing_urls or result.url in seen_urls_to_fetch:
                        continue
                    urls_to_fetch.append(result.url)
                    seen_urls_to_fetch.add(result.url)

            fetched_documents = [document for url in urls_to_fetch if (document := fetcher.fetch(url)) is not None]
            sources = merge_sources(existing_sources, fetched_documents, query_by_url=query_by_url)
            if not sources and current_policy is None:
                raise ValueError("research could not retrieve any usable sources and no initial policy is available")
            sources_version = manager.new_version("sources")
            manager.write_rows_artifact("sources", sources_version, sources)

            new_sources = [source for source in sources if source.url in {document.url for document in fetched_documents}]
            new_chunks = chunk_sources(new_sources, starting_index=len(existing_chunks) + 1)
            chunks = [*existing_chunks, *new_chunks]
            chunks_version = manager.new_version("chunks")
            manager.write_rows_artifact("chunks", chunks_version, chunks)

            source_lookup = {source.source_id: source for source in sources}
            new_candidates = extract_candidate_rules(
                client=research_client,
                model_config=resolved.spec.research.model,
                domain=resolved.spec.domain,
                chunks=new_chunks,
                source_lookup=source_lookup,
                starting_index=len(existing_candidates) + 1,
            )
            candidates = [*existing_candidates, *new_candidates]
            candidate_rules_version = manager.new_version("candidate_rules")
            manager.write_rows_artifact("candidate_rules", candidate_rules_version, candidates)

            filtered_rules = filter_candidate_rules(candidates, starting_index=1)
            filtered_rules_version = manager.new_version("filtered_rules")
            manager.write_rows_artifact("filtered_rules", filtered_rules_version, filtered_rules)

            policy = synthesize_policy_doc(
                client=research_client,
                model_config=resolved.spec.research.model,
                domain=resolved.spec.domain,
                filtered_rules=filtered_rules,
                version=manager.new_version("policy"),
                current_policy=current_policy,
            )
            manager.write_artifact("policy", policy)
            current_policy = policy
            manager.update_status(
                "running",
                current_step="research",
                current_round=round_index,
                current_iteration=iteration_index + 1,
            )

        result = _summarize_current_research_state(manager, current_policy, query_count=query_count)
        manager.record_artifact("research", _artifact_from_result(manager, result), accept=True)
        manager.update_status(
            "completed",
            current_step="research",
            current_round=resolved.spec.research.max_rounds,
            current_iteration=total_iterations,
        )
        return manager, result, current_policy
    except Exception:
        manager.update_status("failed", current_step="research")
        raise


def _load_existing_policy(manager: RunManager, resolved: ResolvedTaskSpec) -> PolicyDoc | None:
    if manager.manifest.current_policy_version is not None:
        return manager.load_artifact("policy", PolicyDoc)
    if resolved.spec.inputs.initial_policy.text or resolved.initial_policy_doc_path is not None:
        return _load_or_write_policy(manager, resolved)
    return None


def _load_rows_if_present(manager: RunManager, kind: str, row_cls):
    descriptor_attr = f"current_{kind}_version"
    if getattr(manager.manifest, descriptor_attr) is None:
        return []
    return manager.load_rows_artifact(kind, row_cls)


def _summarize_current_research_state(
    manager: RunManager,
    policy: PolicyDoc,
    *,
    query_count: int = 0,
) -> ResearchResult:
    sources = _load_rows_if_present(manager, "sources", SourceRecord)
    chunks = _load_rows_if_present(manager, "chunks", ChunkRecord)
    candidate_rules = _load_rows_if_present(manager, "candidate_rules", CandidateRuleRecord)
    filtered_rules = _load_rows_if_present(manager, "filtered_rules", FilteredRuleRecord)
    return ResearchResult(
        completed_iterations=manager.manifest.current_iteration,
        query_count=query_count,
        source_count=len(sources),
        chunk_count=len(chunks),
        candidate_rule_count=len(candidate_rules),
        filtered_rule_count=len(filtered_rules),
        policy_version=policy.version,
        sources_version=manager.manifest.current_sources_version,
        chunks_version=manager.manifest.current_chunks_version,
        candidate_rules_version=manager.manifest.current_candidate_rules_version,
        filtered_rules_version=manager.manifest.current_filtered_rules_version,
    )


def _artifact_from_result(manager: RunManager, result: ResearchResult) -> ResearchArtifact:
    return ResearchArtifact(
        version=manager.new_version("research"),
        completed_iterations=result.completed_iterations,
        query_count=result.query_count,
        source_count=result.source_count,
        chunk_count=result.chunk_count,
        candidate_rule_count=result.candidate_rule_count,
        filtered_rule_count=result.filtered_rule_count,
        policy_version=result.policy_version,
        sources_version=result.sources_version,
        chunks_version=result.chunks_version,
        candidate_rules_version=result.candidate_rules_version,
        filtered_rules_version=result.filtered_rules_version,
    )


def _load_current_research_result(manager: RunManager, policy: PolicyDoc, spec: TaskSpec) -> ResearchResult:
    if manager.manifest.current_research_version is not None:
        artifact = manager.load_artifact("research", ResearchArtifact)
        return ResearchResult.from_artifact(artifact)
    return _summarize_current_research_state(
        manager,
        policy,
        query_count=spec.research.queries_per_iteration * manager.manifest.current_iteration,
    )
