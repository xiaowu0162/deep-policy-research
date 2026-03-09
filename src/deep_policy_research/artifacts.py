from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._serialization import drop_nones
from .policy import SourcePointer


@dataclass(slots=True)
class ArtifactVersion:
    version: str
    path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactVersion":
        return cls(version=data["version"], path=data["path"], created_at=data["created_at"])


@dataclass(slots=True)
class RunManifest:
    run_id: str
    task_id: str
    created_at: str
    updated_at: str
    status: str
    current_round: int
    current_iteration: int
    current_step: str
    spec_path: str
    run_config_path: str
    current_policy_version: str | None = None
    current_example_pool_version: str | None = None
    current_prompt_version: str | None = None
    current_metrics_version: str | None = None
    policy_versions: list[ArtifactVersion] = field(default_factory=list)
    example_pool_versions: list[ArtifactVersion] = field(default_factory=list)
    prompt_versions: list[ArtifactVersion] = field(default_factory=list)
    metrics_versions: list[ArtifactVersion] = field(default_factory=list)

    def validate(self) -> None:
        if self.status not in {"running", "completed", "failed", "interrupted"}:
            raise ValueError(f"unsupported run status: {self.status}")
        if self.current_step not in {"research", "redteam", "optimize", "eval"}:
            raise ValueError(f"unsupported current_step: {self.current_step}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        value = cls(
            run_id=data["run_id"],
            task_id=data["task_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data["status"],
            current_round=data["current_round"],
            current_iteration=data["current_iteration"],
            current_step=data["current_step"],
            spec_path=data["spec_path"],
            run_config_path=data["run_config_path"],
            current_policy_version=data.get("current_policy_version"),
            current_example_pool_version=data.get("current_example_pool_version"),
            current_prompt_version=data.get("current_prompt_version"),
            current_metrics_version=data.get("current_metrics_version"),
            policy_versions=[ArtifactVersion.from_dict(item) for item in data.get("policy_versions", [])],
            example_pool_versions=[ArtifactVersion.from_dict(item) for item in data.get("example_pool_versions", [])],
            prompt_versions=[ArtifactVersion.from_dict(item) for item in data.get("prompt_versions", [])],
            metrics_versions=[ArtifactVersion.from_dict(item) for item in data.get("metrics_versions", [])],
        )
        value.validate()
        return value

    @classmethod
    def sample(cls) -> "RunManifest":
        return cls(
            run_id="run_20260309_120000",
            task_id="toy_moderation_task",
            created_at="2026-03-09T12:00:00Z",
            updated_at="2026-03-09T12:00:00Z",
            status="running",
            current_round=1,
            current_iteration=0,
            current_step="research",
            spec_path="spec.json",
            run_config_path="run_config.json",
            policy_versions=[
                ArtifactVersion(
                    version="run_20260309_120000__policy__001",
                    path="policy_run_20260309_120000__policy__001.json",
                    created_at="2026-03-09T12:00:00Z",
                )
            ],
        )


@dataclass(slots=True)
class ReaderPromptMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReaderPromptMessage":
        return cls(role=data["role"], content=data["content"])


@dataclass(slots=True)
class ReaderPrompts:
    version: str
    messages: list[ReaderPromptMessage]

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReaderPrompts":
        return cls(
            version=data["version"],
            messages=[ReaderPromptMessage.from_dict(item) for item in data.get("messages", [])],
        )

    @classmethod
    def sample(cls) -> "ReaderPrompts":
        return cls(
            version="run_20260309_120000__reader_prompts__001",
            messages=[
                ReaderPromptMessage(role="system", content="Apply the policy faithfully."),
                ReaderPromptMessage(role="user", content="Policy:\n{{policy}}\n\nLabels: {{labels}}\n\nInput:\n{{input}}"),
            ],
        )


@dataclass(slots=True)
class SplitMetrics:
    n_examples: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitMetrics":
        return cls(n_examples=data["n_examples"], metrics=data.get("metrics", {}))


@dataclass(slots=True)
class MetricsArtifact:
    version: str
    policy_version: str
    example_pool_version: str
    prompt_version: str
    splits: dict[str, SplitMetrics]

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricsArtifact":
        return cls(
            version=data["version"],
            policy_version=data["policy_version"],
            example_pool_version=data["example_pool_version"],
            prompt_version=data["prompt_version"],
            splits={name: SplitMetrics.from_dict(item) for name, item in data.get("splits", {}).items()},
        )

    @classmethod
    def sample(cls) -> "MetricsArtifact":
        return cls(
            version="run_20260309_120000__metrics__001",
            policy_version="run_20260309_120000__policy__001",
            example_pool_version="run_20260309_120000__example_pool__001",
            prompt_version="run_20260309_120000__reader_prompts__001",
            splits={"validation": SplitMetrics(n_examples=12, metrics={"f1": 0.75})},
        )


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    url: str
    retrieved_at: str
    queries: list[str]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRecord":
        return cls(
            source_id=data["source_id"],
            url=data["url"],
            retrieved_at=data["retrieved_at"],
            queries=data.get("queries", []),
            text=data["text"],
        )

    @classmethod
    def sample(cls) -> "SourceRecord":
        return cls(
            source_id="src_001",
            url="https://example.com/policy",
            retrieved_at="2026-03-09T12:00:00Z",
            queries=["online harassment examples"],
            text="Long source text goes here.",
        )


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=data["chunk_id"],
            source_id=data["source_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
        )

    @classmethod
    def sample(cls) -> "ChunkRecord":
        return cls(chunk_id="chunk_001", source_id="src_001", chunk_index=0, text="Chunk text.")


@dataclass(slots=True)
class CandidateRuleRecord:
    candidate_id: str
    text: str
    keyphrases: list[str] = field(default_factory=list)
    sources: list[SourcePointer] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRuleRecord":
        return cls(
            candidate_id=data["candidate_id"],
            text=data["text"],
            keyphrases=data.get("keyphrases", []),
            sources=[SourcePointer.from_dict(item) for item in data.get("sources", [])],
            source_chunk_ids=data.get("source_chunk_ids", []),
        )

    @classmethod
    def sample(cls) -> "CandidateRuleRecord":
        return cls(
            candidate_id="cand_001",
            text="Reject targeted threats of physical harm.",
            keyphrases=["targeted threats"],
            sources=[SourcePointer(url="https://example.com/policy")],
            source_chunk_ids=["chunk_001"],
        )


@dataclass(slots=True)
class FilteredRuleRecord:
    rule_id: str
    text: str
    relevance_score: float
    keyphrases: list[str] = field(default_factory=list)
    sources: list[SourcePointer] = field(default_factory=list)
    source_candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilteredRuleRecord":
        return cls(
            rule_id=data["rule_id"],
            text=data["text"],
            relevance_score=data["relevance_score"],
            keyphrases=data.get("keyphrases", []),
            sources=[SourcePointer.from_dict(item) for item in data.get("sources", [])],
            source_candidate_ids=data.get("source_candidate_ids", []),
        )

    @classmethod
    def sample(cls) -> "FilteredRuleRecord":
        return cls(
            rule_id="rule_001",
            text="Reject targeted threats of physical harm.",
            relevance_score=0.95,
            keyphrases=["targeted threats"],
            sources=[SourcePointer(url="https://example.com/policy")],
            source_candidate_ids=["cand_001"],
        )
