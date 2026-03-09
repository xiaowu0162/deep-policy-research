from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._serialization import drop_nones


@dataclass(slots=True)
class DomainSpec:
    name: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainSpec":
        return cls(name=data["name"], description=data["description"])


@dataclass(slots=True)
class ModelConfig:
    model: str
    base_url: str
    api_key_env: str
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    request_defaults: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            model=data["model"],
            base_url=data["base_url"],
            api_key_env=data["api_key_env"],
            client_kwargs=data.get("client_kwargs", {}),
            request_defaults=data.get("request_defaults", {}),
        )


@dataclass(slots=True)
class SearchConfig:
    provider: str = "serper"
    api_key_env: str = "SERPER_API_KEY"
    num_results: int = 10
    country: str = "us"
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        return cls(
            provider=data.get("provider", "serper"),
            api_key_env=data.get("api_key_env", "SERPER_API_KEY"),
            num_results=data.get("num_results", 10),
            country=data.get("country", "us"),
            language=data.get("language", "en"),
        )


@dataclass(slots=True)
class InitialPolicyInput:
    text: str | None = None
    policy_doc_path: str | None = None

    def validate(self) -> None:
        if self.text and self.policy_doc_path:
            raise ValueError("initial_policy.text and initial_policy.policy_doc_path are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InitialPolicyInput":
        value = cls(text=data.get("text"), policy_doc_path=data.get("policy_doc_path"))
        value.validate()
        return value


@dataclass(slots=True)
class DataConfig:
    train_path: str | None = None
    validation_path: str | None = None
    test_path: str | None = None
    format: str = "jsonl"
    bootstrap_train_ratio: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        return cls(
            train_path=data.get("train_path"),
            validation_path=data.get("validation_path"),
            test_path=data.get("test_path"),
            format=data.get("format", "jsonl"),
            bootstrap_train_ratio=data.get("bootstrap_train_ratio", 0.8),
        )


@dataclass(slots=True)
class TaskInputs:
    initial_policy: InitialPolicyInput = field(default_factory=InitialPolicyInput)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInputs":
        return cls(
            initial_policy=InitialPolicyInput.from_dict(data.get("initial_policy", {})),
            data=DataConfig.from_dict(data.get("data", {})),
        )


@dataclass(slots=True)
class ResearchStageConfig:
    model: ModelConfig
    search: SearchConfig = field(default_factory=SearchConfig)
    max_rounds: int = 3
    iterations_per_round: int = 1
    queries_per_iteration: int = 3
    pages_per_query: int = 10

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchStageConfig":
        return cls(
            model=ModelConfig.from_dict(data["model"]),
            search=SearchConfig.from_dict(data.get("search", {})),
            max_rounds=data.get("max_rounds", 3),
            iterations_per_round=data.get("iterations_per_round", 1),
            queries_per_iteration=data.get("queries_per_iteration", 3),
            pages_per_query=data.get("pages_per_query", 10),
        )


@dataclass(slots=True)
class RedteamStageConfig:
    model: ModelConfig
    search: SearchConfig = field(default_factory=SearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedteamStageConfig":
        return cls(
            model=ModelConfig.from_dict(data["model"]),
            search=SearchConfig.from_dict(data.get("search", {})),
        )


@dataclass(slots=True)
class OptimizeStageConfig:
    reader_model: ModelConfig
    acceptance_split: str = "validation"
    acceptance_metric: str = "f1"

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeStageConfig":
        return cls(
            reader_model=ModelConfig.from_dict(data["reader_model"]),
            acceptance_split=data.get("acceptance_split", "validation"),
            acceptance_metric=data.get("acceptance_metric", "f1"),
        )


@dataclass(slots=True)
class MetricSpec:
    name: str

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricSpec":
        return cls(name=data["name"])


@dataclass(slots=True)
class EvaluateStageConfig:
    task_type: str = "classification"
    label_space: list[str] = field(default_factory=list)
    metrics: list[MetricSpec] = field(default_factory=lambda: [MetricSpec(name="f1")])

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluateStageConfig":
        return cls(
            task_type=data.get("task_type", "classification"),
            label_space=data.get("label_space", []),
            metrics=[MetricSpec.from_dict(item) for item in data.get("metrics", [{"name": "f1"}])],
        )


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    domain: DomainSpec
    research: ResearchStageConfig
    redteam: RedteamStageConfig
    optimize: OptimizeStageConfig
    evaluate: EvaluateStageConfig
    version: str = "0.1"
    inputs: TaskInputs = field(default_factory=TaskInputs)

    def validate(self) -> None:
        self.inputs.initial_policy.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        value = cls(
            version=data.get("version", "0.1"),
            task_id=data["task_id"],
            domain=DomainSpec.from_dict(data["domain"]),
            inputs=TaskInputs.from_dict(data.get("inputs", {})),
            research=ResearchStageConfig.from_dict(data["research"]),
            redteam=RedteamStageConfig.from_dict(data["redteam"]),
            optimize=OptimizeStageConfig.from_dict(data["optimize"]),
            evaluate=EvaluateStageConfig.from_dict(data["evaluate"]),
        )
        value.validate()
        return value

    @classmethod
    def sample(cls) -> "TaskSpec":
        model = ModelConfig(
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
        return cls(
            task_id="toy_moderation_task",
            domain=DomainSpec(
                name="Toy Moderation",
                description="Decide whether a message should be approved or rejected.",
            ),
            inputs=TaskInputs(
                initial_policy=InitialPolicyInput(text="Reject explicit threats and approve harmless chit-chat."),
                data=DataConfig(train_path="data/train.jsonl", validation_path="data/validation.jsonl"),
            ),
            research=ResearchStageConfig(model=model),
            redteam=RedteamStageConfig(model=model),
            optimize=OptimizeStageConfig(reader_model=model),
            evaluate=EvaluateStageConfig(label_space=["APPROVED", "REJECTED"]),
        )
