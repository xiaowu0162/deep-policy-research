from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._serialization import drop_nones
from .policy import SourcePointer


@dataclass(slots=True)
class ExampleRecord:
    id: str
    input: str
    label: str
    split: str
    source_type: str
    provenance: list[SourcePointer] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    policy_version: str | None = None

    def validate(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.source_type not in {"seed", "research", "redteam", "perturbation"}:
            raise ValueError(f"unsupported source_type: {self.source_type}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExampleRecord":
        value = cls(
            id=data["id"],
            input=data["input"],
            label=data["label"],
            split=data["split"],
            source_type=data["source_type"],
            provenance=[SourcePointer.from_dict(item) for item in data.get("provenance", [])],
            keyphrases=data.get("keyphrases", []),
            policy_version=data.get("policy_version"),
        )
        value.validate()
        return value


@dataclass(slots=True)
class ExamplePool:
    version: str
    examples: list[ExampleRecord]

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExamplePool":
        return cls(
            version=data["version"],
            examples=[ExampleRecord.from_dict(item) for item in data.get("examples", [])],
        )

    @classmethod
    def sample(cls) -> "ExamplePool":
        return cls(
            version="run_20260309__example_pool__001",
            examples=[
                ExampleRecord(
                    id="ex_001",
                    input="I will find you and hurt you.",
                    label="REJECTED",
                    split="train",
                    source_type="seed",
                    keyphrases=["threat"],
                )
            ],
        )
