from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._serialization import drop_nones


@dataclass(slots=True)
class SourcePointer:
    url: str
    supporting_excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourcePointer":
        return cls(url=data["url"], supporting_excerpt=data.get("supporting_excerpt"))


@dataclass(slots=True)
class PolicyRule:
    text: str
    keyphrases: list[str] = field(default_factory=list)
    sources: list[SourcePointer] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyRule":
        return cls(
            text=data["text"],
            keyphrases=data.get("keyphrases", []),
            sources=[SourcePointer.from_dict(item) for item in data.get("sources", [])],
        )


@dataclass(slots=True)
class PolicySubsection:
    title: str
    rules: list[PolicyRule]
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicySubsection":
        return cls(
            title=data["title"],
            summary=data.get("summary"),
            rules=[PolicyRule.from_dict(item) for item in data.get("rules", [])],
        )


@dataclass(slots=True)
class PolicySection:
    title: str
    content_type: str
    rules: list[PolicyRule] = field(default_factory=list)
    subsections: list[PolicySubsection] = field(default_factory=list)
    summary: str | None = None

    def validate(self) -> None:
        if self.content_type not in {"rules", "subsections"}:
            raise ValueError(f"unsupported section content_type: {self.content_type}")
        if self.content_type == "rules" and self.subsections:
            raise ValueError("rules sections cannot contain subsections")
        if self.content_type == "subsections" and self.rules:
            raise ValueError("subsection sections cannot contain direct rules")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicySection":
        value = cls(
            title=data["title"],
            summary=data.get("summary"),
            content_type=data["content_type"],
            rules=[PolicyRule.from_dict(item) for item in data.get("rules", [])],
            subsections=[PolicySubsection.from_dict(item) for item in data.get("subsections", [])],
        )
        value.validate()
        return value


@dataclass(slots=True)
class PolicyDoc:
    version: str
    sections: list[PolicySection]

    def to_dict(self) -> dict[str, Any]:
        return drop_nones(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyDoc":
        return cls(
            version=data["version"],
            sections=[PolicySection.from_dict(item) for item in data.get("sections", [])],
        )

    @classmethod
    def sample(cls) -> "PolicyDoc":
        return cls(
            version="run_20260309__policy__001",
            sections=[
                PolicySection(
                    title="Direct Threats",
                    content_type="rules",
                    rules=[
                        PolicyRule(
                            text="Reject direct, credible threats of violence.",
                            keyphrases=["direct threats", "credible violence"],
                            sources=[
                                SourcePointer(
                                    url="https://example.com/policy",
                                    supporting_excerpt="Direct threats of violence should be rejected.",
                                )
                            ],
                        )
                    ],
                ),
                PolicySection(
                    title="Harassment",
                    content_type="subsections",
                    subsections=[
                        PolicySubsection(
                            title="Targeted Abuse",
                            rules=[
                                PolicyRule(
                                    text="Reject targeted abusive language aimed at a protected person.",
                                    keyphrases=["targeted abuse"],
                                    sources=[SourcePointer(url="https://example.com/harassment")],
                                )
                            ],
                        )
                    ],
                ),
            ],
        )
