from __future__ import annotations

import re
from collections.abc import Callable

from .artifacts import ReaderPromptMessage, ReaderPrompts
from .policy import PolicyDoc


SUPPORTED_PLACEHOLDERS = {"policy", "input", "labels"}
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


def default_reader_prompts(*, version: str) -> ReaderPrompts:
    return ReaderPrompts(
        version=version,
        messages=[
            ReaderPromptMessage(
                role="system",
                content="Follow the policy and reply with exactly one allowed label.",
            ),
            ReaderPromptMessage(
                role="user",
                content=(
                    "Policy:\n{{policy}}\n\n"
                    "Allowed labels:\n{{labels}}\n\n"
                    "Input:\n{{input}}\n\n"
                    "Reply with exactly one label from the allowed labels."
                ),
            ),
        ],
    )


def generate_prompt_variants(base_prompts: ReaderPrompts, *, version_factory: Callable[[], str]) -> list[ReaderPrompts]:
    variants: list[list[ReaderPromptMessage]] = []
    variants.append(
        [
            ReaderPromptMessage(
                role="system",
                content="Follow the policy carefully. You may think through the decision, but the final line must be exactly one allowed label.",
            ),
            ReaderPromptMessage(
                role="user",
                content=(
                    "Policy:\n{{policy}}\n\n"
                    "Allowed labels:\n{{labels}}\n\n"
                    "Input:\n{{input}}\n\n"
                    "Decide which label applies. Keep any explanation brief and end with exactly one allowed label on the final line."
                ),
            ),
        ]
    )
    variants.append(
        [
            ReaderPromptMessage(
                role="system",
                content="Classify the input by following the policy exactly.",
            ),
            ReaderPromptMessage(
                role="user",
                content=(
                    "Policy:\n{{policy}}\n\n"
                    "Allowed labels:\n{{labels}}\n\n"
                    "Input:\n{{input}}\n\n"
                    "Output exactly one label from the allowed labels and nothing else."
                ),
            ),
        ]
    )
    variants.append(
        [
            ReaderPromptMessage(
                role="system",
                content="Apply the policy faithfully and avoid unsupported guesses.",
            ),
            ReaderPromptMessage(
                role="user",
                content=(
                    "Allowed labels:\n{{labels}}\n\n"
                    "Policy:\n{{policy}}\n\n"
                    "Input:\n{{input}}\n\n"
                    "Answer in two lines:\n"
                    "Reason: <very short explanation>\n"
                    "Label: <one allowed label>"
                ),
            ),
        ]
    )
    variants.append(
        [
            ReaderPromptMessage(
                role="system",
                content="Use the policy as the sole decision standard.",
            ),
            ReaderPromptMessage(
                role="user",
                content=(
                    "Allowed labels:\n{{labels}}\n\n"
                    "Policy:\n{{policy}}\n\n"
                    "Task:\nChoose the single best label for the input.\n\n"
                    "Input:\n{{input}}\n\n"
                    "Identify the policy-relevant evidence briefly, then finish with a line containing only the chosen label."
                ),
            ),
        ]
    )

    if base_prompts.messages:
        derived_messages = [ReaderPromptMessage(role=message.role, content=message.content) for message in base_prompts.messages]
        derived_messages[-1] = ReaderPromptMessage(
            role=derived_messages[-1].role,
            content=(
                derived_messages[-1].content
                + "\n\nIf you provide any explanation, the final non-empty line must be exactly one allowed label."
            ),
        )
        variants.append(derived_messages)

    deduped: list[ReaderPrompts] = []
    seen = {_prompt_signature(base_prompts)}
    for messages in variants:
        prompts = ReaderPrompts(version=version_factory(), messages=list(messages))
        validate_reader_prompts(prompts)
        signature = _prompt_signature(prompts)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(prompts)
    return deduped


def render_reader_messages(
    prompts: ReaderPrompts,
    *,
    policy_text: str,
    input_text: str,
    labels: list[str],
) -> list[dict[str, str]]:
    validate_reader_prompts(prompts)
    replacements = {
        "policy": policy_text.strip(),
        "input": input_text,
        "labels": "\n".join(f"- {label}" for label in labels),
    }

    rendered = []
    for message in prompts.messages:
        content = PLACEHOLDER_PATTERN.sub(lambda match: replacements[match.group(1)], message.content)
        rendered.append({"role": message.role, "content": content})
    return rendered


def validate_reader_prompts(prompts: ReaderPrompts) -> None:
    for message in prompts.messages:
        for placeholder in PLACEHOLDER_PATTERN.findall(message.content):
            if placeholder not in SUPPORTED_PLACEHOLDERS:
                allowed = ", ".join(sorted(SUPPORTED_PLACEHOLDERS))
                raise ValueError(f"unsupported prompt placeholder {placeholder!r}; expected one of {allowed}")


def render_policy_doc(policy: PolicyDoc) -> str:
    lines: list[str] = []
    for section in policy.sections:
        lines.append(f"## {section.title}")
        if section.summary:
            lines.append(section.summary)
        if section.content_type == "rules":
            for rule in section.rules:
                lines.append(f"- {rule.text}")
        else:
            for subsection in section.subsections:
                lines.append(f"### {subsection.title}")
                if subsection.summary:
                    lines.append(subsection.summary)
                for rule in subsection.rules:
                    lines.append(f"- {rule.text}")
        lines.append("")

    return "\n".join(lines).strip()


def _prompt_signature(prompts: ReaderPrompts) -> tuple[tuple[str, str], ...]:
    return tuple((message.role, _normalize_message(message.content)) for message in prompts.messages)


def _normalize_message(content: str) -> str:
    return " ".join(content.strip().split())
