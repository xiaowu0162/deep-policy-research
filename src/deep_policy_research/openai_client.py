from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI, OpenAI

from .spec import ModelConfig


def create_sync_client(config: ModelConfig) -> OpenAI:
    return OpenAI(
        base_url=config.base_url,
        api_key=_resolve_api_key(config),
        **config.client_kwargs,
    )


def create_async_client(config: ModelConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=_resolve_api_key(config),
        **config.client_kwargs,
    )


def chat_completion_request(
    config: ModelConfig,
    messages: list[dict[str, str]],
    **overrides: Any,
) -> dict[str, Any]:
    request = dict(config.request_defaults)
    request.update(overrides)
    request["model"] = config.model
    request["messages"] = messages
    return request


def create_chat_completion(
    client: OpenAI,
    config: ModelConfig,
    messages: list[dict[str, str]],
    **overrides: Any,
) -> Any:
    request = chat_completion_request(config, messages, **overrides)
    return client.chat.completions.create(**request)


async def create_chat_completion_async(
    client: AsyncOpenAI,
    config: ModelConfig,
    messages: list[dict[str, str]],
    **overrides: Any,
) -> Any:
    request = chat_completion_request(config, messages, **overrides)
    return await client.chat.completions.create(**request)


def probe_model(client: OpenAI, config: ModelConfig, retries: int = 1) -> None:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            create_chat_completion(
                client,
                config,
                [{"role": "user", "content": "Reply with OK."}],
                max_tokens=1,
                temperature=0,
                timeout=10.0,
            )
            return
        except Exception as exc:  # pragma: no cover - exercised through tests with a fake client
            last_error = exc

    raise RuntimeError(
        f"failed reachability probe for model {config.model!r} at {config.base_url!r}: {last_error}"
    ) from last_error


async def probe_model_async(client: AsyncOpenAI, config: ModelConfig, retries: int = 1) -> None:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            await create_chat_completion_async(
                client,
                config,
                [{"role": "user", "content": "Reply with OK."}],
                max_tokens=1,
                temperature=0,
                timeout=10.0,
            )
            return
        except Exception as exc:  # pragma: no cover - exercised through tests with a fake client
            last_error = exc

    raise RuntimeError(
        f"failed reachability probe for model {config.model!r} at {config.base_url!r}: {last_error}"
    ) from last_error


def extract_text_from_chat_response(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(content)


def _resolve_api_key(config: ModelConfig) -> str:
    api_key = os.environ.get(config.api_key_env)
    if api_key:
        return api_key
    if _is_local_base_url(config.base_url):
        return "EMPTY"
    raise ValueError(
        f"environment variable {config.api_key_env!r} is required for model {config.model!r}"
    )


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}
