from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .spec import SearchConfig


SERPER_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_FILTERED_DOMAINS = {
    ".pdf",
    "youtube.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "pinterest.com",
}


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    url: str
    rank: int
    title: str | None = None
    snippet: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "url": self.url,
            "rank": self.rank,
            "title": self.title,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SearchResult":
        return cls(
            query=str(data["query"]),
            url=str(data["url"]),
            rank=int(data["rank"]),
            title=_optional_str(data.get("title")),
            snippet=_optional_str(data.get("snippet")),
        )


class SearchProvider(Protocol):
    def search(self, query: str, *, num_results: int | None = None) -> list[SearchResult]:
        ...


class SerperSearchProvider:
    def __init__(self, config: SearchConfig, *, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, *, num_results: int | None = None) -> list[SearchResult]:
        limit = num_results or self.config.num_results
        cache_path = self.cache_dir / f"{_cache_key(query, self.config, limit)}.json"
        if cache_path.exists():
            return _load_cached_results(cache_path)

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"environment variable {self.config.api_key_env!r} is required for search provider "
                f"{self.config.provider!r}"
            )

        payload = {
            "q": query,
            "gl": self.config.country,
            "hl": self.config.language,
            "num": limit,
        }
        request = Request(
            SERPER_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network errors are environment dependent
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"serper search failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:  # pragma: no cover - network errors are environment dependent
            raise RuntimeError(f"serper search failed: {exc}") from exc

        results = []
        for rank, item in enumerate(payload.get("organic", [])[:limit], start=1):
            url = str(item.get("link") or item.get("url") or "").strip()
            if not url or _should_skip_url(url):
                continue
            results.append(
                SearchResult(
                    query=query,
                    url=url,
                    rank=rank,
                    title=_optional_str(item.get("title")),
                    snippet=_optional_str(item.get("snippet")),
                )
            )

        cache_path.write_text(
            json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return results


class FixtureSearchProvider:
    def __init__(self, config: SearchConfig, *, cache_dir: Path):
        del cache_dir
        if not config.fixture_path:
            raise ValueError("search.fixture_path is required for provider='fixture'")
        self.config = config
        self.fixture_path = Path(config.fixture_path).expanduser().resolve()
        self.payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))

    def search(self, query: str, *, num_results: int | None = None) -> list[SearchResult]:
        limit = num_results or self.config.num_results
        query_map = self.payload.get("queries", {}) if isinstance(self.payload, dict) else {}
        raw_results = query_map.get(query) or query_map.get("*") or query_map.get("default") or []
        results: list[SearchResult] = []
        for rank, item in enumerate(raw_results[:limit], start=1):
            if not isinstance(item, dict):
                continue
            url = _resolve_fixture_result_url(item.get("url"), base_dir=self.fixture_path.parent)
            if not url or _should_skip_url(url):
                continue
            results.append(
                SearchResult(
                    query=query,
                    url=url,
                    rank=rank,
                    title=_optional_str(item.get("title")),
                    snippet=_optional_str(item.get("snippet")),
                )
            )
        return results


def create_search_provider(config: SearchConfig, *, cache_dir: Path) -> SearchProvider:
    provider = config.provider.strip().lower()
    if provider == "serper":
        return SerperSearchProvider(config, cache_dir=cache_dir)
    if provider == "fixture":
        return FixtureSearchProvider(config, cache_dir=cache_dir)
    raise ValueError(f"unsupported search provider {config.provider!r}")


def _load_cached_results(cache_path: Path) -> list[SearchResult]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [SearchResult.from_dict(item) for item in data]


def _cache_key(query: str, config: SearchConfig, limit: int) -> str:
    payload = {
        "query": query,
        "provider": config.provider,
        "country": config.country,
        "language": config.language,
        "limit": limit,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _should_skip_url(url: str) -> bool:
    lowered = url.lower()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return any(token in lowered or token == host for token in DEFAULT_FILTERED_DOMAINS)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_fixture_result_url(value: object, *, base_dir: Path) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    path = (base_dir / text).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixture search result path does not exist: {path}")
    return path.as_uri()
