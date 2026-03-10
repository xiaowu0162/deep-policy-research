from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .artifacts import ChunkRecord, SourceRecord


DEFAULT_USER_AGENT = "deep-policy-research/0.1 (+https://github.com/openai)"
DEFAULT_CHUNK_SIZE = 2400
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 200


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    url: str
    retrieved_at: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"url": self.url, "retrieved_at": self.retrieved_at, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "FetchedDocument":
        return cls(url=data["url"], retrieved_at=data["retrieved_at"], text=data["text"])


class UrlFetcher(Protocol):
    def fetch(self, url: str) -> FetchedDocument | None:
        ...


class CachedUrlFetcher:
    def __init__(self, *, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str) -> FetchedDocument | None:
        cache_path = self.cache_dir / f"{sha256(url.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            return FetchedDocument.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))

        request = Request(
            url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw_bytes = response.read()
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
        except HTTPError:
            return None
        except URLError:
            return None

        decoded = raw_bytes.decode(charset, errors="replace")
        if content_type == "text/plain":
            text = _normalize_text(decoded)
        else:
            text = _normalize_text(_extract_text_from_html(decoded))
        if not text:
            return None

        document = FetchedDocument(
            url=url,
            retrieved_at=_utc_now(),
            text=text,
        )
        cache_path.write_text(json.dumps(document.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return document


def create_url_fetcher(*, cache_dir: Path) -> UrlFetcher:
    return CachedUrlFetcher(cache_dir=cache_dir)


def merge_sources(
    existing_sources: list[SourceRecord],
    fetched_documents: list[FetchedDocument],
    *,
    query_by_url: dict[str, list[str]],
) -> list[SourceRecord]:
    sources_by_url = {source.url: source for source in existing_sources}
    next_index = len(existing_sources) + 1

    for url, queries in query_by_url.items():
        existing = sources_by_url.get(url)
        if existing is not None:
            existing.queries = list(dict.fromkeys([*existing.queries, *queries]))

    for document in fetched_documents:
        queries = list(dict.fromkeys(query_by_url.get(document.url, [])))
        existing = sources_by_url.get(document.url)
        if existing is not None:
            existing.queries = list(dict.fromkeys([*existing.queries, *queries]))
            if document.text and len(document.text) > len(existing.text):
                existing.text = document.text
                existing.retrieved_at = document.retrieved_at
            continue

        source = SourceRecord(
            source_id=f"src_{next_index:04d}",
            url=document.url,
            retrieved_at=document.retrieved_at,
            queries=queries,
            text=document.text,
        )
        sources_by_url[source.url] = source
        next_index += 1

    return sorted(sources_by_url.values(), key=lambda item: item.source_id)


def chunk_sources(
    sources: list[SourceRecord],
    *,
    starting_index: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    next_index = starting_index
    for source in sources:
        for chunk_index, text in enumerate(
            split_text_into_chunks(source.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"chunk_{next_index:05d}",
                    source_id=source.source_id,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            next_index += 1
    return chunks


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    normalized = _normalize_text(text)
    if len(normalized) <= chunk_size:
        return [normalized] if normalized else []

    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.extend(_flush_chunk(current, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        current = paragraph
    if current:
        chunks.extend(_flush_chunk(current, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    filtered = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]
    return filtered or [chunk for chunk in chunks if chunk]


def _flush_chunk(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return pieces


def _extract_text_from_html(html: str) -> str:
    stripped = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    parser = _HTMLTextExtractor()
    parser.feed(stripped)
    parser.close()
    return parser.get_text()


def _normalize_text(text: str) -> str:
    value = unescape(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    paragraphs: list[str] = []
    current_lines: list[str] = []
    for line in value.split("\n"):
        stripped = line.strip()
        if stripped:
            current_lines.append(stripped)
            continue
        if current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []
    if current_lines:
        paragraphs.append(" ".join(current_lines))
    return "\n\n".join(paragraphs).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        text = data.strip()
        if text:
            self._parts.append(text)
            self._parts.append(" ")

    def get_text(self) -> str:
        return "".join(self._parts)
