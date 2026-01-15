import os
import json
from typing import Dict, List, Optional

import pandas as pd
import requests
from newspaper import Article


GOOGLE_SEARCH_CACHE_DIR = "cache/google_search"
SERPER_SEARCH_CACHE_DIR = "cache/serper_search"
SCRAPE_CACHE_DIR = "cache/url_scrape"

SERPER_ENDPOINT = "https://google.serper.dev/search"
DEFAULT_FILTERED_DOMAINS = [
    ".pdf",
    "youtube.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "pinterest.com",
]


def _safe_filename(value: str) -> str:
    safe_filename = "".join(c if c.isalnum() else "_" for c in value)
    return safe_filename[:100]


def _get_cache_filename(query: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    safe_filename = _safe_filename(query)
    return os.path.join(cache_dir, f"{safe_filename}.json")


def _save_search_results(query: str, urls: List[str], cache_dir: str) -> None:
    cache_file = _get_cache_filename(query, cache_dir)
    with open(cache_file, "w") as f:
        json.dump(
            {
                "query": query,
                "timestamp": pd.Timestamp.now().isoformat(),
                "urls": urls,
            },
            f,
            indent=2,
        )


def _load_search_results(query: str, cache_dir: str) -> List[str]:
    cache_file = _get_cache_filename(query, cache_dir)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                return data.get("urls", [])
        except Exception:
            return []
    return []


def _filter_urls(urls: List[str]) -> List[str]:
    filtered = []
    for url in urls:
        if not any(x in url.lower() for x in DEFAULT_FILTERED_DOMAINS):
            filtered.append(url)
    return filtered


def google_search(query: str, num_results: int = 30, use_cache: bool = True) -> List[str]:
    from googlesearch import search
    if use_cache:
        cached_urls = _load_search_results(query, GOOGLE_SEARCH_CACHE_DIR)
        if cached_urls:
            return cached_urls

    try:
        urls = []
        for url in search(query, stop=num_results, pause=3.0):
            urls.append(url)
        urls = _filter_urls(urls)
        if urls:
            _save_search_results(query, urls, GOOGLE_SEARCH_CACHE_DIR)
        return urls
    except Exception as e:
        print(f"Error in google_search: {str(e)}")
        return []


def serper_search(
    query: str,
    num_results: int = 30,
    use_cache: bool = True,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
) -> List[str]:
    cache_key = f"{query}__{num_results}__{gl or ''}__{hl or ''}"
    if use_cache:
        cached_urls = _load_search_results(cache_key, SERPER_SEARCH_CACHE_DIR)
        if cached_urls:
            return cached_urls

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SERPER_API_KEY environment variable.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: Dict[str, object] = {"q": query}
    if gl:
        payload["gl"] = gl
    if hl:
        payload["hl"] = hl

    response = requests.post(SERPER_ENDPOINT, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    urls = []
    for item in data.get("organic", [])[:num_results]:
        url = item.get("link") or item.get("url") or ""
        if url:
            urls.append(url)

    urls = _filter_urls(urls)
    if urls and use_cache:
        _save_search_results(cache_key, urls, SERPER_SEARCH_CACHE_DIR)
    return urls


def search_urls(
    query: str,
    num_results: int = 30,
    use_cache: bool = True,
    tool: str = "google",
    serper_gl: Optional[str] = None,
    serper_hl: Optional[str] = None,
) -> List[str]:
    tool = tool.lower().strip()
    if tool == "google":
        return google_search(query, num_results=num_results, use_cache=use_cache)
    if tool == "serper":
        return serper_search(
            query,
            num_results=num_results,
            use_cache=use_cache,
            gl=serper_gl,
            hl=serper_hl,
        )
    raise ValueError(f"Unknown search tool: {tool}")


def scrape_and_parse(url: str) -> Optional[Dict[str, str]]:
    article = Article(url)
    try:
        article.download()
        article.parse()
        return {"url": url, "text": article.text}
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None


def _get_scrape_cache_filename(url: str) -> str:
    os.makedirs(SCRAPE_CACHE_DIR, exist_ok=True)
    safe_filename = _safe_filename(url)
    return os.path.join(SCRAPE_CACHE_DIR, f"{safe_filename}.json")


def _save_scraped_content(url: str, content: Dict[str, str]) -> None:
    cache_file = _get_scrape_cache_filename(url)
    with open(cache_file, "w") as f:
        json.dump(
            {
                "url": url,
                "timestamp": pd.Timestamp.now().isoformat(),
                "content": content,
            },
            f,
            indent=2,
        )


def _load_scraped_content(url: str) -> Optional[Dict[str, str]]:
    cache_file = _get_scrape_cache_filename(url)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                return data.get("content")
        except Exception as e:
            print(f"Error loading cached scraped content: {str(e)}")
            return None
    return None


def scrape_and_parse_with_cache(url: str, use_cache: bool = True) -> Optional[Dict[str, str]]:
    if use_cache:
        cached_content = _load_scraped_content(url)
        if cached_content:
            return cached_content

    content = scrape_and_parse(url)
    if content and use_cache:
        _save_scraped_content(url, content)
    return content


def chunk_text(text: str, tokenizer, chunk_size: int) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks
