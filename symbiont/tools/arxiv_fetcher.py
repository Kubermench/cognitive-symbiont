"""Simple arXiv client and search helper."""

from __future__ import annotations

import datetime as _dt
import threading
import time
from dataclasses import dataclass
from typing import List

import feedparser
import requests


API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "symbiont-foresight/1.0"
API_TIMEOUT = 10
RATE_LIMIT_SECONDS = 2.0  # Keep at most ~30 requests/minute (well under arXiv guidelines).

_SESSION = requests.Session()
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


@dataclass
class ArxivResult:
    """Normalized view of an arXiv paper."""

    title: str
    summary: str
    link: str
    published: str
    authors: List[str]

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "published": self.published,
            "authors": ", ".join(self.authors),
        }


def _parse_entry(entry: dict) -> ArxivResult:
    title = (entry.get("title") or "").strip()
    summary = (entry.get("summary") or "").strip()
    link = ""
    for link_obj in entry.get("links", []):
        if link_obj.get("rel") == "alternate":
            link = link_obj.get("href", "")
            break
    published = entry.get("published") or ""
    if published:
        try:
            published = _dt.datetime.fromisoformat(published.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            published = published[:10]
    authors = [a.get("name", "") for a in entry.get("authors", [])]
    return ArxivResult(title=title, summary=summary, link=link, published=published, authors=authors)


def _throttle_requests() -> None:
    """Rate-limit outbound API calls to stay within arXiv guidelines."""

    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        wait = RATE_LIMIT_SECONDS - (now - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _LAST_REQUEST_AT = now


def search_arxiv(query: str, *, max_results: int = 5) -> List[ArxivResult]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT}
    _throttle_requests()
    try:
        resp = _SESSION.get(API_URL, params=params, headers=headers, timeout=API_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    parsed = feedparser.parse(resp.text)
    entries = parsed.get("entries", [])
    results: List[ArxivResult] = []
    for entry in entries[:max_results]:
        try:
            results.append(_parse_entry(entry))
        except Exception:
            continue
    return results


__all__ = ["ArxivResult", "search_arxiv"]
