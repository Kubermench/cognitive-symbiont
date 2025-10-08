"""Simple arXiv client and search helper."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from typing import Iterable, List

import feedparser
import requests


API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "symbiont-foresight/1.0"


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


def search_arxiv(query: str, *, max_results: int = 5) -> List[ArxivResult]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(API_URL, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
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
