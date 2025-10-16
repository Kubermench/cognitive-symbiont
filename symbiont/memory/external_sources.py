from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests
from .graphrag import add_claim
from .db import MemoryDB


CACHE_DIR = Path("data/external")
DEFAULT_TTL_SECONDS = 24 * 3600  # one day
USER_AGENT = "SymbiontExternalFetcher/0.1 (+https://github.com/)"


_DEFAULT_FETCHER: Optional["ExternalSourceFetcher"] = None


@dataclass
class ExternalItem:
    """Normalized representation of an external source result."""

    source: str
    title: str
    url: str
    summary: str
    relevance: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "relevance": self.relevance,
            "metadata": self.metadata,
        }


class ExternalSourceFetcher:
    """Fetches and caches external context from research feeds."""

    def __init__(
        self,
        *,
        cache_dir: Path | str = CACHE_DIR,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        session: Optional[requests.Session] = None,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._session = session or requests.Session()
        self._rate_limit_seconds = max(0.0, float(rate_limit_seconds))
        self._last_call: Dict[str, float] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        max_items: int = 8,
        min_relevance: float = 0.0,
    ) -> List[ExternalItem]:
        query = query.strip()
        if not query:
            return []

        cached = self._read_cache(query, max_items)
        if cached is not None:
            return [ExternalItem(**item) for item in cached if item.get("relevance", 0.0) >= min_relevance]

        fresh = self._fetch_live(query=query, max_items=max_items)
        self._write_cache(query, max_items, [item.to_dict() for item in fresh])
        if min_relevance <= 0:
            return fresh
        return [item for item in fresh if item.relevance >= min_relevance]

    # ------------------------------------------------------------------
    def _fetch_live(self, *, query: str, max_items: int) -> List[ExternalItem]:
        window = max(1, min(20, int(max_items)))
        items: List[ExternalItem] = []
        try:
            items.extend(self._fetch_arxiv(query=query, limit=window))
        except Exception:
            # Network parse errors are tolerated; we still try other feeds.
            pass

        try:
            items.extend(self._fetch_semantic_scholar(query=query, limit=window))
        except Exception:
            pass

        deduped: Dict[str, ExternalItem] = {}
        for item in items:
            key = item.url or item.title
            if key in deduped:
                # keep the higher relevance version
                if item.relevance > deduped[key].relevance:
                    deduped[key] = item
            else:
                deduped[key] = item

        ranked = sorted(deduped.values(), key=lambda it: it.relevance, reverse=True)
        return ranked[:window]

    # ------------------------------------------------------------------
    def _fetch_arxiv(self, *, query: str, limit: int) -> Iterable[ExternalItem]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(10, limit)),
        }
        url = "http://export.arxiv.org/api/query"
        self._throttle(url)
        response = self._session.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()

        import xml.etree.ElementTree as ET

        try:
            feed = ET.fromstring(response.text)
        except ET.ParseError:
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in feed.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = ""
            for link_elem in entry.findall("atom:link", ns):
                if (link_elem.attrib.get("rel") == "alternate") and link_elem.attrib.get("href"):
                    link = link_elem.attrib["href"]
                    break
            link = link or (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [
                (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                for author in entry.findall("atom:author", ns)
            ]
            metadata = {"published": published, "authors": [a for a in authors if a]}
            relevance = self._score_relevance(query, title, summary)
            yield ExternalItem(
                source="arxiv",
                title=_clean_whitespace(title),
                summary=_clean_whitespace(summary),
                url=link,
                relevance=relevance,
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    def _fetch_semantic_scholar(self, *, query: str, limit: int) -> Iterable[ExternalItem]:
        params = {
            "query": query,
            "limit": max(1, min(10, limit)),
            "fields": "title,abstract,url,venue,year,authors",
        }
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self._throttle(url)
        response = self._session.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        for item in data:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            abstract = str(item.get("abstract") or "").strip()
            url = str(item.get("url") or "").strip()
            metadata = {
                "venue": item.get("venue"),
                "year": item.get("year"),
                "authors": [auth.get("name") for auth in item.get("authors") or [] if auth.get("name")],
                "paperId": item.get("paperId"),
            }
            relevance = self._score_relevance(query, title, abstract)
            yield ExternalItem(
                source="semantic_scholar",
                title=_clean_whitespace(title),
                summary=_clean_whitespace(abstract),
                url=url,
                relevance=relevance,
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    def _read_cache(self, query: str, max_items: int) -> Optional[List[Dict[str, Any]]]:
        path = self._cache_path(query, max_items)
        if not path.exists():
            return None
        try:
            if time.time() - path.stat().st_mtime > self.ttl_seconds:
                return None
        except OSError:
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(raw, dict) and "items" in raw:
            items = raw.get("items") or []
        else:
            items = raw
        if not isinstance(items, list):
            return None
        return items

    def _write_cache(self, query: str, max_items: int, payload: List[Dict[str, Any]]) -> None:
        path = self._cache_path(query, max_items)
        try:
            path.write_text(
                json.dumps(
                    {
                        "query": query,
                        "max_items": max_items,
                        "items": payload,
                        "updated_at": int(time.time()),
                        "item_count": len(payload),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _cache_path(self, query: str, max_items: int) -> Path:
        fingerprint = hashlib.sha256(f"{query}|{max_items}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{fingerprint}.json"

    def _throttle(self, url: str) -> None:
        if self._rate_limit_seconds <= 0:
            return
        try:
            host = urlparse(url).netloc or "default"
        except Exception:
            host = "default"
        with self._lock:
            last = self._last_call.get(host)
            now = time.monotonic()
            if last is not None:
                wait_for = self._rate_limit_seconds - (now - last)
                if wait_for > 0:
                    time.sleep(wait_for)
                    now = time.monotonic()
            self._last_call[host] = now

    # ------------------------------------------------------------------
    def _score_relevance(self, query: str, title: str, summary: str) -> float:
        tokens_query = _tokenize(query)
        tokens_text = _tokenize(f"{title} {summary}")
        if not tokens_query or not tokens_text:
            return 0.15
        overlap = len(tokens_query & tokens_text)
        ratio = overlap / max(1, len(tokens_query))
        boost = 0.1 if len(title) <= 120 else 0.0
        score = min(1.0, max(0.05, ratio + boost))
        return round(score, 4)


# ----------------------------------------------------------------------
def fetch_and_store_external_context(
    db: MemoryDB,
    query: str,
    *,
    max_items: int = 8,
    min_relevance: float = 0.7,
    fetcher: Optional[ExternalSourceFetcher] = None,
) -> Dict[str, Any]:
    """Fetch external knowledge and merge high-confidence claims into GraphRAG."""

    global _DEFAULT_FETCHER
    if fetcher is None:
        if _DEFAULT_FETCHER is None:
            _DEFAULT_FETCHER = ExternalSourceFetcher()
        fetcher = _DEFAULT_FETCHER
    items = fetcher.search(query, max_items=max_items, min_relevance=min_relevance)
    accepted: List[ExternalItem] = []
    claims: List[Dict[str, Any]] = []

    for item in items:
        if item.relevance < min_relevance:
            continue
        accepted.append(item)
        importance = float(max(min(item.relevance, 1.0), 0.05))

        claim_id, status = add_claim(
            db,
            subject=query,
            relation="supported_by",
            obj=item.title,
            importance=importance,
            source_url=item.url or None,
        )
        claims.append(
            {
                "id": claim_id,
                "status": status,
                "subject": query,
                "relation": "supported_by",
                "object": item.title,
                "source": item.source,
                "relevance": item.relevance,
                "url": item.url,
            }
        )

        if item.summary:
            summary_obj = item.summary[:480]
            summary_claim_id, summary_status = add_claim(
                db,
                subject=item.title,
                relation="summary",
                obj=summary_obj,
                importance=max(importance * 0.9, 0.05),
                source_url=item.url or None,
            )
            claims.append(
                {
                    "id": summary_claim_id,
                    "status": summary_status,
                    "subject": item.title,
                    "relation": "summary",
                    "object": summary_obj,
                    "source": item.source,
                    "relevance": item.relevance,
                    "url": item.url,
                }
            )

    return {
        "query": query,
        "accepted": [item.to_dict() for item in accepted],
        "claims": claims,
    }


# ----------------------------------------------------------------------
def list_cache_entries(cache_dir: Path | str = CACHE_DIR) -> List[Dict[str, Any]]:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            items = payload.get("items") or []
            if not isinstance(items, list):
                items = []
            entry = {
                "path": str(path),
                "query": str(payload.get("query") or "unknown"),
                "max_items": payload.get("max_items"),
                "items": items,
                "updated_at": payload.get("updated_at"),
                "item_count": payload.get("item_count", len(items)),
            }
        elif isinstance(payload, list):
            entry = {
                "path": str(path),
                "query": "unknown",
                "max_items": None,
                "items": payload,
            }
        else:
            continue
        try:
            stat = path.stat()
            entry.setdefault("updated_at", int(stat.st_mtime))
            entry["size"] = stat.st_size
        except OSError:
            entry["updated_at"] = None
            entry["size"] = None
        entry.setdefault("item_count", len(entry["items"]))
        entries.append(entry)
    return entries


def clear_cache(
    *,
    query: Optional[str] = None,
    max_items: Optional[int] = None,
    cache_dir: Path | str = CACHE_DIR,
) -> int:
    """
    Clear cached external fetches.

    Returns the number of cache files removed.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    if query:
        if max_items is None:
            raise ValueError("max_items is required when clearing a specific query cache")
        fetcher = ExternalSourceFetcher(cache_dir=cache_dir)
        path = fetcher._cache_path(query, max_items)
        if path.exists():
            path.unlink()
            return 1
        return 0
    removed = 0
    for path in cache_dir.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


# ----------------------------------------------------------------------
def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tok for tok in tokens if len(tok) > 2}
