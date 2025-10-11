from __future__ import annotations

from typing import Any, Dict, Optional

from .backends import resolve_backend
from .db import MemoryDB
from .external_sources import ExternalSourceFetcher


def build_indices(
    db: MemoryDB,
    limit_if_new: Optional[int] = None,
    *,
    backend: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Embed new memory content using the configured backend."""

    return resolve_backend(backend, db, config=config).build_indices(limit_if_new=limit_if_new)


def search(
    db: MemoryDB,
    query: str,
    *,
    k: int = 5,
    backend: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Search the configured memory backend."""

    return resolve_backend(backend, db, config=config).search(query, k=k)


def fetch_external_context(
    db: MemoryDB,
    query: str,
    *,
    max_items: int = 8,
    min_relevance: float = 0.7,
    fetcher: Optional[ExternalSourceFetcher] = None,
    backend: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch and merge external research through the configured backend."""

    return resolve_backend(backend, db, config=config).fetch_external_context(
        query,
        max_items=max_items,
        min_relevance=min_relevance,
        fetcher=fetcher,
    )
