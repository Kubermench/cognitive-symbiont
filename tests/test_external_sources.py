from __future__ import annotations

import json
from pathlib import Path

import pytest

from symbiont.memory.db import MemoryDB
from symbiont.memory.external_sources import ExternalSourceFetcher, fetch_and_store_external_context
from symbiont.memory import graphrag


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Agentic AI for Reliable Systems</title>
    <summary>This paper explores guarded workflows for autonomous agents.</summary>
    <published>2025-01-01T00:00:00Z</published>
    <link rel="alternate" href="http://arxiv.org/abs/1234.5678v1"/>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
  </entry>
</feed>
"""

SEMANTIC_RESPONSE = {
    "data": [
        {
            "paperId": "S123",
            "title": "Semantic Bridges for Agentic Systems",
            "abstract": "Bridging localized agents with external knowledge graphs.",
            "url": "https://semanticscholar.org/paper/S123",
            "venue": "ICLR",
            "year": 2025,
            "authors": [{"name": "Ada Lovelace"}],
        }
    ]
}


class FakeSession:
    def __init__(self):
        self._calls = 0

    @property
    def call_count(self) -> int:
        return self._calls

    def get(self, url, params=None, headers=None, timeout=None):  # noqa: D401 - mimic requests API
        self._calls += 1
        if "arxiv.org/api/query" in url:
            return _FakeResponse(text=ARXIV_FEED)
        if "semanticscholar.org/graph" in url:
            return _FakeResponse(payload=SEMANTIC_RESPONSE)
        raise AssertionError(f"Unexpected URL requested: {url}")


class _FakeResponse:
    def __init__(self, text: str | None = None, payload: dict | None = None):
        self.text = text or ""
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("JSON requested for non-JSON response")
        return json.loads(json.dumps(self._payload))


@pytest.fixture()
def memory_db(tmp_path: Path) -> MemoryDB:
    db_path = tmp_path / "symbiont.db"
    db = MemoryDB(db_path=str(db_path))
    db.ensure_schema()
    return db


def test_fetcher_uses_cache_and_merges_claims(memory_db: MemoryDB, tmp_path: Path) -> None:
    session = FakeSession()
    fetcher = ExternalSourceFetcher(cache_dir=tmp_path / "cache", ttl_seconds=3600, session=session)

    first_items = fetcher.search("agentic systems", max_items=4, min_relevance=0.0)
    assert session.call_count == 2  # arxiv + semantic scholar
    assert len(first_items) >= 2

    cached_items = fetcher.search("agentic systems", max_items=4, min_relevance=0.0)
    assert cached_items, "Expected cached items to be returned"
    assert session.call_count == 2, "Cache hit should avoid additional HTTP calls"

    result = fetch_and_store_external_context(
        memory_db,
        "agentic systems",
        max_items=4,
        min_relevance=0.5,
        fetcher=fetcher,
    )
    assert result["accepted"], "High-relevance items should be accepted"
    assert result["claims"], "Claims should be recorded for accepted items"

    claims = graphrag.query_claims(memory_db, "agentic systems", limit=10)
    assert claims, "GraphRAG should contain inserted claims"
    subjects = {claim["subject"] for claim in claims}
    assert "agentic systems" in subjects

    titles = {claim["object"] for claim in claims}
    assert any("Agentic AI for Reliable Systems" in title for title in titles)


def test_fetcher_enforces_rate_limit(monkeypatch, tmp_path: Path) -> None:
    session = FakeSession()
    fetcher = ExternalSourceFetcher(
        cache_dir=tmp_path / "cache",
        ttl_seconds=3600,
        session=session,
        rate_limit_seconds=2.0,
    )

    timeline = {"now": 0.0}
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return timeline["now"]

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)
        timeline["now"] += duration

    monkeypatch.setattr("symbiont.memory.external_sources.time.monotonic", fake_monotonic)
    monkeypatch.setattr("symbiont.memory.external_sources.time.sleep", fake_sleep)

    list(fetcher._fetch_arxiv(query="agentic systems", limit=1))
    timeline["now"] += 0.5
    list(fetcher._fetch_arxiv(query="agentic systems", limit=1))

    assert session.call_count == 2
    assert sleeps and pytest.approx(sleeps[0], rel=1e-3) == 1.5
