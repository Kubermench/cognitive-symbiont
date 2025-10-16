from __future__ import annotations

import json
from pathlib import Path

from symbiont.memory.external_sources import (
    ExternalSourceFetcher,
    ExternalItem,
    clear_cache,
    list_cache_entries,
)


def _make_item() -> ExternalItem:
    return ExternalItem(
        source="arxiv",
        title="Agentic Systems",
        url="http://example.com",
        summary="Guarded autonomy bridges.",
        relevance=0.9,
        metadata={"year": 2025},
    )


def test_list_cache_entries_reads_metadata(tmp_path: Path) -> None:
    fetcher = ExternalSourceFetcher(cache_dir=tmp_path, ttl_seconds=3600)
    items = [_make_item().to_dict()]
    fetcher._write_cache("agentic systems", 4, items)

    entries = list_cache_entries(cache_dir=tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["query"] == "agentic systems"
    assert entry["max_items"] == 4
    assert entry["item_count"] == 1
    assert entry["items"][0]["title"] == "Agentic Systems"


def test_list_cache_entries_supports_legacy_format(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps([{"title": "Legacy"}]), encoding="utf-8")

    entries = list_cache_entries(cache_dir=tmp_path)
    assert entries
    entry = entries[0]
    assert entry["query"] == "unknown"
    assert entry["item_count"] == 1


def test_clear_cache_handles_specific_and_all(tmp_path: Path) -> None:
    fetcher = ExternalSourceFetcher(cache_dir=tmp_path, ttl_seconds=3600)
    fetcher._write_cache("agentic cleanup", 2, [_make_item().to_dict()])
    fetcher._write_cache("keep", 3, [_make_item().to_dict()])

    removed = clear_cache(query="agentic cleanup", max_items=2, cache_dir=tmp_path)
    assert removed == 1
    entries = list_cache_entries(cache_dir=tmp_path)
    assert all(entry["query"] != "agentic cleanup" for entry in entries)

    removed_all = clear_cache(cache_dir=tmp_path)
    assert removed_all >= 1
    assert not list_cache_entries(cache_dir=tmp_path)
