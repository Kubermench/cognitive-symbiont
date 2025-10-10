from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pytest

from symbiont.orchestrator import Orchestrator
from symbiont.memory import retrieval as retrieval_module
from symbiont.agents import reflector as reflector_module


@pytest.fixture(autouse=True)
def redirect_state(tmp_path, monkeypatch):
    monkeypatch.setattr(reflector_module, "STATE_FILE", tmp_path / "state.json")
    yield


def _base_config(tmp_path: Path) -> Dict[str, Any]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return {
        "db_path": str(tmp_path / "symbiont.db"),
        "initiative": {
            "repo_path": str(repo_root),
            "state": {
                "backend": "sqlite",
                "path": str(tmp_path / "initiative.db"),
                "swarm_ttl_seconds": 120,
            },
        },
        "evolution": {"swarm_enabled": False},
        "llm": {"provider": "stub"},
        "max_tokens": 0,
    }


def test_auto_external_fetch_enabled(monkeypatch, tmp_path):
    captured: Dict[str, Any] = {}

    def fake_fetch(db, query, *, max_items, min_relevance, fetcher=None):
        captured["query"] = query
        captured["max_items"] = max_items
        captured["min_relevance"] = min_relevance
        return {"accepted": [{"title": "Agentic"}], "claims": [{"id": 1}]}

    monkeypatch.setattr(retrieval_module, "fetch_external_context", fake_fetch)

    cfg = _base_config(tmp_path)
    cfg["retrieval"] = {"external": {"enabled": True, "max_items": 3, "min_relevance": 0.55, "log": False}}
    orch = Orchestrator(cfg)

    result = orch._maybe_fetch_external("agentic AI")

    assert captured["query"] == "agentic AI"
    assert captured["max_items"] == 3
    assert captured["min_relevance"] == 0.55
    assert result["accepted"], "Expected external context when enabled"


def test_auto_external_fetch_disabled(monkeypatch, tmp_path):
    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch_external_context should not be invoked when disabled")

    monkeypatch.setattr(retrieval_module, "fetch_external_context", fail_fetch)

    cfg = _base_config(tmp_path)
    cfg["retrieval"] = {"external": {"enabled": False}}
    orch = Orchestrator(cfg)

    result = orch._maybe_fetch_external("agentic AI")

    assert result == {}
