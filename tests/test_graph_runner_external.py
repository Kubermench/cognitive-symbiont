from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from symbiont.agents.registry import AgentRegistry, AgentSpec, CrewSpec
from symbiont.memory.db import MemoryDB
from symbiont.orchestration.graph import GraphRunner, GraphSpec, NodeSpec


def _minimal_registry() -> AgentRegistry:
    agent = AgentSpec(name="architect", role="architect", llm={}, cache=None, tools=[])
    crew = CrewSpec(name="default", sequence=["architect"])
    return AgentRegistry({"architect": agent}, {"default": crew})


def _build_runner(tmp_path: Path, cfg: Dict[str, Any]) -> GraphRunner:
    crew_path = tmp_path / "crew.yaml"
    crew_path.write_text("agents:\n  architect:\n    role: architect\n    llm: {}\n    tools: []\ncrews:\n  default:\n    agents:\n      - architect\n")
    spec = GraphSpec(
        start="start",
        nodes={"start": NodeSpec(name="start", agent="architect")},
        crew_config=crew_path,
        simulation=None,
        parallel_groups=[],
    )
    db = MemoryDB(db_path=str(tmp_path / "symbiont.db"))
    registry = _minimal_registry()
    return GraphRunner(spec, registry, cfg, db, state_dir=tmp_path)


def test_graph_runner_fetches_external_when_enabled(monkeypatch, tmp_path: Path):
    captured: Dict[str, Any] = {}
    cfg = {
        "db_path": str(tmp_path / "symbiont.db"),
        "retrieval": {"external": {"enabled": True, "max_items": 4, "min_relevance": 0.6, "log": False}},
    }
    runner = _build_runner(tmp_path, cfg)

    def fake_fetch(query, *, max_items, min_relevance, fetcher=None):
        captured["query"] = query
        captured["max_items"] = max_items
        captured["min_relevance"] = min_relevance
        return {"accepted": [{"title": "Agentic"}], "claims": [{"id": 1}]}

    monkeypatch.setattr(runner.memory_backend, "fetch_external_context", fake_fetch)

    result = runner._maybe_fetch_external("agentic graph work")

    assert captured["query"] == "agentic graph work"
    assert captured["max_items"] == 4
    assert captured["min_relevance"] == 0.6
    assert result["accepted"]


def test_graph_runner_skips_external_when_disabled(monkeypatch, tmp_path: Path):
    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch_external_context should not be called when disabled")

    cfg = {"db_path": str(tmp_path / "symbiont.db"), "retrieval": {"external": {"enabled": False}}}
    runner = _build_runner(tmp_path, cfg)
    monkeypatch.setattr(runner.memory_backend, "fetch_external_context", fail_fetch)

    result = runner._maybe_fetch_external("skip me")

    assert result == {}
