import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from symbiont.agents.registry import AgentRegistry, CrewRunner
from symbiont.llm.client import LLMClient
from symbiont.memory.db import MemoryDB
from symbiont.memory.dynamic_analyzer import BayesianTrendAnalyzer
from symbiont.tools import research, systems_os


def test_foresight_weaver_creates_log(monkeypatch, tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    (systems_dir / "Foresight.md").write_text(
        "## Foresight Research Log\n\n| Date | Topic | Highlight | Proposal | Approval |\n|------|-------|-----------|----------|----------|\n",
        encoding="utf-8",
    )
    original_root = systems_os.SYSTEMS_ROOT
    systems_os.SYSTEMS_ROOT = systems_dir  # type: ignore

    outputs = {
        "research:scout": json.dumps(
            {
                "topic": "Agentic self-evolution",
                "items": [
                    {
                        "title": "Scaling Agents via Continual Pre-training",
                        "url": "https://arxiv.org/abs/2509.13310",
                        "summary": "Continual pre-training teaches agent behaviours",
                    }
                ],
            }
        ),
        "research:analyze": json.dumps(
            {
                "highlight": "Continual pretraining boosts eternal mode",
                "implication": "Add offline skill loop to reflector",
            }
        ),
        "research:proposal": json.dumps(
            {
                "proposal": "Add continual_pretrain stub",
                "diff": "diff --git a/reflector.py b/reflector.py\n+def continual_pretrain(): pass\n",
            }
        ),
        "research:validate": json.dumps(
            {
                "approve": True,
                "risk": 0.3,
                "tests": ["pytest", "chaos 20%"],
            }
        ),
    }

    def fake_generate(self, prompt, label=None):  # type: ignore[override]
        return outputs.get(label or "", json.dumps({"summary": "ok"}))

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    cfg = {"db_path": str(tmp_path / "sym.db"), "llm": {}}
    crew_yaml = Path("configs/crews/foresight_weaver.yaml")
    registry = AgentRegistry.from_yaml(crew_yaml)
    db = MemoryDB(cfg["db_path"])
    runner = CrewRunner(registry, cfg, db)
    runner.run("foresight_weaver", "Self-research agenda")

    log = (systems_dir / "Foresight.md").read_text(encoding="utf-8")
    assert "Continual pretraining" in log

    systems_os.SYSTEMS_ROOT = original_root  # type: ignore


def test_foresight_metadata_artifacts(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    (systems_dir / "Foresight.md").write_text(
        "## Foresight Research Log\n\n| Date | Topic | Highlight | Proposal | Approval |\n|------|-------|-----------|----------|----------|\n",
        encoding="utf-8",
    )
    original_root = systems_os.SYSTEMS_ROOT
    systems_os.SYSTEMS_ROOT = systems_dir  # type: ignore

    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"fake")

    def fake_scout(llm, goal, max_items=3):
        return {
            "topic": "Metadata Test",
            "items": [
                {
                    "title": "Signal",
                    "url": "https://example.com/signal",
                    "summary": "High quality source",
                }
            ],
            "meta": {
                "token_delta": 42,
                "cost_estimate": 0.0123,
                "total_candidates": 2,
                "dropped_low_score": 1,
                "source_breakdown": {"arxiv": {"count": 1, "avg_score": 1.2}},
                "source_plot": str(plot_path),
            },
        }

    def fake_analyze(llm, topic, sources):
        return {"topic": topic, "highlight": "Insight", "implication": "Act"}

    def fake_proposal(llm, insight):
        return {"proposal": "", "diff": ""}

    def fake_validate(llm, proposal):
        return {"approve": False, "risk": 0.7, "tests": []}

    monkeypatch.setattr(research, "scout_insights", fake_scout)
    monkeypatch.setattr(research, "analyze_insights", fake_analyze)
    monkeypatch.setattr(research, "draft_proposal", fake_proposal)
    monkeypatch.setattr(research, "validate_proposal", fake_validate)

    cfg = {"db_path": str(tmp_path / "sym.db"), "llm": {}}
    crew_yaml = repo_root / "configs/crews/foresight_weaver.yaml"
    registry = AgentRegistry.from_yaml(crew_yaml)
    db = MemoryDB(cfg["db_path"])
    runner = CrewRunner(registry, cfg, db)

    monkeypatch.chdir(tmp_path)
    artifact_path = runner.run("foresight_weaver", "Goal Meta")

    meta_dir = Path("data/artifacts/foresight/meta")
    meta_files = sorted(meta_dir.glob("*_meta.json"))
    assert meta_files, "expected foresight meta artifact"
    meta_payload = json.loads(meta_files[-1].read_text(encoding="utf-8"))
    assert meta_payload["meta"].get("token_delta") is not None
    assert meta_payload["meta"]["dropped_low_score"] == 1
    breakdown = meta_payload["meta"].get("source_breakdown", {})
    assert breakdown.get("arxiv", {}).get("count") == 1

    with sqlite3.connect(cfg["db_path"]) as conn:
        rows = conn.execute(
            "SELECT id, summary FROM artifacts WHERE type='foresight_meta' ORDER BY id DESC"
        ).fetchall()
        assert rows, "artifact summary missing"
        summary = rows[0][1]
        assert "tokens=" in summary

        vector_rows = conn.execute(
            "SELECT ref_id FROM vectors WHERE ref_table='artifacts' AND kind='artifact' ORDER BY id DESC"
        ).fetchall()
        assert any(r[0] == rows[0][0] for r in vector_rows), "embedding not stored"

    crew_payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    suggester_output = next(
        (entry for entry in crew_payload["outputs"] if entry["agent"] == "foresight_suggester_agent"),
        None,
    )
    assert suggester_output is not None
    proposal = suggester_output["result"].get("proposal", "")
    assert "Document and review" in proposal  # fallback engaged

    systems_os.SYSTEMS_ROOT = original_root  # type: ignore


def test_async_trend_hunt_merges_sources(monkeypatch):
    monkeypatch.setattr(
        research,
        "_fetch_live_sources",
        lambda goal, limit: [
            {
                "title": "Async signal",
                "summary": goal,
                "source": "arxiv",
                "url": "https://example.com/async",
                "published": "2025-01-01",
            }
        ],
    )
    monkeypatch.setattr(
        research,
        "call_peer_collaborators",
        lambda llm, goal, models=None, max_items=2: {
            "items": [
                {
                    "title": "Peer ping",
                    "summary": "Collaborator insight",
                    "source": "peer",
                    "url": "https://peer.example",
                    "peer_support": 0.9,
                }
            ],
            "source": "peer",
            "contributors": ["peer"],
        },
    )
    result = asyncio.run(
        research.gather_trend_sources_async(
            LLMClient({}),
            "async foresight",
            include_collaborators=True,
        )
    )
    assert any(item["source"] == "peer" for item in result["items"])


def test_async_trend_handles_failures(monkeypatch):
    monkeypatch.setattr(research, "_fetch_live_sources", lambda *_: [])

    def _boom(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr(research, "call_peer_collaborators", _boom)
    result = asyncio.run(
        research.gather_trend_sources_async(
            LLMClient({}),
            "resilient",
            include_collaborators=True,
        )
    )
    assert result["items"] == []


@given(
    priors=st.dictionaries(
        keys=st.sampled_from(["arxiv", "peer", "rss", "web"]),
        values=st.floats(min_value=0.05, max_value=0.95),
        min_size=1,
    )
)
def test_bayesian_posterior_bounds(priors):
    analyzer = BayesianTrendAnalyzer(priors=priors)
    source = next(iter(priors.keys()))
    item = {
        "source": source,
        "summary": "agentic foresight autonomy",
        "published": "2025-01-01",
        "peer_support": 0.5,
    }
    ranked = analyzer.rank_sources("agentic foresight", [item])
    assert ranked
    score = ranked[0]["score"]
    assert 0.0 <= score <= 1.0
