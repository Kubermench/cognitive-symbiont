import asyncio
from pathlib import Path
from typing import Any, Dict

import pytest

from symbiont.initiative import daemon


class DummyRunner:
    def __init__(self, registry, cfg, db):  # noqa: D401 - signature matches target
        self.last_context: Dict[str, Any] = {
            "foresight_sources": {"items": [], "meta": {}},
        }

    def run(self, crew_name: str, goal: str) -> Path:
        artifact = Path("data/artifacts/crews/foresight/test_artifact.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"goal": "%s"}' % goal, encoding="utf-8")
        return artifact


@pytest.fixture
def cfg(tmp_path):
    return {
        "db_path": str(tmp_path / "symbiont.db"),
        "foresight": {"enabled": True},
    }


def test_run_foresight_records_state(monkeypatch, tmp_path, cfg):
    from symbiont.foresight import analyzer as analyzer_mod

    async def fake_run_hunt_async(llm, goal, config=None):
        artifact = tmp_path / "foresight_hunt.json"
        artifact.write_text('{"goal": "%s"}' % goal, encoding="utf-8")
        payload = {
            "topic": goal,
            "items": [{"title": "Signal", "summary": "Insight", "source": "arxiv"}],
            "meta": {"contributors": ["arxiv"], "avg_score": 0.8},
        }
        return payload, artifact

    monkeypatch.setattr(daemon, "run_hunt_async", fake_run_hunt_async)
    monkeypatch.setattr(daemon, "CrewRunner", DummyRunner)
    state_path = tmp_path / "daemon_state.json"
    monkeypatch.setattr(daemon, "STATE_PATH", str(state_path))
    monkeypatch.setattr(daemon, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(analyzer_mod, "DIFF_DIR", tmp_path / "diffs")
    monkeypatch.setattr(analyzer_mod, "SNAPSHOT_DIR", tmp_path / "snapshots")

    crew_config = str(Path("configs/crews/foresight_weaver.yaml").resolve())
    foresight_cfg = {
        "enabled": True,
        "crew": "foresight_weaver",
        "goal": "Edge-safe foresight",
        "crew_config": crew_config,
    }
    state = {
        "foresight_last_run_ts": 0,
        "foresight_last_goal": "",
        "foresight_last_artifact": "",
        "last_proposal_ts": 0,
    }
    reasons = ["test.trigger"]

    result = daemon._run_foresight(cfg, foresight_cfg, state, now=1234567890, reasons=reasons)

    assert state["foresight_last_goal"] == "Edge-safe foresight"
    assert Path(state["foresight_last_artifact"]).exists()
    assert Path(state["foresight_last_hunt_artifact"]).exists()
    assert state["foresight_last_reflection"]
    assert result["goal"] == "Edge-safe foresight"
