import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.orchestration.dynamics_weaver import run_dynamics_weaver


@pytest.fixture
def tmp_cfg(tmp_path):
    db_path = tmp_path / "sym.db"
    cfg = {"db_path": str(db_path)}
    return cfg


def test_dynamics_weaver_generates_artifacts(tmp_cfg):
    goal = "Simulate rogue drift in guarded swarm"
    result = run_dynamics_weaver(goal, tmp_cfg)

    assert result.goal == goal
    assert 0.0 <= result.risk_score <= 1.0
    assert Path(result.artifact_path).exists()

    payload = json.loads(Path(result.artifact_path).read_text())
    assert "sd_results" in payload
    assert "abm_results" in payload
    assert payload["sd_results"]["trajectory"]
    assert payload["abm_results"]["perturbed_trajectory"]


def test_dynamics_weaver_telemetry_written(tmp_cfg):
    goal = "Forecast latency surge with guard"
    run_dynamics_weaver(goal, tmp_cfg)

    db_path = Path(tmp_cfg["db_path"])
    assert db_path.exists()
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM sd_runs").fetchone()[0]
    assert count >= 1
