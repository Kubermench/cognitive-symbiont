import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.orchestration.graph import GraphRunner, GraphSpec, NodeSpec
from symbiont.orchestration.systems import forecast_rogue_drift, governance_snapshot
from symbiont.cli import app as cli_app
from symbiont.memory.db import MemoryDB


def test_forecast_rogue_drift_progresses_toward_one():
    history = [{"result": {"rogue_score": 0.2}}]
    forecast = forecast_rogue_drift(history, drift_rate=0.1, horizon=3)
    assert forecast == [0.28, 0.352, 0.4168]


def test_governance_snapshot_blank_when_no_scores():
    snapshot = governance_snapshot([{"result": {"note": "no risk"}}])
    assert snapshot == {}


def test_persist_artifact_includes_governance(tmp_path):
    spec = GraphSpec(start="scout", nodes={}, crew_config=tmp_path)
    runner = GraphRunner(
        spec,
        registry=object(),
        cfg={
            "db_path": str(tmp_path / "db.sqlite"),
            "evolution": {
                "rogue_drift_rate": 0.1,
                "rogue_forecast_horizon": 2,
                "rogue_alert_threshold": 0.25,
            },
        },
        db=object(),
        state_dir=tmp_path / "state",
    )
    runner.graph_artifacts = tmp_path / "graphs"
    runner.graph_artifacts.mkdir(parents=True, exist_ok=True)

    history = [
        {"node": "critic", "result": {"rogue_score": 0.3}},
    ]
    path = runner._persist_artifact("goal", history, timestamp=123)

    data = json.loads(path.read_text())
    assert "governance" in data
    governance = data["governance"]
    assert governance["rogue_baseline"] == 0.3
    assert len(governance["rogue_forecast"]) == 2
    assert governance["alert"] is True
    assert governance["alert_threshold"] == 0.25


def test_graph_runner_runs_simulation_for_sd_modeler(tmp_path):
    crews = tmp_path / "crews.yaml"
    crews.write_text("agents: {}\n", encoding="utf-8")

    spec = GraphSpec(
        start="sd",
        nodes={
            "sd": NodeSpec(name="sd", agent="sd_modeler", next="END"),
        },
        crew_config=crews,
        simulation={
            "horizon": 5,
            "noise": 0.0,
            "blueprint": {
                "timestep": 1.0,
                "stocks": [
                    {"name": "autonomy", "initial": 0.4, "min": 0.0, "max": 1.0},
                    {"name": "rogue", "initial": 0.2, "min": 0.0, "max": 1.0},
                ],
                "flows": [
                    {"name": "autonomy_gain", "target": "autonomy", "expression": "0.1 * (1 - rogue)"},
                    {"name": "rogue_decay", "target": "rogue", "expression": "-0.05 * autonomy"},
                ],
                "auxiliaries": [],
                "parameters": {},
            },
        },
    )

    class DummyAgentSpec:
        def __init__(self, role: str):
            self.role = role

        def create_llm_client(self):
            class DummyLLM:
                def generate(self, prompt: str) -> str:
                    return ""

            return DummyLLM()

    class DummyRegistry:
        def __init__(self):
            self._agents = {"sd_modeler": DummyAgentSpec("sd_modeler")}

        def get_agent(self, name: str):
            return self._agents[name]

    class DummyDB:
        def __init__(self):
            self.runs: list[dict[str, Any]] = []

        def ensure_schema(self):
            return None

        def add_sd_run(self, **kwargs):
            self.runs.append(kwargs)

    db = DummyDB()
    runner = GraphRunner(
        spec,
        registry=DummyRegistry(),
        cfg={"db_path": str(tmp_path / "db.sqlite")},
        db=db,
        state_dir=tmp_path / "state",
    )

    artifact = runner.run("Forecast autonomy drift")
    data = json.loads(artifact.read_text())
    nodes = [entry["node"] for entry in data["history"]]
    assert nodes[0] == "simulation_baseline"
    assert any(entry.get("simulation") for entry in data["history"] if entry["node"] == "sd")
    assert db.runs, "sd_runs telemetry should be recorded"


def test_sd_runs_cli_lists_recent_entries(tmp_path):
    db_path = tmp_path / "db.sqlite"
    cfg_path = tmp_path / "config.yaml"

    memory = MemoryDB(db_path=str(db_path))
    memory.ensure_schema()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sd_runs (goal, label, horizon, timestep, stats_json, plot_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "my-goal",
                "baseline",
                60,
                1.0,
                json.dumps({"rogue": {"last": 0.82}, "autonomy": {"last": 0.95}}),
                None,
                0,
            ),
        )

    cfg_path.write_text(f"db_path: {db_path}\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli_app, ["sd-runs", "--config-path", str(cfg_path), "--limit", "1"])

    assert result.exit_code == 0
    assert "my-goal" in result.stdout
    assert "rogue=0.82" in result.stdout
