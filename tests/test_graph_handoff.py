import json
from pathlib import Path
import sys

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from symbiont.orchestration.graph import GraphSpec, GraphRunner
from symbiont.agents.registry import AgentRegistry
from symbiont.memory.db import MemoryDB
from symbiont.cli import app as cli_app


@pytest.fixture()
def handoff_environment(tmp_path, monkeypatch):
    crews_yaml = tmp_path / "crews.yaml"
    crews_yaml.write_text(
        """
agents:
  handoff_agent:
    role: handoff_agent
    llm: {}
    cache: in_memory
    tools: []
  finisher_agent:
    role: finisher
    llm: {}
    cache: in_memory
    tools: []
crew:
  default:
    sequence:
      - handoff_agent
      - finisher_agent
""",
        encoding="utf-8",
    )

    graph_yaml = tmp_path / "graph.yaml"
    graph_yaml.write_text(
        """
crew_config: crews.yaml

graph:
  start: kickoff
  nodes:
    kickoff:
      agent: handoff_agent
      next: finish
    finish:
      agent: finisher_agent
      next: END
""",
        encoding="utf-8",
    )

    spec = GraphSpec.from_yaml(graph_yaml)
    registry = AgentRegistry.from_yaml(crews_yaml)
    cfg = {"db_path": str(tmp_path / "sym.db")}
    db = MemoryDB(cfg["db_path"])
    db.ensure_schema()
    runner = GraphRunner(
        spec,
        registry,
        cfg,
        db,
        graph_path=graph_yaml,
        state_dir=tmp_path / "state",
    )
    runner.graph_artifacts = tmp_path / "graphs"
    runner.graph_artifacts.mkdir(parents=True, exist_ok=True)
    runner.simulation_dir = runner.graph_artifacts / "simulations"
    runner.simulation_dir.mkdir(parents=True, exist_ok=True)

    calls: list[str] = []

    def fake_run(self, context, memory):
        calls.append(self.name)
        if self.name == "handoff_agent":
            return {
                "handoff": {
                    "description": "Need manual validation",
                    "assignee": "human",
                }
            }
        return {"verdict": "ok"}

    monkeypatch.setattr("symbiont.agents.subself.SubSelf.run", fake_run)

    return {
        "runner": runner,
        "registry": registry,
        "cfg": cfg,
        "db": db,
        "calls": calls,
        "graph_yaml": graph_yaml,
        "tmp_path": tmp_path,
    }


def test_graph_runner_emits_handoff(handoff_environment):
    runner = handoff_environment["runner"]
    cfg = handoff_environment["cfg"]

    result = runner.run("Async goal")
    assert isinstance(result, dict)
    assert result["status"] == "handoff_pending"
    state_path = Path(result["state"])
    assert state_path.exists()

    state_data = json.loads(state_path.read_text())
    handoff = state_data.get("handoff")
    assert handoff and handoff["status"] == "pending"
    assert handoff.get("description") == "Need manual validation"
    assert state_data.get("awaiting_human") is True

    with MemoryDB(cfg["db_path"])._conn() as conn:
        row = conn.execute("SELECT status, assignee_role FROM tasks ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row[0] == "pending"
    assert row[1] == "human"


def test_graph_handoff_resolution_and_resume(handoff_environment, monkeypatch):
    runner = handoff_environment["runner"]
    cfg = handoff_environment["cfg"]
    db = handoff_environment["db"]
    tmp_path = handoff_environment["tmp_path"]
    calls = handoff_environment["calls"]

    pending = runner.run("Async goal")
    state_path = Path(pending["state"])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    cli = CliRunner()
    result = cli.invoke(
        cli_app,
        [
            "graph-handoff-complete",
            str(state_path),
            "--outcome",
            "success",
            "--result",
            json.dumps({"verdict": "ok"}),
            "--config-path",
            str(cfg_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    state_data = json.loads(state_path.read_text())
    handoff = state_data.get("handoff")
    assert handoff and handoff["status"] == "resolved"
    assert handoff["outcome"] == "success"
    assert state_data.get("awaiting_human") is False

    with db._conn() as conn:
        row = conn.execute("SELECT status, result FROM tasks ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row[0] == "success"
    assert json.loads(row[1]) == {"verdict": "ok"}

    resumed = runner.run("Async goal", resume_state=state_path)
    assert isinstance(resumed, Path)
    assert resumed.exists()

    assert calls.count("handoff_agent") == 1
    assert "finisher" in calls
