import json
from pathlib import Path

import pytest

from symbiont.tools import systems_os


def test_write_coupling_map(tmp_path: Path) -> None:
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    original_root = systems_os.SYSTEMS_ROOT
    systems_os.SYSTEMS_ROOT = systems_dir  # type: ignore
    try:
        systems_os.write_coupling_map([("a", "b", 1.5)])
        csv_path = systems_dir / "CouplingMap.csv"
        content = csv_path.read_text().strip().splitlines()
        assert content[0] == "component_a,component_b,coupling_score"
        assert content[1] == "a,b,1.500"
    finally:
        systems_os.SYSTEMS_ROOT = original_root  # type: ignore


def test_update_flow_metrics(tmp_path: Path) -> None:
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    original_root = systems_os.SYSTEMS_ROOT
    systems_os.SYSTEMS_ROOT = systems_dir  # type: ignore
    payload = {
        "lead_time_hours": 0.5,
        "deploy_frequency_per_day": 2.0,
        "mttr_minutes": 30,
        "change_fail_rate": 0.1,
    }
    try:
        systems_os.update_flow_metrics(payload)
        stored = json.loads((systems_dir / "FlowMetrics.json").read_text())
        assert stored["lead_time_hours"] == 0.5
    finally:
        systems_os.SYSTEMS_ROOT = original_root  # type: ignore


def test_coupling_analyzer(tmp_path: Path) -> None:
    graph_yaml = tmp_path / "graph.yaml"
    crew_path = Path("configs/crews.yaml").resolve()
    graph_yaml.write_text(
        f"""
crew_config: {crew_path}
graph:
  start: scout
  nodes:
    scout:
      agent: scout
      next: architect
    architect:
      agent: architect
      on_success: critic
    critic:
      agent: critic
      on_success: null
""",
        encoding="utf-8",
    )
    from symbiont.tools import coupling_analyzer

    result = coupling_analyzer.analyze(graph_yaml)
    assert result["entries"], "Coupling analysis should produce entries"
