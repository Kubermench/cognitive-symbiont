import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.orchestration.graph import GraphSpec
from symbiont.agents.registry import AgentRegistry


def test_graph_validation_requires_agent(tmp_path):
    graph_yaml = tmp_path / "graph.yaml"
    graph_yaml.write_text(
        """
crew_config: ./crew_template.yaml
graph:
  start: node_a
  nodes:
    node_a: {}
""",
        encoding="utf-8",
    )
    with pytest.raises((ValueError, ValidationError)):
        GraphSpec.from_yaml(graph_yaml)


def test_crew_validation_requires_list(tmp_path):
    crew_yaml = tmp_path / "crew.yaml"
    crew_yaml.write_text(
        """
agents:
  scout:
    role: scout
crew:
  default:
    sequence: architect
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        AgentRegistry.from_yaml(crew_yaml)
