import importlib.util
from pathlib import Path

import pytest

from symbiont.orchestration.graph import GraphRunner, GraphSpec
from symbiont.agents.registry import AgentRegistry
from symbiont.memory.db import MemoryDB


langgraph_spec = importlib.util.find_spec("langgraph")
requires_langgraph = pytest.mark.skipif(
    langgraph_spec is None, reason="langgraph not installed"
)


@requires_langgraph
def test_graph_runner_langgraph_compile(tmp_path):
    graph_yaml = tmp_path / "foresight_graph.yaml"
    crew_config = Path("configs/crews/foresight_weaver.yaml").resolve()
    graph_yaml.write_text(
        """graph:
  start: scout
  crew_config: "{crew}"
  nodes:
    scout:
      agent: foresight_scout_agent
      next: analyze
    analyze:
      agent: foresight_analyzer_agent
""".format(crew=crew_config),
        encoding="utf-8",
    )

    spec = GraphSpec.from_yaml(graph_yaml)
    registry = AgentRegistry.from_yaml(crew_config)
    db = MemoryDB(db_path=":memory:")
    runner = GraphRunner(spec, registry, cfg={"db_path": ":memory:"}, db=db)

    compiled = runner.compile_langgraph()
    assert compiled is not None
