import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.orchestration.graph import GraphSpec


label_strategy = st.text(min_size=1, max_size=6, alphabet=st.characters(min_codepoint=97, max_codepoint=122))


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(nodes=st.lists(label_strategy, min_size=1, max_size=5, unique=True))
def test_graphspec_loads_generated_graph(tmp_path, nodes):
    crews_yaml = tmp_path / "crews.yaml"
    crews_yaml.write_text("agents: {}\n", encoding="utf-8")
    graph_yaml = tmp_path / "graph.yaml"
    graph_config = {
        "crew_config": str(crews_yaml),
        "graph": {
            "start": nodes[0],
            "nodes": {name: {"agent": f"{name}_agent"} for name in nodes},
        },
    }
    graph_yaml.write_text(yaml.safe_dump(graph_config), encoding="utf-8")
    spec = GraphSpec.from_yaml(graph_yaml)
    assert spec.start == nodes[0]
    assert set(spec.nodes.keys()) == set(nodes)


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    nodes=st.lists(label_strategy, min_size=1, max_size=5, unique=True),
    missing=st.sampled_from(["crew", "start", "agent"]),
)
def test_graphspec_missing_fields_raise(tmp_path, nodes, missing):
    crews_yaml = tmp_path / "crews.yaml"
    crews_yaml.write_text("agents: {}\n", encoding="utf-8")
    graph_yaml = tmp_path / "graph.yaml"
    graph_config = {
        "crew_config": str(crews_yaml),
        "graph": {
            "start": nodes[0],
            "nodes": {name: {"agent": f"{name}_agent"} for name in nodes},
        },
    }
    if missing == "crew":
        graph_config.pop("crew_config")
    elif missing == "start":
        graph_config["graph"].pop("start")
    else:
        target = nodes[0]
        graph_config["graph"]["nodes"][target].pop("agent")

    graph_yaml.write_text(yaml.safe_dump(graph_config), encoding="utf-8")
    with pytest.raises(ValueError):
        GraphSpec.from_yaml(graph_yaml)
