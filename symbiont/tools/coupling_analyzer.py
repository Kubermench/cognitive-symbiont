from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    nx = None

def _build_graph(spec: GraphSpec) -> "nx.DiGraph":
    if nx is None:
        raise RuntimeError("networkx is required for coupling analysis; install via `pip install networkx`.")
    graph = nx.DiGraph()
    for name in spec.nodes:
        graph.add_node(name)
    for name, node in spec.nodes.items():
        targets = [node.next, node.on_success, node.on_failure, node.on_block]
        for target in targets:
            if target and target in spec.nodes:
                graph.add_edge(name, target)
    return graph


def analyze(graph_path: Path) -> Dict[str, any]:
    from symbiont.orchestration.graph import GraphSpec

    spec = GraphSpec.from_yaml(graph_path)
    graph = _build_graph(spec)
    entries: List[Tuple[str, str, float]] = []
    total_heat = 0.0
    for u, v in graph.edges():
        fan_out = graph.out_degree(u)
        fan_in = graph.in_degree(v)
        cycles = 1 if nx.has_path(graph, v, u) else 0
        score = min(3.0, 0.5 * fan_out + 0.5 * fan_in + cycles)
        entries.append((u, v, score))
        total_heat += score
    return {
        "entries": entries,
        "heat": total_heat,
        "nodes": list(graph.nodes()),
        "edges": list(graph.edges()),
    }
