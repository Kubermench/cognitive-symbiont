from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from ..agents.registry import AgentRegistry
from ..agents.subself import SubSelf
from ..memory.db import MemoryDB
from ..tools import scriptify

logger = logging.getLogger(__name__)


@dataclass
class NodeSpec:
    name: str
    agent: str
    next: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    on_block: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "NodeSpec":
        if "agent" not in data:
            raise ValueError(f"Graph node '{name}' missing required 'agent' key")
        return cls(
            name=name,
            agent=data["agent"],
            next=data.get("next"),
            on_success=data.get("on_success"),
            on_failure=data.get("on_failure"),
            on_block=data.get("on_block"),
        )


@dataclass
class GraphSpec:
    start: str
    nodes: Dict[str, NodeSpec]
    crew_config: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "GraphSpec":
        data = yaml.safe_load(path.read_text()) or {}
        graph_data = data.get("graph", {})
        start = graph_data.get("start")
        if not start:
            raise ValueError("Graph spec missing 'graph.start'")
        nodes_section = graph_data.get("nodes", {})
        nodes = {name: NodeSpec.from_dict(name, spec) for name, spec in nodes_section.items()}
        crew_config = Path(data.get("crew_config", data.get("crews"))) if data else None
        if not crew_config:
            raise ValueError("Graph spec missing 'crew_config' pointing to crews YAML")
        crew_config = crew_config.expanduser().resolve()
        if not crew_config.exists():
            raise ValueError(f"Crew config not found at {crew_config}")
        return cls(start=start, nodes=nodes, crew_config=crew_config)


class GraphRunner:
    def __init__(
        self,
        spec: GraphSpec,
        registry: AgentRegistry,
        cfg: Dict[str, Any],
        db: MemoryDB,
        *,
        graph_path: Optional[Path] = None,
        state_dir: Path = Path("data/evolution"),
    ):
        self.spec = spec
        self.registry = registry
        self.cfg = cfg
        self.db = db
        self.graph_path = graph_path
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.graph_artifacts = Path("data/artifacts/graphs")
        self.graph_artifacts.mkdir(parents=True, exist_ok=True)

    def run(self, goal: str, resume_state: Optional[Path] = None) -> Path:
        if resume_state:
            state = json.loads(resume_state.read_text())
            current = state.get("current_node")
            history = state.get("history", [])
            goal = state.get("goal", goal)
            timestamp = state.get("timestamp", int(time.time()))
        else:
            current = self.spec.start
            history = []
            timestamp = int(time.time())
        state_path = self.state_dir / f"graph_state_{timestamp}.json"
        context = {"goal": goal, "episode_id": None, "cwd": Path.cwd()}
        self.db.ensure_schema()
        latest_bullets: Iterable[str] = []

        while current and current.upper() != "END":
            node = self.spec.nodes.get(current)
            if not node:
                logger.warning("Unknown node '%s' – stopping graph", current)
                break
            agent_spec = self.registry.get_agent(node.agent)
            llm_client = agent_spec.create_llm_client()
            role_dict = {"name": agent_spec.role}
            agent = SubSelf(role_dict, self.cfg, llm_client=llm_client)
            result = agent.run(context, self.db)
            history.append({
                "node": current,
                "agent": node.agent,
                "result": result,
            })
            self._save_state(state_path, goal, current, history, timestamp)

            outcome = self._classify_result(result)
            if agent_spec.role == "architect":
                latest_bullets = result.get("bullets", [])
            if agent_spec.role == "executor":
                script_path = self._write_script(latest_bullets, context)
                history[-1]["script"] = str(script_path)

            next_node = self._determine_next(node, outcome)
            if not next_node:
                break
            current = next_node

        artifact = self._persist_artifact(goal, history, timestamp)
        self._save_state(state_path, goal, current, history, timestamp, final=True)
        return artifact

    def _classify_result(self, result: Dict[str, Any]) -> str:
        verdict = result.get("verdict") if isinstance(result, dict) else None
        if verdict and verdict.lower() == "block":
            return "block"
        if result.get("error"):
            return "failure"
        return "success"

    def _determine_next(self, node: NodeSpec, outcome: str) -> Optional[str]:
        if outcome == "block" and node.on_block:
            return node.on_block
        if outcome == "failure" and node.on_failure:
            return node.on_failure
        if outcome == "success" and node.on_success:
            return node.on_success
        return node.next

    def _write_script(self, bullets: Iterable[str], context: Dict[str, Any]) -> Path:
        script_dir = Path(self.cfg.get("db_path", "./data/symbiont.db")).parent / "artifacts" / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            scriptify.write_script(
                list(bullets),
                base_dir=str(script_dir),
                db_path=self.cfg.get("db_path"),
                episode_id=context.get("episode_id"),
            )
        )

    def _persist_artifact(self, goal: str, history: list[dict[str, Any]], timestamp: int) -> Path:
        artifact_path = self.graph_artifacts / f"graph_{timestamp}.json"
        payload = {
            "goal": goal,
            "timestamp": timestamp,
            "history": history,
        }
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return artifact_path

    def _save_state(
        self,
        state_path: Path,
        goal: str,
        current_node: Optional[str],
        history: list[dict[str, Any]],
        timestamp: int,
        *,
        final: bool = False,
    ) -> None:
        data = {
            "goal": goal,
            "timestamp": timestamp,
            "current_node": current_node,
            "history": history,
            "graph_start": self.spec.start,
            "crew_config": str(self.spec.crew_config),
            "graph_path": str(self.graph_path) if self.graph_path else None,
            "final": final,
        }
        state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
