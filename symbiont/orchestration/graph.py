from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import yaml
import requests
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..tools.retry_utils import (
    RetryConfig,
    WEBHOOK_RETRY_CONFIG,
    retry_call,
    get_circuit_breaker,
)

from ..agents.registry import AgentRegistry
from ..agents.subself import SubSelf
from ..llm.budget import TokenBudget
from ..memory.db import MemoryDB
from ..memory import (
    coerce_backend_name,
    resolve_backend,
    MemoryBackendError,
)
from ..tools import scriptify, sd_engine
from .schema import GraphFileModel
from .systems import governance_snapshot

try:  # Optional LangGraph dependency
    from langgraph.graph import StateGraph  # type: ignore
except Exception:  # pragma: no cover - optional
    StateGraph = None  # type: ignore

logger = logging.getLogger(__name__)


def _is_url_allowed(url: str, allow_domains: list[str]) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    if not allow_domains:
        return True
    normalized = {domain.lower() for domain in allow_domains}
    return parsed.netloc.lower() in normalized


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
    simulation: Optional[Dict[str, Any]] = None
    parallel_groups: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "GraphSpec":
        raw = yaml.safe_load(path.read_text()) or {}
        model = GraphFileModel.model_validate(raw)
        section = model.graph
        start = section.start
        nodes = {name: NodeSpec.from_dict(name, node.model_dump()) for name, node in section.nodes.items()}
        crew_config_value = model.require_crew_path()
        crew_config = Path(crew_config_value).expanduser() if crew_config_value else None
        if not crew_config:
            raise ValueError("Graph spec missing 'crew_config' pointing to crews YAML")
        if not crew_config.is_absolute():
            crew_config = (path.parent / crew_config).resolve()
        else:
            crew_config = crew_config.resolve()
        if not crew_config.exists():
            raise ValueError(f"Crew config not found at {crew_config}")
        simulation = model.simulation or None
        parallel_groups = [list(group) for group in section.parallel]
        return cls(
            start=start,
            nodes=nodes,
            crew_config=crew_config,
            simulation=simulation,
            parallel_groups=parallel_groups,
        )

    def langgraph_blueprint(self) -> Dict[str, Any]:
        """Return a lightweight structure consumable by LangGraph."""
        edges: list[tuple[str, str]] = []
        for node in self.nodes.values():
            if node.next:
                edges.append((node.name, node.next))
            for branch in (node.on_success, node.on_failure, node.on_block):
                if branch:
                    edges.append((node.name, branch))
        return {
            "start": self.start,
            "nodes": {name: node.agent for name, node in self.nodes.items()},
            "edges": edges,
        }


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
        backend_name = coerce_backend_name(cfg)
        memory_cfg = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}
        try:
            backend = resolve_backend(backend_name, db, config=memory_cfg)
        except MemoryBackendError as exc:
            logger.warning("GraphRunner memory backend '%s' unavailable: %s; falling back to local.", backend_name, exc)
            backend = resolve_backend("local", db, config=memory_cfg)
        self.memory_backend = backend
        self.memory_backend_name = backend.name
        os.environ["SYMBIONT_MEMORY_LAYER"] = self.memory_backend_name
        self.graph_path = graph_path
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.graph_artifacts = Path("data/artifacts/graphs")
        self.graph_artifacts.mkdir(parents=True, exist_ok=True)
        self.simulation_dir = self.graph_artifacts / "simulations"
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_groups = spec.parallel_groups
        self.parallel_map: Dict[str, list[str]] = {}
        for group in self.parallel_groups:
            for node_name in group:
                self.parallel_map[node_name] = group
        self.parallel_progress: Dict[str, int] = {
            self._group_key(group): 0 for group in self.parallel_groups
        }
        self.current_handoff: Optional[Dict[str, Any]] = None
        self.token_budget: Optional[TokenBudget] = None

    def compile_langgraph(self):
        """Compile the current graph into a LangGraph StateGraph if available."""
        if not StateGraph:  # pragma: no cover - optional dependency
            raise RuntimeError("langgraph is not installed; run `pip install langgraph` to enable interop.")
        blueprint = self.spec.langgraph_blueprint()
        graph = StateGraph(dict)
        for node_name, agent_name in blueprint["nodes"].items():
            graph.add_node(node_name, lambda state, agent=agent_name: state | {"last_agent": agent})
        for src, dst in blueprint["edges"]:
            graph.add_edge(src, dst)
        graph.set_entry_point(blueprint["start"])
        return graph.compile()

    def run(self, goal: str, resume_state: Optional[Path] = None) -> Path | Dict[str, Any]:
        limit = 0
        if isinstance(self.cfg, dict):
            try:
                limit = int(self.cfg.get("max_tokens", 0) or 0)
            except (TypeError, ValueError):
                limit = 0
        data_root = Path(
            self.cfg.get("data_root")
            or Path(self.cfg.get("db_path", "./data/symbiont.db")).parent
        )
        budget_label = "graph"
        if self.graph_path:
            budget_label = f"graph:{self.graph_path.stem}"
        sink_path = data_root / "token_budget" / f"{budget_label}.json"
        history_path = sink_path.parent / "history.jsonl"
        self.token_budget = TokenBudget(
            limit=limit,
            label=budget_label,
            sink_path=sink_path,
            history_path=history_path,
        )

        if resume_state:
            state = json.loads(resume_state.read_text())
            current = state.get("current_node")
            history = state.get("history", [])
            goal = state.get("goal", goal)
            timestamp = state.get("timestamp", int(time.time()))
            state_path = resume_state
            saved_progress = state.get("parallel_progress") or {}
            for group in self.parallel_groups:
                key = self._group_key(group)
                if key in saved_progress:
                    self.parallel_progress[key] = int(saved_progress[key])
            self.current_handoff = state.get("handoff") or None
            self.token_budget.restore(state.get("token_budget"))
            self.token_budget.sink_path = sink_path
            self.token_budget.history_path = history_path
            if self.current_handoff and self.current_handoff.get("status") != "resolved":
                return {
                    "status": "handoff_pending",
                    "state": str(state_path),
                    "handoff": self.current_handoff,
                    "current_node": current,
                }
        else:
            current = self.spec.start
            history = []
            timestamp = int(time.time())
            state_path = self.state_dir / f"graph_state_{timestamp}.json"
        context = {
            "goal": goal,
            "episode_id": None,
            "cwd": Path.cwd(),
            "token_budget": self.token_budget,
        }
        self.db.ensure_schema()
        external_context = self._maybe_fetch_external(goal)
        if external_context:
            context["external_context"] = external_context
        latest_bullets: Iterable[str] = []
        pause_between = bool((self.cfg.get("ui") or {}).get("pause_between_nodes", False))

        if resume_state:
            # Rehydrate context from history so resumed runs behave deterministically.
            for entry in history:
                agent_name = entry.get("agent")
                if not agent_name:
                    continue
                try:
                    role = self.registry.get_agent(agent_name).role
                except KeyError:
                    role = agent_name
                result = entry.get("result") or {}
                if role == "dynamics_scout" and isinstance(result, dict):
                    blueprint = result.get("sd_blueprint")
                    if blueprint:
                        context["sd_blueprint"] = blueprint
                if role == "sd_modeler" and isinstance(result, dict):
                    context["sd_projection"] = result
                    context["sd_projection_summary"] = result.get("stats")
                if role == "architect" and isinstance(result, dict):
                    latest_bullets = result.get("bullets", [])
            last_node = state.get("last_node")
            if last_node and history:
                node_spec = self.spec.nodes.get(last_node)
                if node_spec:
                    outcome = self._classify_result(history[-1].get("result", {}))
                    current = self._determine_next(node_spec, outcome)
                    self._save_state(
                        state_path,
                        goal,
                        current,
                        history,
                        timestamp,
                        awaiting_human=False,
                        last_node=last_node,
                        handoff=self.current_handoff,
                    )
            if not current:
                return self._persist_artifact(goal, history, timestamp)
            if self.current_handoff and self.current_handoff.get("status") == "resolved":
                current, latest_bullets = self._finalize_handoff(
                    current,
                    history,
                    context,
                    latest_bullets,
                    goal=goal,
                    timestamp=timestamp,
                    state_path=state_path,
                )
        if not resume_state and self.spec.simulation:
            blueprint = self.spec.simulation.get("blueprint") or self._default_blueprint()
            context["sd_blueprint"] = blueprint
            try:
                baseline = self._execute_simulation(
                    blueprint,
                    goal=goal,
                    timestamp=timestamp,
                    label="baseline",
                    horizon=int(self.spec.simulation.get("horizon", 60)),
                    noise=float(self.spec.simulation.get("noise", 0.0)),
                )
                context["sd_projection"] = baseline
                context["sd_projection_summary"] = baseline.get("stats")
                history.append(
                    {
                        "node": "simulation_baseline",
                        "agent": "sd_engine",
                        "result": self._compact_simulation(baseline),
                    }
                )
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Baseline simulation failed: %s", exc)

        while current and current.upper() != "END":
            node = self.spec.nodes.get(current)
            if not node:
                logger.warning("Unknown node '%s' – stopping graph", current)
                break
            agent_spec = self.registry.get_agent(node.agent)
            llm_client = agent_spec.create_llm_client()
            role_dict = {"name": agent_spec.role}
            agent = SubSelf(
                role_dict,
                self.cfg,
                llm_client=llm_client,
                token_budget=self.token_budget,
            )
            result = agent.run(context, self.db)
            history.append(
                {
                    "node": current,
                    "agent": node.agent,
                    "result": result,
                }
            )
            handoff_response = self._maybe_begin_handoff(
                node,
                agent_spec,
                result,
                history,
                goal=goal,
                state_path=state_path,
                timestamp=timestamp,
            )
            if handoff_response:
                return handoff_response

            outcome, latest_bullets = self._process_agent_result(
                agent_spec,
                node,
                result,
                history,
                context,
                latest_bullets,
                goal=goal,
                timestamp=timestamp,
            )

            next_node = self._determine_next(node, outcome)
            next_node = self._resolve_parallel_next(node, outcome, next_node)

            if pause_between and node.agent != "sd_engine":
                self._save_state(
                    state_path,
                    goal,
                    next_node,
                    history,
                    timestamp,
                    awaiting_human=True,
                    last_node=node.name,
                    handoff=self.current_handoff,
                )
                return {
                    "status": "paused",
                    "state": str(state_path),
                    "last_node": node.name,
                    "next_node": next_node,
                }

            if not next_node:
                break
            current = next_node
            self._save_state(
                state_path,
                goal,
                current,
                history,
                timestamp,
                awaiting_human=False,
                last_node=node.name,
                handoff=self.current_handoff,
            )

        artifact = self._persist_artifact(goal, history, timestamp)
        self._save_state(
            state_path,
            goal,
            current,
            history,
            timestamp,
            final=True,
            awaiting_human=False,
            handoff=self.current_handoff,
        )
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
        evo_cfg = self.cfg.get("evolution") if isinstance(self.cfg, dict) else None
        snapshot = governance_snapshot(
            history,
            drift_rate=float((evo_cfg or {}).get("rogue_drift_rate", 0.05)),
            horizon=int((evo_cfg or {}).get("rogue_forecast_horizon", 50)),
            alert_threshold=float((evo_cfg or {}).get("rogue_alert_threshold", 0.6)),
        )
        if snapshot:
            payload["governance"] = snapshot
            if snapshot.get("alert"):
                logger.warning(
                    "Governance alert: rogue baseline %.2f exceeds threshold %.2f",
                    snapshot["rogue_baseline"],
                    snapshot["alert_threshold"],
                )
        if hasattr(self, "token_budget") and self.token_budget:
            payload["token_budget"] = self.token_budget.snapshot()
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return artifact_path

    def _execute_simulation(
        self,
        blueprint: Dict[str, Any],
        *,
        goal: str,
        timestamp: int,
        label: str,
        horizon: int,
        noise: float,
    ) -> Dict[str, Any]:
        spec = sd_engine.build_spec(blueprint)
        result = sd_engine.simulate(
            spec,
            horizon=horizon,
            noise=noise,
            artifacts_dir=self.simulation_dir,
        )
        plot_path = result.get("plot_path")
        if plot_path:
            try:
                src = Path(plot_path)
                if src.exists():
                    dest = self.simulation_dir / f"simulation_{label}_{timestamp}.png"
                    src.replace(dest)
                    result["plot_path"] = str(dest)
            except Exception:  # pragma: no cover - best effort
                pass
        payload = {
            "goal": goal,
            "label": label,
            "timestamp": timestamp,
            "blueprint": blueprint,
            "result": result,
        }
        sim_path = self.simulation_dir / f"simulation_{label}_{timestamp}.json"
        sim_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        add_run = getattr(self.db, "add_sd_run", None)
        if callable(add_run):
            try:
                add_run(
                    goal=goal,
                    label=label,
                    horizon=horizon,
                    timestep=result.get("timestep", 1.0),
                    stats=result.get("stats", {}),
                    plot_path=result.get("plot_path"),
                )
            except Exception:  # pragma: no cover
                logger.debug("Failed to persist sd_run telemetry", exc_info=True)
        return result

    def _compact_simulation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stats": result.get("stats"),
            "plot_path": result.get("plot_path"),
            "timestep": result.get("timestep"),
        }

    def _merge_blueprints(
        self,
        existing: Optional[Dict[str, Any]],
        incoming: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not existing:
            return incoming
        merged = dict(existing)
        for key in ("stocks", "flows", "auxiliaries"):
            if key in incoming and incoming[key]:
                merged[key] = incoming[key]
        if incoming.get("parameters"):
            params = dict(existing.get("parameters", {}))
            params.update(incoming["parameters"])
            merged["parameters"] = params
        if "timestep" in incoming:
            merged["timestep"] = incoming["timestep"]
        return merged

    def _default_blueprint(self) -> Dict[str, Any]:
        return {
            "timestep": 1.0,
            "stocks": [
                {"name": "autonomy", "initial": 0.5, "min": 0.0, "max": 1.0},
                {"name": "rogue", "initial": 0.2, "min": 0.0, "max": 1.0},
            ],
            "flows": [
                {"name": "autonomy_gain", "target": "autonomy", "expression": "0.1 * (1 - rogue)"},
                {"name": "rogue_decay", "target": "rogue", "expression": "-0.05 * autonomy"},
            ],
            "auxiliaries": [],
            "parameters": {},
        }

    def _maybe_fetch_external(self, goal: str) -> Dict[str, Any]:
        ext_cfg = (self.cfg.get("retrieval") or {}).get("external") or {}
        if not ext_cfg.get("enabled"):
            return {}
        max_items = int(ext_cfg.get("max_items", 6))
        min_relevance = float(ext_cfg.get("min_relevance", 0.7))
        log_enabled = bool(ext_cfg.get("log", True))
        try:
            result = self.memory_backend.fetch_external_context(
                goal,
                max_items=max_items,
                min_relevance=min_relevance,
            )
        except Exception as exc:
            if log_enabled:
                logger.warning("Graph external fetch skipped: %s", exc)
            return {}
        if log_enabled:
            accepted = len(result.get("accepted") or [])
            claims = len(result.get("claims") or [])
            logger.info(
                "Graph external context fetched: goal=%s accepted=%d claims=%d min_relevance=%.2f",
                goal,
                accepted,
                claims,
                min_relevance,
            )
        return result

    def _save_state(
        self,
        state_path: Path,
        goal: str,
        current_node: Optional[str],
        history: list[dict[str, Any]],
        timestamp: int,
        *,
        final: bool = False,
        awaiting_human: bool = False,
        last_node: Optional[str] = None,
        handoff: Optional[Dict[str, Any]] = None,
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
            "awaiting_human": awaiting_human,
            "last_node": last_node,
        }
        data["parallel_progress"] = {
            self._group_key(group): progress for group, progress in self._iter_group_progress()
        }
        if hasattr(self, "token_budget") and self.token_budget:
            data["token_budget"] = self.token_budget.snapshot()
        if handoff:
            data["handoff"] = handoff
        state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _maybe_begin_handoff(
        self,
        node: NodeSpec,
        agent_spec,
        result: Dict[str, Any],
        history: list[dict[str, Any]],
        *,
        goal: str,
        state_path: Path,
        timestamp: int,
    ) -> Optional[Dict[str, Any]]:
        handoff = result.get("handoff") if isinstance(result, dict) else None
        if not isinstance(handoff, dict):
            return None

        description = handoff.get("description") or handoff.get("title") or f"Handoff from {node.name}"
        assignee = handoff.get("assignee") or agent_spec.role
        episode_id = handoff.get("episode_id") or (
            self.cfg.get("episode_id") if isinstance(self.cfg, dict) else None
        )
        task_id: Optional[int] = None
        try:
            task_id = self.db.add_task(
                episode_id=episode_id,
                description=description,
                status="pending",
                assignee_role=str(assignee),
            )
        except Exception:
            logger.debug("Failed to persist handoff task", exc_info=True)

        self.current_handoff = {
            "node": node.name,
            "agent": node.agent,
            "status": "pending",
            "created_at": int(time.time()),
            "description": description,
            "assignee": assignee,
            "task_id": task_id,
            "payload": handoff,
        }
        result.setdefault("handoff", {})
        result["handoff"].update({"task_id": task_id, "status": "pending"})
        history[-1]["handoff"] = self.current_handoff
        self._notify_handoff(self.current_handoff)
        self._save_state(
            state_path,
            goal,
            node.name,
            history,
            timestamp,
            awaiting_human=True,
            last_node=node.name,
            handoff=self.current_handoff,
        )
        return {
            "status": "handoff_pending",
            "state": str(state_path),
            "handoff": self.current_handoff,
            "last_node": node.name,
        }

    def _process_agent_result(
        self,
        agent_spec,
        node: NodeSpec,
        result: Dict[str, Any],
        history: list[dict[str, Any]],
        context: Dict[str, Any],
        latest_bullets: Iterable[str],
        *,
        goal: str,
        timestamp: int,
    ) -> tuple[str, Iterable[str]]:
        if agent_spec.role == "dynamics_scout" and isinstance(result, dict):
            blueprint = result.get("sd_blueprint")
            if blueprint:
                combined = self._merge_blueprints(context.get("sd_blueprint"), blueprint)
                context["sd_blueprint"] = combined
        if agent_spec.role == "sd_modeler" and isinstance(result, dict):
            blueprint = context.get("sd_blueprint") or self._default_blueprint()
            horizon = int(
                result.get(
                    "horizon",
                    self.spec.simulation.get("horizon", 60) if self.spec.simulation else 60,
                )
            )
            noise = float(
                result.get(
                    "noise",
                    self.spec.simulation.get("noise", 0.0) if self.spec.simulation else 0.0,
                )
            )
            try:
                projection = self._execute_simulation(
                    blueprint,
                    goal=goal,
                    timestamp=timestamp,
                    label=node.name,
                    horizon=horizon,
                    noise=noise,
                )
                context["sd_projection"] = projection
                context["sd_projection_summary"] = projection.get("stats")
                history[-1]["simulation"] = self._compact_simulation(projection)
            except Exception as exc:  # pragma: no cover - safety
                history[-1]["simulation_error"] = str(exc)

        outcome = self._classify_result(result)
        if agent_spec.role == "architect" and isinstance(result, dict):
            latest_bullets = result.get("bullets", [])
        if agent_spec.role == "executor" and isinstance(result, dict):
            script_path = self._write_script(latest_bullets, context)
            history[-1]["script"] = str(script_path)
        return outcome, latest_bullets

    def _finalize_handoff(
        self,
        current: Optional[str],
        history: list[dict[str, Any]],
        context: Dict[str, Any],
        latest_bullets: Iterable[str],
        *,
        goal: str,
        timestamp: int,
        state_path: Path,
    ) -> tuple[Optional[str], Iterable[str]]:
        handoff = self.current_handoff or {}
        node_name = handoff.get("node")
        if not node_name:
            return current, latest_bullets
        node_spec = self.spec.nodes.get(node_name)
        if not node_spec:
            logger.warning("Handoff references unknown node '%s'", node_name)
            self.current_handoff = None
            return current, latest_bullets

        resolution = handoff.get("result") or {}
        outcome = handoff.get("outcome")
        if not outcome:
            outcome = self._classify_result(resolution)
            handoff["outcome"] = outcome

        for entry in reversed(history):
            if entry.get("node") == node_name:
                entry["result"] = resolution
                entry.setdefault("handoff_resolution", {}).update(
                    {
                        "outcome": outcome,
                        "resolved_at": handoff.get("resolved_at"),
                        "note": handoff.get("note"),
                    }
                )
                break

        agent_spec = self.registry.get_agent(node_spec.agent)
        _, latest_bullets = self._process_agent_result(
            agent_spec,
            node_spec,
            resolution,
            history,
            context,
            latest_bullets,
            goal=goal,
            timestamp=timestamp,
        )

        next_node = self._determine_next(node_spec, outcome)
        next_node = self._resolve_parallel_next(node_spec, outcome, next_node)

        task_id = handoff.get("task_id")
        if task_id:
            try:
                self.db.update_task_status(
                    task_id,
                    status=str(outcome),
                    result=json.dumps(resolution),
                )
            except Exception:
                logger.debug("Failed to update handoff task status", exc_info=True)

        self.current_handoff = None
        self._save_state(
            state_path,
            goal,
            next_node,
            history,
            timestamp,
            awaiting_human=False,
            last_node=node_name,
            handoff=None,
        )
        return next_node, latest_bullets

    def _notify_handoff(self, handoff: Dict[str, Any]) -> None:
        notif_cfg = (self.cfg.get("notifications") or {}) if isinstance(self.cfg, dict) else {}
        url = notif_cfg.get("handoff_webhook_url")
        if not url:
            return

        allow_domains = notif_cfg.get("allow_domains") or []
        if not _is_url_allowed(url, allow_domains):
            logger.warning("Skipping handoff webhook: URL '%s' not allowlisted", url)
            return

        for key in ("slack_webhook_url", "pagerduty_webhook_url"):
            extra_url = notif_cfg.get(key)
            if extra_url and not _is_url_allowed(extra_url, allow_domains):
                logger.warning("Skipping %s: '%s' not allowlisted", key, extra_url)

        # Enhanced retry configuration for webhooks
        retry_cfg = notif_cfg.get("retry", {}) or {}
        webhook_retry_config = RetryConfig(
            attempts=max(1, int(retry_cfg.get("attempts", 3) or 3)),
            initial_wait=max(0.5, float(retry_cfg.get("initial_wait", 2.0) or 2.0)),
            max_wait=max(5.0, float(retry_cfg.get("max_wait", 60.0) or 60.0)),
            multiplier=max(1.0, float(retry_cfg.get("multiplier", 2.0) or 2.0)),
            jitter=bool(retry_cfg.get("jitter", True)),
            failure_threshold=max(1, int(retry_cfg.get("failure_threshold", 3) or 3)),
            recovery_timeout=max(60.0, float(retry_cfg.get("recovery_timeout", 300.0) or 300.0)),
        )
        timeout = float(notif_cfg.get("timeout_seconds", 10.0) or 10.0)
        circuit_breaker_name = f"webhook_{url.split('/')[-2] if '/' in url else 'default'}"

        if isinstance(self.cfg, dict):
            data_root = Path(
                self.cfg.get("data_root")
                or Path(self.cfg.get("db_path", "./data/symbiont.db")).parent
            )
        else:
            data_root = Path("data")

        log_path_cfg = notif_cfg.get("log_path")
        log_path = Path(log_path_cfg).expanduser() if log_path_cfg else data_root / "logs" / "handoff_notifications.jsonl"

        payload = {
            "goal": handoff.get("description"),
            "node": handoff.get("node"),
            "agent": handoff.get("agent"),
            "assignee": handoff.get("assignee"),
            "state": handoff.get("status"),
            "created_at": handoff.get("created_at"),
            "task_id": handoff.get("task_id"),
            "timestamp": int(time.time()),
        }

        def _write_log(status: str, message: Optional[str] = None) -> None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                entry = dict(payload)
                entry["status"] = status
                if message:
                    entry["message"] = message
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
            except Exception:
                logger.debug("Failed writing handoff notification log", exc_info=True)

        def sender() -> None:
            def send_webhook():
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                return response
            
            try:
                retry_call(
                    send_webhook,
                    config=webhook_retry_config,
                    circuit_breaker=circuit_breaker_name,
                )
                _write_log("success")
                logger.info("Handoff webhook sent successfully to %s", url)
            except Exception as exc:
                _write_log("failure", str(exc))
                logger.error(
                    "Handoff webhook failed after %d attempts: %s",
                    webhook_retry_config.attempts,
                    exc,
                )
                
                # Get circuit breaker metrics for monitoring
                cb_metrics = get_circuit_breaker(circuit_breaker_name, webhook_retry_config).get_metrics()
                if cb_metrics["state"] == "open":
                    logger.warning(
                        "Webhook circuit breaker '%s' is open (success_rate: %.2f%%)",
                        circuit_breaker_name,
                        cb_metrics["success_rate"] * 100,
                    )

        thread = threading.Thread(target=sender, name="handoff_webhook", daemon=True)
        thread.start()

    def get_webhook_circuit_breaker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all webhook circuit breakers."""
        from ..tools.retry_utils import list_circuit_breakers
        
        webhook_metrics = {}
        for name, metrics in list_circuit_breakers().items():
            if name.startswith("webhook_"):
                webhook_metrics[name] = metrics
        return webhook_metrics

    def _resolve_parallel_next(
        self,
        node: NodeSpec,
        outcome: str,
        candidate: Optional[str],
    ) -> Optional[str]:
        group = self.parallel_map.get(node.name)
        if not group:
            return candidate
        key = self._group_key(group)
        try:
            idx = group.index(node.name)
        except ValueError:
            return candidate

        # Guard explicit routing first: honor explicit targets outside the group.
        if outcome in {"block", "failure"}:
            self.parallel_progress[key] = 0
            if candidate and candidate not in group:
                return candidate
            return None

        # For success, respect explicit branch that leaves the group.
        if candidate and candidate not in group:
            self.parallel_progress[key] = 0
            return candidate

        next_idx = idx + 1
        if next_idx < len(group):
            self.parallel_progress[key] = next_idx
            return group[next_idx]

        # Completed the group; fall back to candidate if it escapes, else None.
        self.parallel_progress[key] = 0
        if candidate and candidate not in group:
            return candidate
        return candidate if candidate else None

    def _group_key(self, group: list[str]) -> str:
        return "|".join(group)

    def _iter_group_progress(self) -> Iterable[tuple[list[str], int]]:
        for group in self.parallel_groups:
            yield group, self.parallel_progress.get(self._group_key(group), 0)
