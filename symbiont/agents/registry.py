from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .subself import SubSelf
from ..llm.client import LLMClient
from ..llm.budget import TokenBudget
from ..memory.db import MemoryDB
from ..tools import scriptify, systems_os
from ..orchestration.schema import CrewFileModel

logger = logging.getLogger(__name__)


class CacheBackend:
    def __init__(self, name: str):
        self.name = name

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    def __init__(self):
        super().__init__("in_memory")
        self._store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value


class RedisCache(CacheBackend):
    def __init__(self, url: str):
        super().__init__(f"redis:{url}")
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "redis package not installed. Install with `pip install redis`."
            ) from exc
        self._client = redis.from_url(url)

    def get(self, key: str) -> Optional[str]:
        value = self._client.get(key)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def set(self, key: str, value: str) -> None:
        self._client.set(key, value)


class CachedLLMClient:
    def __init__(self, agent_id: str, llm_cfg: Dict[str, Any], cache_cfg: Optional[str]):
        self.agent_id = agent_id
        self.client = LLMClient(llm_cfg)
        self.cache = self._init_cache(cache_cfg)

    def _init_cache(self, cache_cfg: Optional[str]) -> CacheBackend:
        if not cache_cfg:
            return InMemoryCache()
        if cache_cfg == "in_memory":
            return InMemoryCache()
        if cache_cfg.startswith("redis://"):
            try:
                return RedisCache(cache_cfg)
            except RuntimeError as exc:
                logger.warning("Redis cache unavailable (%s); falling back to memory", exc)
                return InMemoryCache()
        logger.warning("Unknown cache backend %s; using in-memory", cache_cfg)
        return InMemoryCache()

    def generate(self, prompt: str, **kwargs) -> str:
        key_src = f"{self.agent_id}:{prompt}".encode("utf-8")
        cache_key = hashlib.sha256(key_src).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        response = self.client.generate(prompt, **kwargs)
        if response:
            self.cache.set(cache_key, response)
        return response


@dataclass
class AgentSpec:
    name: str
    role: str
    llm: Dict[str, Any]
    cache: Optional[str]
    tools: Iterable[str]

    def create_llm_client(self) -> CachedLLMClient:
        return CachedLLMClient(self.name, self.llm or {}, self.cache)


@dataclass
class CrewSpec:
    name: str
    sequence: Iterable[str]


class AgentRegistry:
    def __init__(self, agents: Dict[str, AgentSpec], crews: Dict[str, CrewSpec]):
        self.agents = agents
        self.crews = crews

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentRegistry":
        raw = yaml.safe_load(path.read_text()) or {}
        model = CrewFileModel.model_validate(raw)
        agents_data = model.agents
        crews_data = model.resolved_crews()
        agents: Dict[str, AgentSpec] = {}
        for name, spec in agents_data.items():
            agents[name] = AgentSpec(
                name=name,
                role=spec.role,
                llm=spec.llm,
                cache=spec.cache,
                tools=spec.tools,
            )
        crews: Dict[str, CrewSpec] = {}
        for name, spec in crews_data.items():
            crews[name] = CrewSpec(name=name, sequence=spec.resolved_sequence())
        return cls(agents, crews)

    def get_agent(self, name: str) -> AgentSpec:
        if name not in self.agents:
            raise KeyError(f"Unknown agent '{name}'")
        return self.agents[name]

    def get_crew(self, name: str) -> CrewSpec:
        if name not in self.crews:
            raise KeyError(f"Unknown crew '{name}'")
        return self.crews[name]


class CrewRunner:
    def __init__(self, registry: AgentRegistry, cfg: Dict[str, Any], db: MemoryDB):
        self.registry = registry
        self.cfg = cfg
        self.db = db
        self.script_dir = Path(self.cfg.get("db_path", "./data/symbiont.db")).parent / "artifacts" / "scripts"
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.crew_artifacts = Path("data/artifacts/crews")
        self.crew_artifacts.mkdir(parents=True, exist_ok=True)
        self.last_context: Dict[str, Any] = {}

    def run(self, crew_name: str, goal: str) -> Path:
        crew = self.registry.get_crew(crew_name)
        context = {"goal": goal, "episode_id": None, "cwd": Path.cwd()}
        memory = self.db
        memory.ensure_schema()
        memory_conn = memory
        outputs = []
        latest_bullets: list[str] = []

        limit = 0
        try:
            limit = int(self.cfg.get("max_tokens", 0) or 0)
        except (TypeError, ValueError):
            limit = 0
        data_root = Path(
            self.cfg.get("data_root")
            or Path(self.cfg.get("db_path", "./data/symbiont.db")).parent
        )
        sink_path = data_root / "token_budget" / f"crew_{crew_name}.json"
        budget = TokenBudget(limit=limit, label=f"crew:{crew_name}", sink_path=sink_path)
        context["token_budget"] = budget

        for agent_id in crew.sequence:
            agent_spec = self.registry.get_agent(agent_id)
            llm_client = agent_spec.create_llm_client()
            role_dict = {"name": agent_spec.role}
            agent = SubSelf(
                role_dict,
                self.cfg,
                llm_client=llm_client,
                token_budget=budget,
            )
            result = agent.run(context, memory_conn)
            outputs.append({"agent": agent_id, "result": result})
            context_key = f"result:{agent_spec.role}"
            context[context_key] = result
            if agent_spec.role == "architect":
                latest_bullets = result.get("bullets", [])
            if agent_spec.role == "executor":
                script_path = scriptify.write_script(
                    latest_bullets,
                    base_dir=str(self.script_dir),
                    db_path=self.cfg.get("db_path"),
                    episode_id=context.get("episode_id"),
                )
                outputs.append({"agent": agent_id, "script": script_path})
            if agent_spec.role == "loop_mapper":
                context["systems_loops"] = result.get("loops", [])
            if agent_spec.role == "leverage_ranker":
                context["leverage_points"] = result.get("leverage_points", [])
            if agent_spec.role == "cynefin_classifier":
                context["cynefin_domain"] = result.get("domain")
                context["cynefin_reason"] = result.get("reason")
                context["cynefin_signals"] = result.get("signals", [])
            if agent_spec.role == "cynefin_planner":
                context["cynefin_rule"] = result.get("rule")
                context["cynefin_actions"] = result.get("actions", [])
                context["cynefin_probes"] = result.get("probes", [])
            if agent_spec.role == "model_challenger":
                context["mental_model"] = {
                    "model": result.get("model"),
                    "counter_bet": result.get("counter_bet"),
                    "experiment": result.get("experiment"),
                    "signal": result.get("signal"),
                }
            if agent_spec.role == "success_miner":
                context["safety_entry"] = result
            if agent_spec.role == "coupling_analyzer":
                context["coupling_entries"] = result.get("entries", [])
                context["coupling_heat"] = result.get("heat", 0.0)

        artifact_path = self._persist_outputs(crew_name, goal, outputs)
        self.last_context = context
        self._post_process(crew_name, context)
        return artifact_path

    def _persist_outputs(self, crew_name: str, goal: str, outputs: list[dict[str, Any]]) -> Path:
        import time

        ts = int(time.time())
        crew_dir = self.crew_artifacts / crew_name
        crew_dir.mkdir(parents=True, exist_ok=True)
        path = crew_dir / f"crew_{ts}.json"
        payload = {
            "crew": crew_name,
            "goal": goal,
            "timestamp": ts,
            "outputs": outputs,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _post_process(self, crew_name: str, context: Dict[str, Any]) -> None:
        if crew_name == "leverage_scanner":
            loops = context.get("systems_loops", [])
            leverage = context.get("leverage_points", [])
            if loops:
                rows = []
                for loop in loops:
                    rows.append(
                        "| {date} | {name} | {type} | {stocks} | {flows} | {note} |".format(
                            date=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%d"),
                            name=loop.get("name", "Unnamed"),
                            type=loop.get("type", "unknown"),
                            stocks=", ".join(loop.get("stocks", []))[:60],
                            flows=", ".join(loop.get("flows", []))[:60],
                            note=loop.get("note", "")[:80],
                        )
                    )
                systems_os.append_markdown("Loops.md", rows)
            if leverage:
                rows = []
                for idx, point in enumerate(leverage, start=1):
                    rows.append(
                        "| {rank} | {name} | {effort:.2f} | {impact:.2f} | {note} |".format(
                            rank=idx,
                            name=point.get("name", "Leverage"),
                            effort=point.get("effort", 0.0),
                            impact=point.get("impact", 0.0),
                            note=point.get("description", "")[:80],
                        )
                    )
                systems_os.append_markdown("LeverageList.md", rows)
        if crew_name == "cynefin_router":
            domain = context.get("cynefin_domain", "disorder")
            rule = context.get("cynefin_rule", "Assess further")
            signals = context.get("cynefin_signals", [])
            actions = context.get("cynefin_actions", [])
            row = "| {domain} | {reason} | {signals} | {rule} |".format(
                domain=domain.capitalize(),
                reason=context.get("cynefin_reason", "")[:80],
                signals=", ".join(signals)[:80],
                rule=rule[:80],
            )
            systems_os.append_markdown("Cynefin.md", [row])
        if crew_name == "model_challenger":
            mm = context.get("mental_model") or {}
            row = "| {date} | {model} | {counter} | {experiment} | pending |".format(
                date=__import__("datetime").datetime.utcnow().strftime("%Y-%m-%d"),
                model=(mm.get("model") or "").replace("|", " ")[:60],
                counter=(mm.get("counter_bet") or "").replace("|", " ")[:60],
                experiment=(mm.get("experiment") or "").replace("|", " ")[:60],
            )
            systems_os.append_markdown("MentalModels.md", [row])
        if crew_name == "success_miner":
            entry = context.get("safety_entry") or {}
            text = (
                f"## {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d')}\n"
                f"What went right: {entry.get('what_went_right', '')}\n"
                f"Adaptations: {entry.get('adaptations', '')}\n"
                f"Signals: {entry.get('signals', '')}\n"
                f"Next step: {entry.get('next_step', '')}\n"
            )
            systems_os.append_success_entry(text)
        if crew_name == "coupling_analyzer":
            entries = context.get("coupling_entries", [])
            if entries:
                systems_os.write_coupling_map(entries)
        if crew_name == "foresight_weaver":
            sources = context.get("foresight_sources") or {}
            analysis = context.get("foresight_analysis") or {}
            proposal = (context.get("foresight_proposal") or {}).get("proposal", "")
            result = context.get("foresight_result") or {}
            approval = "yes" if result.get("approved") else f"no (risk {result.get('validation', {}).get('risk', 'n/a')})"
            row = "| {date} | {topic} | {highlight} | {proposal} | {approval} |".format(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                topic=str(sources.get("topic", "n/a"))[:40],
                highlight=str(analysis.get("highlight", ""))[:60],
                proposal=str(proposal)[:60],
                approval=approval[:40],
            )
            systems_os.append_markdown("Foresight.md", [row])
