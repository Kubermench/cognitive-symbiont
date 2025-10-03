from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .subself import SubSelf
from ..llm.client import LLMClient
from ..memory.db import MemoryDB
from ..tools import scriptify

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

    def generate(self, prompt: str) -> str:
        key_src = f"{self.agent_id}:{prompt}".encode("utf-8")
        cache_key = hashlib.sha256(key_src).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        response = self.client.generate(prompt)
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
        data = yaml.safe_load(path.read_text()) or {}
        agents_data = data.get("agents", {})
        crews_data = data.get("crew", {}) or data.get("crews", {})
        agents: Dict[str, AgentSpec] = {}
        for name, spec in agents_data.items():
            agents[name] = AgentSpec(
                name=name,
                role=spec.get("role", name),
                llm=spec.get("llm", {}),
                cache=spec.get("cache"),
                tools=spec.get("tools", []),
            )
        crews: Dict[str, CrewSpec] = {}
        for name, spec in crews_data.items():
            seq = spec.get("sequence") or spec.get("roles") or []
            crews[name] = CrewSpec(name=name, sequence=seq)
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

    def run(self, crew_name: str, goal: str) -> Path:
        crew = self.registry.get_crew(crew_name)
        context = {"goal": goal, "episode_id": None, "cwd": Path.cwd()}
        memory = self.db
        memory.ensure_schema()
        memory_conn = memory
        outputs = []
        latest_bullets: list[str] = []

        for agent_id in crew.sequence:
            agent_spec = self.registry.get_agent(agent_id)
            llm_client = agent_spec.create_llm_client()
            role_dict = {"name": agent_spec.role}
            agent = SubSelf(role_dict, self.cfg, llm_client=llm_client)
            result = agent.run(context, memory_conn)
            outputs.append({"agent": agent_id, "result": result})
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

        artifact_path = self._persist_outputs(crew_name, goal, outputs)
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

