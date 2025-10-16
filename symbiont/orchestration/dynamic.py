"""Dynamic crew generation utilities."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from pydantic import BaseModel, ValidationError

from symbiont.llm.client import LLMClient


DEFAULT_AGENT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "scout": {
        "role": "scout",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "local",
        },
        "cache": "in_memory",
        "tools": ["repo_scan"],
    },
    "architect": {
        "role": "architect",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "local",
        },
        "cache": "in_memory",
        "tools": [],
    },
    "critic": {
        "role": "critic",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "local",
        },
        "cache": "in_memory",
        "tools": [],
    },
    "executor": {
        "role": "executor",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "local",
        },
        "cache": "in_memory",
        "tools": ["scriptify"],
    },
    "ops_researcher": {
        "role": "ops_researcher",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 0,
        },
        "cache": "in_memory",
        "tools": [],
    },
    "judge": {
        "role": "judge",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 400,
        },
        "cache": "in_memory",
        "tools": [],
    },
    "foresight_scout": {
        "role": "foresight_scout",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 0,
        },
        "cache": "in_memory",
        "tools": [],
    },
    "foresight_analyzer": {
        "role": "foresight_analyzer",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 0,
        },
        "cache": "in_memory",
        "tools": [],
    },
    "foresight_suggester": {
        "role": "foresight_suggester",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 0,
        },
        "cache": "in_memory",
        "tools": [],
    },
    "foresight_validator": {
        "role": "foresight_validator",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 0,
        },
        "cache": "in_memory",
        "tools": [],
    },
    "foresight_evolver": {
        "role": "foresight_evolver",
        "llm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "mode": "hybrid",
            "hybrid_threshold_tokens": 0,
        },
        "cache": "in_memory",
        "tools": [],
    },
}

ALLOWED_ROLES = set(DEFAULT_AGENT_LIBRARY.keys())


class GeneratedRole(BaseModel):
    name: str
    tools: List[str] = []


class GeneratedCrew(BaseModel):
    crew_name: str
    roles: List[GeneratedRole]
    supervisor: Dict[str, Any] | None = None


def _heuristic_rogue_score(roles: Iterable[GeneratedRole]) -> float:
    roles = list(roles)
    score = max(0.0, (len(roles) - 5) * 0.15)
    if not any(r.name == "critic" for r in roles):
        score += 0.25
    if any("exec" in r.name.lower() for r in roles if r.name not in ALLOWED_ROLES):
        score += 0.25
    return min(score, 1.0)


def _fallback_roles() -> List[GeneratedRole]:
    return [
        GeneratedRole(name="scout", tools=["repo_scan"]),
        GeneratedRole(name="architect"),
        GeneratedRole(name="critic"),
    ]


def _build_agent_entry(base_name: str, tools: Iterable[str]) -> Dict[str, Any]:
    template = deepcopy(DEFAULT_AGENT_LIBRARY.get(base_name, DEFAULT_AGENT_LIBRARY["architect"]))
    unique_tools = sorted({*template.get("tools", []), *tools})
    template["tools"] = unique_tools
    return template


def generate_dynamic_crew_yaml(
    goal: str,
    cfg: Dict[str, Any],
    *,
    context: str = "",
    output_dir: Path | None = None,
) -> Tuple[str, Path]:
    """Generate a dynamic crew YAML for the given goal.

    Returns the crew name and the YAML path.
    """

    llm_client = LLMClient(cfg.get("llm", {}))
    prompt = f"""
You orchestrate Symbiont crews. Select 3-5 roles from the set {sorted(ALLOWED_ROLES)}
for the goal: "{goal}". Consider context: "{context[:400]}".
Return strict JSON:
{{
  "crew_name": "<slug>",
  "roles": [{{"name": "scout", "tools": ["repo_scan"]}}, ...],
  "supervisor": {{"routing": "sequential"}}
}}
Always include a 'critic'.
"""

    try:
        generated = llm_client.generate(prompt, label="dynamic:crew")
    except Exception:
        generated = ""

    roles: List[GeneratedRole]
    supervisor: Dict[str, Any] | None = None
    crew_name_suffix = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:8]

    if generated:
        try:
            payload = json.loads(generated)
            crew = GeneratedCrew.model_validate(payload)
            roles = [r for r in crew.roles if r.name in ALLOWED_ROLES]
            if not roles:
                roles = _fallback_roles()
            if _heuristic_rogue_score(roles) > 0.5:
                roles = _fallback_roles()
            crew_name_suffix = crew.crew_name[:16] or crew_name_suffix
            supervisor = crew.supervisor
        except (json.JSONDecodeError, ValidationError):
            roles = _fallback_roles()
    else:
        roles = _fallback_roles()

    crew_name = f"dynamic_{crew_name_suffix}"
    output_dir = output_dir or Path("data/artifacts/dynamic")
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{crew_name}.yaml"

    agents: Dict[str, Dict[str, Any]] = {}
    sequence: List[str] = []
    for idx, role in enumerate(roles):
        agent_key = f"agent_{idx}_{role.name}"
        agents[agent_key] = _build_agent_entry(role.name, role.tools)
        sequence.append(agent_key)

    config = {
        "agents": agents,
        "crew": {crew_name: {"sequence": sequence}},
    }
    if supervisor:
        config["crew"][crew_name]["supervisor"] = supervisor

    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return crew_name, yaml_path
