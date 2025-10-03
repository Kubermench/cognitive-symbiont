"""Hybrid SD + ABM orchestrator implementing the Dynamics Weaver crew.

The implementation keeps close to the engineering prompt: a scout maps out
stocks/flows, a macro simulation projects the aggregate behaviour, an ABM
layer injects stochastic noise, and a strategist recommends interventions
based on the combined output.  Results are written to the artifacts folder
and to the ``sd_runs`` telemetry table for downstream analytics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..memory.db import MemoryDB
from ..tools.abm import HybridABMConfig, run_hybrid_abm
from ..tools.ode_solver import MacroModelConfig, solve_sd_macro


@dataclass
class DynamicsWeaverResult:
    goal: str
    blueprint: Dict[str, object]
    sd_results: Dict[str, object]
    abm_results: Dict[str, object]
    interventions: List[Dict[str, object]]
    risk_score: float
    artifact_path: Path


def run_dynamics_weaver(goal: str, cfg: Dict[str, object]) -> DynamicsWeaverResult:
    db = MemoryDB(db_path=cfg["db_path"])
    db.ensure_schema()

    blueprint = _blueprint_from_goal(goal)
    macro_config = MacroModelConfig(
        stocks=blueprint["stocks"],
        flows=blueprint["flows"],
        initial_conditions=blueprint["initial_conditions"],
        timestep=blueprint["timestep"],
        horizon=blueprint["horizon"],
    )
    sd_results = solve_sd_macro(macro_config)

    abm_config = HybridABMConfig(
        trajectory=sd_results["trajectory"],
        num_agents=blueprint["abm"]["num_agents"],
        noise_variance=blueprint["abm"]["noise_variance"],
        steps=blueprint["abm"]["steps"],
    )
    abm_results = run_hybrid_abm(abm_config)

    risk_score = _estimate_risk(sd_results, abm_results)
    interventions = _propose_interventions(sd_results, abm_results, risk_score)

    artifact_path = _persist_results(goal, blueprint, sd_results, abm_results, interventions)

    db.add_sd_run(
        goal=goal,
        label="dynamics_weaver",
        horizon=blueprint["horizon"],
        timestep=blueprint["timestep"],
        stats=sd_results.get("stats", {}),
        plot_path=sd_results.get("plot_path"),
    )

    return DynamicsWeaverResult(
        goal=goal,
        blueprint=blueprint,
        sd_results=sd_results,
        abm_results=abm_results,
        interventions=interventions,
        risk_score=risk_score,
        artifact_path=artifact_path,
    )


def _blueprint_from_goal(goal: str) -> Dict[str, object]:
    lowered = goal.lower()
    autonomy_start = 0.55 if "autonomy" in lowered else 0.45
    rogue_start = 0.28 if "rogue" in lowered else 0.2
    latency_start = 0.3 if "latency" in lowered else 0.18

    return {
        "stocks": ["autonomy", "rogue", "latency", "knowledge"],
        "flows": {
            "autonomy": "0.08*(1 - rogue) + 0.05*knowledge - 0.04*latency",
            "rogue": "0.04*autonomy - 0.06*knowledge + 0.02",
            "latency": "0.03*autonomy - 0.05",
            "knowledge": "0.07*autonomy - 0.03*rogue",
        },
        "initial_conditions": {
            "autonomy": autonomy_start,
            "rogue": rogue_start,
            "latency": latency_start,
            "knowledge": 0.5,
        },
        "timestep": 1.0,
        "horizon": 80,
        "abm": {
            "num_agents": 12,
            "noise_variance": 0.12,
            "steps": 60,
        },
    }


def _estimate_risk(sd_results: Dict[str, object], abm_results: Dict[str, object]) -> float:
    stats = sd_results.get("stats", {})
    rogue_last = float(stats.get("rogue", {}).get("last", 0.0))
    latency_peak = float(stats.get("latency", {}).get("max", 0.0))
    abm_noise = len(abm_results.get("noise_events", []))
    score = 0.6 * rogue_last + 0.3 * latency_peak + 0.001 * abm_noise
    return min(1.0, round(score, 3))


def _propose_interventions(
    sd_results: Dict[str, object],
    abm_results: Dict[str, object],
    risk_score: float,
) -> List[Dict[str, object]]:
    interventions: List[Dict[str, object]] = []
    stats = sd_results.get("stats", {})
    rogue_last = float(stats.get("rogue", {}).get("last", 0.0))
    latency_avg = float(stats.get("latency", {}).get("avg", 0.0))
    autonomy_last = float(stats.get("autonomy", {}).get("last", 0.0))

    if rogue_last > 0.7:
        interventions.append(
            {
                "var": "guard_threshold",
                "adjustment": -0.1,
                "rationale": "Rogue score trending high; tighten guard escalation",
            }
        )
    if latency_avg > 0.4:
        interventions.append(
            {
                "var": "hybrid_bias",
                "adjustment": -0.2,
                "rationale": "Latency creeping up; shift routing toward local models",
            }
        )
    if autonomy_last < 0.5:
        interventions.append(
            {
                "var": "knowledge_refresh",
                "adjustment": 0.15,
                "rationale": "Autonomy stagnating; expand GraphRAG knowledge refresh",
            }
        )
    if not interventions:
        interventions.append(
            {
                "var": "monitoring",
                "adjustment": 0.0,
                "rationale": "Metrics stable. Continue guarded execution with 5-cycle reviews.",
            }
        )
    interventions.append(
        {
            "var": "risk_score",
            "adjustment": risk_score,
            "rationale": "Composite risk reported for governance dashboards.",
        }
    )
    return interventions


def _persist_results(
    goal: str,
    blueprint: Dict[str, object],
    sd_results: Dict[str, object],
    abm_results: Dict[str, object],
    interventions: List[Dict[str, object]],
) -> Path:
    crew_dir = Path("data/artifacts/crews/dynamics_weaver")
    crew_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    path = crew_dir / f"crew_{timestamp}.json"
    payload = {
        "goal": goal,
        "timestamp": timestamp,
        "blueprint": blueprint,
        "sd_results": sd_results,
        "abm_results": abm_results,
        "interventions": interventions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = ["run_dynamics_weaver", "DynamicsWeaverResult"]

