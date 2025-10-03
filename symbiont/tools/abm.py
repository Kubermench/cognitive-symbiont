"""Simple agent-based simulator used by the Dynamics Weaver crew.

This intentionally avoids external dependencies such as Mesa so the
simulation remains lightweight and edge friendly.  A small group of
agents injects stochastic noise into the macro trajectory returned by the
system-dynamics solver, mirroring the behaviour outlined in the engineering
prompt.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class HybridABMConfig:
    trajectory: List[Dict[str, object]]
    num_agents: int = 10
    noise_variance: float = 0.1
    steps: int = 50
    plot_dir: Path = Path("data/artifacts/plots")


@dataclass
class NoiseEvent:
    step: int
    agent_id: int
    role: str
    noise_value: float
    state: Dict[str, float] = field(default_factory=dict)


def run_hybrid_abm(config: HybridABMConfig) -> Dict[str, object]:
    if not config.trajectory:
        raise ValueError("trajectory must not be empty")

    base_states = []
    for point in config.trajectory:
        if not isinstance(point, dict):
            continue
        state = {k: v for k, v in point.items() if k != "time"}
        if state:
            base_states.append(state)
    if not base_states:
        raise ValueError("trajectory must contain state dictionaries")

    agents = _initialise_agents(base_states[0], config)
    noise_events: List[NoiseEvent] = []
    perturbed: List[Dict[str, object]] = []

    total_steps = min(config.steps, len(base_states))
    for step in range(total_steps):
        macro_state = dict(base_states[step])
        _apply_agent_noise(agents, macro_state, step, config, noise_events)
        perturbed.append({"time": config.trajectory[step].get("time", step), **dict(macro_state)})

    plot_path = _render_plot(base_states[:total_steps], perturbed, config)
    return {
        "num_agents": config.num_agents,
        "noise_variance": config.noise_variance,
        "perturbed_trajectory": perturbed,
        "noise_events": [event.__dict__ for event in noise_events],
        "plot_path": plot_path,
    }


def _initialise_agents(
    prototype_state: Dict[str, float], config: HybridABMConfig
) -> List[Dict[str, object]]:
    roles = ["scout", "architect", "critic", "executor"]
    agents: List[Dict[str, object]] = []
    for agent_id in range(config.num_agents):
        agents.append(
            {
                "id": agent_id,
                "role": roles[agent_id % len(roles)],
                "state": dict(prototype_state),
            }
        )
    return agents


def _apply_agent_noise(
    agents: Iterable[Dict[str, object]],
    macro_state: Dict[str, float],
    step: int,
    config: HybridABMConfig,
    noise_events: List[NoiseEvent],
) -> None:
    for agent in agents:
        noise = random.gauss(0.0, config.noise_variance)
        role = agent.get("role", "scout")
        _perturb_state(role, macro_state, noise)
        noise_events.append(
            NoiseEvent(
                step=step,
                agent_id=agent["id"],
                role=role,
                noise_value=noise,
                state=dict(macro_state),
            )
        )


def _perturb_state(role: str, state: Dict[str, float], noise: float) -> None:
    if "autonomy" in state:
        state["autonomy"] = _clamp(state["autonomy"] + 0.05 * noise)
    if role in {"critic", "executor"} and "rogue" in state:
        delta = abs(noise) * (0.12 if role == "critic" else 0.08)
        state["rogue"] = _clamp(state["rogue"] + delta)
    if role == "architect" and "latency" in state:
        state["latency"] = max(0.0, state["latency"] + 0.03 * noise)


def _render_plot(
    base_states: Sequence[Dict[str, float]],
    perturbed: Sequence[Dict[str, object]],
    config: HybridABMConfig,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    config.plot_dir.mkdir(parents=True, exist_ok=True)
    times = list(range(len(base_states)))
    plt.figure(figsize=(6, 4))
    for key in base_states[0].keys():
        original = [state.get(key, math.nan) for state in base_states]
        noisy = [point.get(key, math.nan) for point in perturbed]
        plt.plot(times, original, label=f"{key} (macro)")
        plt.plot(times, noisy, linestyle="--", label=f"{key} (abm)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Hybrid SD + ABM projection")
    plt.legend()
    path = config.plot_dir / "abm_projection.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return str(path)


def _clamp(value: float) -> float:
    return max(0.0, min(1.5, float(value)))


__all__ = ["HybridABMConfig", "run_hybrid_abm"]
