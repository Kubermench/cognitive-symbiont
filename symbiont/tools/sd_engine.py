"""Lightweight system-dynamics simulation utilities.

This module provides simple stock/flow simulation primitives so crews can
project autonomy, rogue risk, and other metrics before executing actions.
The implementation intentionally keeps dependencies minimal (no SciPy),
using an explicit Euler integrator with guardrails suitable for quick
foresight runs on edge hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "Stock",
    "Flow",
    "Auxiliary",
    "SimulationSpec",
    "build_spec",
    "simulate",
]


Expression = str | Callable[[Dict[str, float], Dict[str, float], Dict[str, float]], float]


@dataclass
class Stock:
    """Represents an accumulated quantity within the SD model."""

    name: str
    initial: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Flow:
    """Defines a rate-of-change feeding into a target stock."""

    name: str
    target: str
    expression: Expression
    weight: float = 1.0


@dataclass
class Auxiliary:
    """Intermediate calculation used by flows."""

    name: str
    expression: Expression


@dataclass
class SimulationSpec:
    """Container summarising the SD setup for a simulation run."""

    stocks: List[Stock]
    flows: List[Flow]
    auxiliaries: List[Auxiliary] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    timestep: float = 1.0

    def stock_index(self) -> Dict[str, int]:
        return {stock.name: idx for idx, stock in enumerate(self.stocks)}


_SAFE_GLOBALS = {
    "__builtins__": {
        "abs": abs,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
    },
    "math": math,
    "np": np,
}


def _evaluate_expression(
    expr: Expression,
    *,
    state: Dict[str, float],
    aux: Dict[str, float],
    params: Dict[str, float],
) -> float:
    if callable(expr):  # Allow advanced users to pass callables directly.
        return float(expr(state, aux, params))
    # Fallback to string evaluation in a constrained namespace.
    locals_dict = {
        **state,
        **aux,
        **params,
    }
    try:
        return float(eval(expr, _SAFE_GLOBALS, locals_dict))
    except Exception as exc:  # pragma: no cover - logged by caller
        raise ValueError(f"Failed evaluating expression '{expr}': {exc}") from exc


def build_spec(config: Dict[str, object]) -> SimulationSpec:
    """Create a :class:`SimulationSpec` from a simple dictionary structure.

    Example ``config`` structure::

        {
            "timestep": 1.0,
            "stocks": [{"name": "autonomy", "initial": 0.4}],
            "flows": [
                {"name": "success_gain", "target": "autonomy", "expression": "0.12 * success_rate"},
                {"name": "guard_loss", "target": "autonomy", "expression": "-0.08 * guard_pressure"},
            ],
            "auxiliaries": [
                {"name": "success_rate", "expression": "min(1.0, proposals)"},
                {"name": "guard_pressure", "expression": "max(0.0, rogue - 0.3)"},
            ],
            "parameters": {"proposals": 0.5, "rogue": 0.25},
        }
    """

    stocks = [
        Stock(
            name=str(entry["name"]),
            initial=float(entry.get("initial", 0.0)),
            min_value=_coerce_optional(entry.get("min")),
            max_value=_coerce_optional(entry.get("max")),
        )
        for entry in config.get("stocks", [])
    ]

    flows = [
        Flow(
            name=str(entry["name"]),
            target=str(entry["target"]),
            expression=entry.get("expression", "0"),
            weight=float(entry.get("weight", 1.0)),
        )
        for entry in config.get("flows", [])
    ]

    auxiliaries = [
        Auxiliary(name=str(entry["name"]), expression=entry.get("expression", "0"))
        for entry in config.get("auxiliaries", [])
    ]

    params = {str(k): float(v) for k, v in dict(config.get("parameters", {})).items()}
    timestep = float(config.get("timestep", 1.0))
    return SimulationSpec(stocks=stocks, flows=flows, auxiliaries=auxiliaries, parameters=params, timestep=timestep)


def simulate(
    spec: SimulationSpec,
    *,
    horizon: int = 50,
    noise: float = 0.0,
    artifacts_dir: Path | str = Path("data/artifacts/plots"),
    make_plot: bool = True,
    stability_tol: float = 0.01,
    min_steps: int = 12,
) -> Dict[str, object]:
    """Run the simulation described by ``spec`` and capture artifacts."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    state = {stock.name: float(stock.initial) for stock in spec.stocks}
    stock_bounds = {stock.name: (stock.min_value, stock.max_value) for stock in spec.stocks}
    timeline: List[Dict[str, float]] = [{"time": 0.0, **state}]

    stable_streak = 0
    for step in range(1, horizon + 1):
        aux_values = {
            aux.name: _evaluate_expression(aux.expression, state=state, aux={}, params=spec.parameters)
            for aux in spec.auxiliaries
        }
        delta: Dict[str, float] = {stock.name: 0.0 for stock in spec.stocks}

        for flow in spec.flows:
            value = _evaluate_expression(flow.expression, state=state, aux=aux_values, params=spec.parameters)
            delta[flow.target] = delta.get(flow.target, 0.0) + flow.weight * value

        deltas_for_stability: Dict[str, float] = {}
        for stock in spec.stocks:
            change = delta.get(stock.name, 0.0) * spec.timestep
            if noise:
                change += np.random.normal(0.0, noise)
            new_value = state[stock.name] + change
            minimum, maximum = stock_bounds[stock.name]
            if minimum is not None:
                new_value = max(minimum, new_value)
            if maximum is not None:
                new_value = min(maximum, new_value)
            deltas_for_stability[stock.name] = abs(new_value - state[stock.name])
            state[stock.name] = new_value
        timeline.append({"time": float(step * spec.timestep), **state})

        if all(delta <= stability_tol for delta in deltas_for_stability.values()):
            stable_streak += 1
            if step >= min_steps and stable_streak >= 3:
                break
        else:
            stable_streak = 0

    stats = _summarise(spec.stocks, timeline)
    plot_path: Optional[str] = None
    if make_plot:
        plot_path = _try_make_plot(spec.stocks, timeline, artifacts_dir)

    return {
        "trajectory": timeline,
        "stats": stats,
        "plot_path": plot_path,
        "timestep": spec.timestep,
        "parameters": spec.parameters,
        "steps": len(timeline) - 1,
    }


def _summarise(stocks: Iterable[Stock], timeline: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for stock in stocks:
        values = [point[stock.name] for point in timeline]
        summary[stock.name] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "last": float(values[-1]),
            "avg": float(sum(values) / len(values)),
        }
    return summary


def _try_make_plot(
    stocks: Iterable[Stock],
    timeline: Sequence[Dict[str, float]],
    artifacts_dir: Path | str,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    times = [point["time"] for point in timeline]
    plt.figure(figsize=(6, 4))
    for stock in stocks:
        plt.plot(times, [point[stock.name] for point in timeline], label=stock.name)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("System Dynamics Projection")
    plt.legend()
    plt.tight_layout()
    path = artifacts_dir / f"sd_projection_{int(timeline[-1]['time'])}.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def _coerce_optional(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [name for name in globals() if name[0].isupper() or name in {"build_spec", "simulate"}]
