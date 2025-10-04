"""System-dynamics utilities dedicated to the Dynamics Weaver crew.

This module offers a thin wrapper around :mod:`symbiont.tools.sd_engine`
so blueprints expressed as stock/flow dictionaries can be simulated with
minimal ceremony.  It keeps the interface close to the engineering prompt
the user supplied (stocks, flows, initial conditions) while reusing the
deterministic integrator implemented in ``sd_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from . import sd_engine


@dataclass
class MacroModelConfig:
    stocks: List[str]
    flows: Dict[str, str]
    initial_conditions: Dict[str, float]
    timestep: float = 1.0
    horizon: int = 60


def solve_sd_macro(config: MacroModelConfig) -> Dict[str, object]:
    """Simulate aggregate behaviour for the supplied configuration.

    Parameters mirror the prompt supplied by the user.  The function returns a
    dictionary containing the initial conditions, resolved equations, the
    simulated trajectory, summary statistics, and the plot location generated
    by :func:`symbiont.tools.sd_engine.simulate`.
    """

    blueprint = _build_blueprint(config)
    spec = sd_engine.build_spec(blueprint)
    result = sd_engine.simulate(
        spec,
        horizon=config.horizon,
        noise=0.0,
        make_plot=True,
    )
    equations = _derive_equations(config)
    payload: Dict[str, object] = {
        "initial_conditions": config.initial_conditions,
        "equations": equations,
        "trajectory": result.get("trajectory", []),
        "plot_path": result.get("plot_path"),
        "stats": result.get("stats", {}),
    }
    return payload


def _build_blueprint(config: MacroModelConfig) -> Dict[str, object]:
    stocks = [
        {
            "name": name,
            "initial": float(config.initial_conditions.get(name, 0.0)),
            "min": 0.0,
            "max": 1.0,
        }
        for name in config.stocks
    ]

    flows = [
        {
            "name": f"flow_{idx}_{target}",
            "target": target,
            "expression": expression,
        }
        for idx, (target, expression) in enumerate(config.flows.items())
    ]

    auxiliaries: List[Dict[str, object]] = []
    return {
        "timestep": config.timestep,
        "stocks": stocks,
        "flows": flows,
        "auxiliaries": auxiliaries,
        "parameters": {},
    }


def _derive_equations(config: MacroModelConfig) -> List[str]:
    equations: List[str] = []
    for stock in config.stocks:
        expr = config.flows.get(stock)
        if expr is None:
            continue
        equations.append(f"d_{stock}/dt = {expr}")
    return equations


__all__ = ["MacroModelConfig", "solve_sd_macro"]

