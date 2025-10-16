"""Utility helpers exposed under ``symbiont.tools``.

The original eager imports pulled in heavyweight optional dependencies, so the
modules are now loaded lazily on demand.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "abm",
    "arxiv_fetcher",
    "code_runner",
    "coupling_analyzer",
    "files",
    "ode_solver",
    "repo_scan",
    "research",
    "scriptify",
    "sd_engine",
    "secrets",
    "systems_os",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
