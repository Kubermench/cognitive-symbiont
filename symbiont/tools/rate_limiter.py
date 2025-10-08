"""Simple configurable rate limiter shared across research adapters."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict

import yaml

_DEFAULT_LIMITS: Dict[str, int] = {"arxiv": 30}
_LIMITS: Dict[str, int] = dict(_DEFAULT_LIMITS)
_STATE: Dict[str, float] = {}
_LOCKS: Dict[str, threading.Lock] = {}
_GLOBAL_LOCK = threading.Lock()


def configure(limits: Dict[str, int] | None) -> None:
    if not limits:
        return
    with _GLOBAL_LOCK:
        for source, per_min in limits.items():
            try:
                per_min_int = int(per_min)
            except (TypeError, ValueError):
                continue
            if per_min_int <= 0:
                continue
            _LIMITS[source.lower()] = per_min_int


def load_from_config(path: str | Path = "configs/config.yaml") -> None:
    config_path = Path(path)
    if not config_path.exists():
        return
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    src_limits = data.get("sources", {}).get("limits")
    if isinstance(src_limits, dict):
        configure(src_limits)


def limit(source: str) -> None:
    if not source:
        source = "default"
    source = source.lower()
    per_min = _LIMITS.get(source, _LIMITS.get("default", 60))
    min_interval = 60.0 / max(1, per_min)
    with _GLOBAL_LOCK:
        lock = _LOCKS.setdefault(source, threading.Lock())
    with lock:
        now = time.monotonic()
        last = _STATE.get(source, 0.0)
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _STATE[source] = now


# Seed limits from config at import time.
try:
    load_from_config()
except Exception:
    pass
