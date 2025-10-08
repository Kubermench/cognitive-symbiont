from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Tuple

SYSTEMS_ROOT = Path("systems")


def ensure_systems_root() -> None:
    SYSTEMS_ROOT.mkdir(parents=True, exist_ok=True)


def append_markdown(table_path: str, rows: Iterable[str]) -> None:
    ensure_systems_root()
    path = SYSTEMS_ROOT / table_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def append_success_entry(text: str) -> None:
    ensure_systems_root()
    log_path = SYSTEMS_ROOT / "SafetyII.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{text.strip()}\n")


def update_flow_metrics(payload: dict[str, Any]) -> None:
    ensure_systems_root()
    flow_path = SYSTEMS_ROOT / "FlowMetrics.json"
    flow_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")


def write_coupling_map(entries: List[Tuple[str, str, float]]) -> None:
    ensure_systems_root()
    path = SYSTEMS_ROOT / "CouplingMap.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("component_a,component_b,coupling_score\n")
        for a, b, score in entries:
            handle.write(f"{a},{b},{score:.3f}\n")
