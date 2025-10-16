from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def render_dashboard(
    *,
    labeled: Dict[str, Any],
    ingest_digest: Dict[str, Any] | None = None,
    top_limit: int = 10,
) -> str:
    meta = labeled.get("meta") or {}
    counts = (labeled.get("labels") or {}).get("counts") or {}
    guards_high = labeled.get("guards", {}).get("high", [])
    guards_medium = labeled.get("guards", {}).get("medium", [])
    cycles_low = labeled.get("cycles", {}).get("low_reward", [])

    header = [
        "# Shadow Dashboard",
        "",
        f"- Source path: `{meta.get('path', 'unknown')}`",
        f"- Total clip types: guards={len(guards_high) + len(guards_medium)} cycles={len(cycles_low)}",
        f"- Unique labels: {len(counts)}",
    ]

    if ingest_digest:
        ingested = ", ".join(f"{label}={count}" for label, count in ingest_digest.get("top", [])[:top_limit]) or "(none)"
        header.append(f"- Last ingest top labels: {ingested}")

    table_rows = ["", "## Top Labels", "", "| Label | Count |", "| --- | --- |"]
    for label, count in _top_items(counts.items(), top_limit):
        table_rows.append(f"| `{label}` | {count} |")
    if len(table_rows) == 5:
        table_rows.append("| _(none)_ | 0 |")

    guard_rows = _build_clip_section("High-risk Guards", guards_high, limit=top_limit)
    guard_rows += _build_clip_section("Medium-risk Guards", guards_medium, limit=top_limit)
    cycle_rows = _build_clip_section("Low-reward Cycles", cycles_low, limit=top_limit)

    return "\n".join(header + table_rows + guard_rows + cycle_rows) + "\n"


def _top_items(items: Iterable[Tuple[str, int]], top: int) -> List[Tuple[str, int]]:
    return sorted(
        ((str(label), int(count)) for label, count in items if label),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top]


def _build_clip_section(title: str, clips: List[Dict[str, Any]], *, limit: int) -> List[str]:
    if not clips:
        return []
    rows = ["", f"## {title}", ""]
    for clip in clips[:limit]:
        labels = ", ".join(clip.get("labels") or []) or "-"
        meta = clip.get("meta") or {}
        rogue = clip.get("rogue_score")
        reward = clip.get("reward")
        extra = []
        if rogue is not None:
            extra.append(f"rogue={rogue}")
        if reward is not None:
            extra.append(f"reward={reward}")
        if meta:
            extra.append(f"meta={meta}")
        rows.append(f"- `{labels}` ({'; '.join(extra) or 'no metrics'})")
    return rows

