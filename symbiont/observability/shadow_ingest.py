from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..memory.db import MemoryDB
from ..memory import beliefs as belief_api


def load_labeled_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Labeled summary must be a JSON object.")
    return data


def latest_labeled_path(data_root: Path) -> Optional[Path]:
    label_dir = data_root / "artifacts" / "shadow" / "labels"
    if not label_dir.exists():
        return None
    candidates = list(label_dir.glob("*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def summarize_labels(data: Dict[str, Any], *, top: int = 5) -> Dict[str, Any]:
    counts: Dict[str, int] = (data.get("labels") or {}).get("counts") or {}
    top_items = _top_items(counts.items(), top=top)
    guard_total = len(data.get("guards", {}).get("high", [])) + len(
        data.get("guards", {}).get("medium", [])
    )
    cycle_total = len(data.get("cycles", {}).get("low_reward", []))
    return {
        "meta": data.get("meta") or {},
        "counts": counts,
        "top": top_items,
        "guard_total": guard_total,
        "cycle_total": cycle_total,
    }


def ingest_labels(
    db: MemoryDB,
    *,
    summary: Dict[str, Any],
    source_path: Path,
    top: int = 5,
) -> Dict[str, Any]:
    digest = summarize_labels(summary, top=top)
    payload = {
        "kind": "shadow_labels",
        "source": str(source_path),
        "meta": digest["meta"],
        "top": digest["top"],
        "guard_total": digest["guard_total"],
        "cycle_total": digest["cycle_total"],
    }
    db.add_message(role="shadow", content=json.dumps(payload), tags="shadow,labels")

    existing = {b["statement"]: b for b in belief_api.list_beliefs(db)}
    for label, count in digest["top"]:
        statement = f"Shadow label: {label} frequency {count}"
        confidence = min(1.0, 0.3 + 0.1 * min(count, 7))
        evidence = {
            "label": label,
            "count": count,
            "meta": digest["meta"],
            "source_path": str(source_path),
        }
        evidence_json = json.dumps(evidence)
        if statement in existing:
            belief_api.update_belief(
                db,
                existing[statement]["id"],
                confidence=confidence,
                evidence_json=evidence_json,
            )
        else:
            belief_api.add_belief(
                db,
                statement=statement,
                confidence=confidence,
                evidence_json=evidence_json,
            )
    return digest


def _top_items(items: Iterable[Tuple[str, int]], *, top: int) -> List[Tuple[str, int]]:
    counter = Counter()  # type: ignore[var-annotated]
    for key, value in items:
        if key:
            counter[str(key)] += int(value)
    return counter.most_common(top)

