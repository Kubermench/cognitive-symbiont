from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

HISTORY_FILENAME = "history.jsonl"


def record_history(
    *,
    data_root: Path,
    labeled: Dict[str, Any],
    summary: Dict[str, Any],
    source_path: Path,
    top_k: int = 5,
    max_entries: int = 200,
) -> Path:
    history_dir = data_root / "artifacts" / "shadow"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / HISTORY_FILENAME
    counts = (labeled.get("labels") or {}).get("counts") or {}
    top = Counter({str(k): int(v) for k, v in counts.items() if k}).most_common(top_k)
    entry = {
        "ts": int(time.time()),
        "source": str(source_path),
        "total_labels": len(counts),
        "guard_total": len(summary.get("guards", {}).get("high", []))
        + len(summary.get("guards", {}).get("medium", [])),
        "cycle_total": len(summary.get("cycles", {}).get("low_reward", [])),
        "top": top,
        "meta": summary.get("meta"),
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=True) + "\n")

    if max_entries and max_entries > 0:
        with history_path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if len(lines) > max_entries:
            lines = lines[-max_entries:]
            history_path.write_text("".join(lines), encoding="utf-8")
    return history_path


def load_history(data_root: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    history_path = data_root / "artifacts" / "shadow" / HISTORY_FILENAME
    if not history_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    entries.sort(key=lambda e: e.get("ts", 0), reverse=True)
    if limit is not None:
        entries = entries[: max(limit, 0)]
    return entries
