from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

_LOCK = threading.Lock()


class ShadowClipCollector:
    """Persist lightweight \"shadow clips\" that describe cycle decisions and guard outcomes.

    The collector writes JSONL rows so downstream crews can build datasets without schema churn.
    Each record carries:
        - kind: `cycle` or `guard`
        - ts: unix timestamp
        - payload: caller supplied dict
        - tags: optional high level categorizations for filtering (e.g. ['eternal', 'blocked'])
        - meta: optional metadata such as rogue score or goal hash
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        filename: str = "shadow_clips.jsonl",
    ) -> None:
        root = Path(base_dir)
        if root.is_file():
            root = root.parent
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / filename

    def record(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a new clip to disk."""
        clip = {
            "kind": kind,
            "ts": int(time.time()),
            "payload": _scrub_json(payload),
            "tags": sorted(set(tags or [])),
            "meta": _scrub_json(meta or {}),
        }
        line = json.dumps(clip, ensure_ascii=True)
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def record_cycle(
        self,
        goal: str,
        decision: Dict[str, Any],
        trace: Iterable[Dict[str, Any]],
        *,
        reward: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "goal": goal,
            "decision": decision,
            "trace": list(trace),
        }
        if reward is not None:
            payload["reward"] = reward
        meta = {"reward": reward, **(meta or {})} if reward is not None else (meta or {})
        self.record("cycle", payload, tags=tags, meta=meta)

    def record_guard(
        self,
        script_path: Union[str, Path],
        analysis: Dict[str, Any],
        *,
        plan_text: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "script_path": str(script_path),
            "analysis": analysis,
        }
        if plan_text:
            payload["plan_text"] = plan_text
        meta = {"rogue_score": analysis.get("rogue_score")} | (meta or {})
        self.record("guard", payload, tags=tags, meta=meta)


def _scrub_json(value: Any) -> Any:
    """Ensure clip payloads are JSON serialisable and bounded."""
    if isinstance(value, dict):
        return {str(k)[:80]: _scrub_json(v) for k, v in list(value.items())[:64]}
    if isinstance(value, list):
        return [_scrub_json(v) for v in value[:64]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 5000:
            return value[:5000] + "...[truncated]"
        return value
    return str(value)

