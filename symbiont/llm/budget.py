from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import threading


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    """Tracks approximate token usage across an agent run."""

    limit: int = 0
    label: str = "run"
    used: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    sink_path: Optional[Path] = None
    history_path: Optional[Path] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sink_path and not self.history_path:
            self.history_path = self.sink_path.parent / "history.jsonl"

    def remaining(self) -> Optional[int]:
        if self.limit <= 0:
            return None
        return max(self.limit - self.used, 0)

    def can_consume(self, prompt_tokens: int) -> bool:
        if self.limit <= 0:
            return True
        return self.used + prompt_tokens <= self.limit

    def log_attempt(
        self,
        *,
        prompt_tokens: int,
        response_tokens: int,
        provider: str,
        model: str,
        label: str,
        source: str,
        outcome: str,
        latency: float,
        attempt: int | None = None,
        error: str | None = None,
    ) -> None:
        if outcome == "ok":
            self.used += prompt_tokens + response_tokens
        event = {
            "ts": int(time.time()),
            "label": label,
            "provider": provider,
            "model": model,
            "source": source,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "outcome": outcome,
            "latency_seconds": round(latency, 3),
            "remaining": self.remaining(),
        }
        if attempt is not None:
            event["attempt"] = attempt
        if error:
            event["error"] = error
        with self._lock:
            self.events.append(event)
            self._write_snapshot()
            self._append_history(event)

    def note_denied(self, *, prompt_tokens: int, provider: str, model: str, label: str, source: str) -> None:
        self.log_attempt(
            prompt_tokens=prompt_tokens,
            response_tokens=0,
            provider=provider,
            model=model,
            label=label,
            source=source,
            outcome="denied",
            latency=0.0,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "used": self.used,
            "events": list(self.events),
            "label": self.label,
        }

    def restore(self, data: Optional[Dict[str, Any]]) -> None:
        if not data:
            return
        self.limit = int(data.get("limit", self.limit))
        self.used = int(data.get("used", 0))
        self.label = data.get("label", self.label)
        self.events = list(data.get("events", []))

    @staticmethod
    def estimate(text: str) -> int:
        return _estimate_tokens(text)

    def _write_snapshot(self) -> None:
        if not self.sink_path:
            return
        try:
            self.sink_path.parent.mkdir(parents=True, exist_ok=True)
            self.sink_path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        except Exception:
            pass

    def _append_history(self, event: Dict[str, Any]) -> None:
        if not self.history_path:
            return
        entry = dict(event)
        entry.setdefault("label", self.label)
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception:
            pass
