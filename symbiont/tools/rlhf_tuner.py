"""Tiny RLHF-style tuner for foresight hunt queries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


def _slug(text: str) -> str:
    return text.lower().strip()


@dataclass
class RLHFTuner:
    """Maintains rolling rewards per query to adapt hunt prompts."""

    state_path: Path = Path("data/foresight/rlhf_state.json")
    decay: float = 0.7
    boost_threshold: float = 0.65
    penalty_threshold: float = 0.4
    history: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.history = data
            except Exception:
                self.history = {}

    def record_outcome(self, query: str, reward: float, proposal: Dict[str, Any]) -> None:
        key = _slug(query)
        state = self.history.setdefault(key, {"reward": 0.5, "count": 0, "hash": ""})
        reward = max(0.0, min(1.0, reward))
        state["reward"] = self.decay * state["reward"] + (1.0 - self.decay) * reward
        state["count"] = int(state.get("count", 0)) + 1
        diff = str(proposal.get("diff", ""))
        state["hash"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]
        self._flush()

    def suggest_query(self, query: str) -> str:
        key = _slug(query)
        state = self.history.get(key)
        if not state:
            return query
        reward = float(state.get("reward", 0.5))
        if reward >= self.boost_threshold and "agentic" not in query.lower():
            return f"{query} agentic"
        if reward <= self.penalty_threshold:
            return f"{query} resilience"
        return query

    def _flush(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
