from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..tools.files import ensure_dirs
from .mutation import MutationEngine, MutationIntent
from .meta_learner import MetaLearner


STATE_FILE = Path("data/evolution/state.json")


@dataclass
class CycleSnapshot:
    episode_id: int
    action: str
    bullets: List[str]
    timestamp: int
    reward: float


class CycleReflector:
    """Tracks cycle outcomes and schedules micro-mutations when heuristics drift."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.repo_root = Path(self.config.get("initiative", {}).get("repo_path", ".")).resolve()
        ensure_dirs([STATE_FILE.parent])
        self._state = self._load_state()
        self.mutation_engine = MutationEngine(config=self.config)
        self.meta_learner = MetaLearner((self.config or {}).get("evolution"))

    # ------------------------------------------------------------------
    def observe_cycle(self, result: Dict[str, Any]) -> None:
        """Record the cycle and schedule a mutation intent if heuristics require it."""

        if not result:
            return
        episode_id = int(result.get("episode_id", 0) or 0)
        decision = (result.get("decision") or {}).get("action", "")
        bullets = self._extract_bullets(result.get("trace", []))
        reward = float(result.get("reward", 0.0) or 0.0)

        snapshot = CycleSnapshot(
            episode_id=episode_id,
            action=decision,
            bullets=bullets,
            timestamp=int(time.time()),
            reward=reward,
        )
        self._record_snapshot(snapshot)
        self.meta_learner.observe(self._state, snapshot)

        intent = self._evaluate(snapshot)
        if intent:
            self.mutation_engine.schedule(intent)
        self._save_state()

    # ------------------------------------------------------------------
    def _extract_bullets(self, trace: Iterable[Dict[str, Any]]) -> List[str]:
        for entry in trace or []:
            if entry.get("role") == "architect":
                return list(entry.get("output", {}).get("bullets", []) or [])
        return []

    def _record_snapshot(self, snapshot: CycleSnapshot) -> None:
        history: List[Dict[str, Any]] = self._state.setdefault("history", [])
        history.append({
            "episode_id": snapshot.episode_id,
            "action": snapshot.action,
            "bullets": snapshot.bullets,
            "ts": snapshot.timestamp,
            "reward": snapshot.reward,
        })
        # Keep the tail small to avoid bloat
        if len(history) > 25:
            del history[: len(history) - 25]

    def _evaluate(self, snapshot: CycleSnapshot) -> Optional[MutationIntent]:
        history = self._state.get("history", [])
        drift_cfg = (self.config or {}).get("evolution", {})
        overrides = self._state.get("meta_adjustments", {})
        min_repeats = int(overrides.get("repeat_threshold", drift_cfg.get("repeat_threshold", 3)))
        empty_limit = int(overrides.get("empty_bullet_threshold", drift_cfg.get("empty_bullet_threshold", 2)))

        # 1) repeated actions -> ask planner to diversify suggestions
        if snapshot.action:
            recent_actions = [h.get("action") for h in history[-min_repeats:]]
            if len(recent_actions) >= min_repeats and len(set(recent_actions)) == 1:
                return MutationIntent(
                    kind="planner_prompt",
                    rationale=(
                        f"Action '{snapshot.action}' repeated {min_repeats}x; tweak architect prompt for diversity."
                    ),
                    details={"strategy": "promote_diversity"},
                )

        # 2) missing bullets -> tighten repo scan hinting
        if not snapshot.bullets:
            empty_streak = self._state.setdefault("empty_streak", 0) + 1
            self._state["empty_streak"] = empty_streak
            if empty_streak >= empty_limit:
                self._state["empty_streak"] = 0
                return MutationIntent(
                    kind="planner_prompt",
                    rationale="Architect produced no bullets twice; reinforce repo-scan heuristics.",
                    details={"strategy": "strengthen_repo_guidance"},
                )
        else:
            self._state["empty_streak"] = 0

        return None

    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        if not STATE_FILE.exists():
            return {}
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self) -> None:
        try:
            STATE_FILE.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except Exception:
            pass
