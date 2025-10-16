from __future__ import annotations

from typing import Any, Dict, List


class MetaLearner:
    """Lightweight meta-learner that tunes evolution heuristics from cycle feedback."""

    WINDOW = 12

    def __init__(self, defaults: Dict[str, Any] | None = None) -> None:
        defaults = defaults or {}
        self.defaults = {
            "repeat_threshold": max(2, int(defaults.get("repeat_threshold", 3))),
            "empty_bullet_threshold": max(1, int(defaults.get("empty_bullet_threshold", 2))),
        }

    # ------------------------------------------------------------------
    def observe(self, state: Dict[str, Any], snapshot: Any) -> Dict[str, Any]:
        """Update rolling statistics and emit heuristic overrides."""

        meta_state = state.setdefault("meta_learner", {})
        window: List[Dict[str, Any]] = meta_state.setdefault("window", [])
        window.append(
            {
                "action": getattr(snapshot, "action", ""),
                "reward": float(getattr(snapshot, "reward", 0.0) or 0.0),
                "bullets": len(getattr(snapshot, "bullets", []) or []),
                "ts": getattr(snapshot, "timestamp", 0),
            }
        )
        if len(window) > self.WINDOW:
            del window[: len(window) - self.WINDOW]

        overrides: Dict[str, Any] = meta_state.setdefault("overrides", {})
        if len(window) < 3:
            meta_state["stats"] = {
                "avg_reward": round(self._avg_reward(window), 3),
                "diversity": 1.0,
                "zero_bullet_ratio": 1.0 if not window else 0.0,
                "window": len(window),
            }
            state["meta_adjustments"] = dict(overrides)
            return overrides

        avg_reward = self._avg_reward(window)
        diversity = self._diversity(window)
        zero_bullet_ratio = self._zero_bullet_ratio(window)

        changed = False

        if avg_reward < 0.5 and diversity < 0.55:
            desired = max(2, self.defaults["repeat_threshold"] - 1)
            if overrides.get("repeat_threshold") != desired:
                overrides["repeat_threshold"] = desired
                changed = True
        elif avg_reward > 0.7 and diversity > 0.7:
            if "repeat_threshold" in overrides:
                overrides.pop("repeat_threshold")
                changed = True

        if avg_reward < 0.5 and zero_bullet_ratio > 0.3:
            if overrides.get("empty_bullet_threshold") != 1:
                overrides["empty_bullet_threshold"] = 1
                changed = True
        elif avg_reward > 0.65 and zero_bullet_ratio < 0.15:
            if "empty_bullet_threshold" in overrides:
                overrides.pop("empty_bullet_threshold")
                changed = True

        if changed:
            meta_state["overrides"] = overrides
        state["meta_adjustments"] = dict(overrides)
        meta_state["stats"] = {
            "avg_reward": round(avg_reward, 3),
            "diversity": round(diversity, 3),
            "zero_bullet_ratio": round(zero_bullet_ratio, 3),
            "window": len(window),
        }
        return overrides

    # ------------------------------------------------------------------
    @staticmethod
    def _avg_reward(window: List[Dict[str, Any]]) -> float:
        if not window:
            return 0.0
        return sum(entry.get("reward", 0.0) for entry in window) / len(window)

    @staticmethod
    def _diversity(window: List[Dict[str, Any]]) -> float:
        actions = [entry.get("action") for entry in window if entry.get("action")]
        if not actions:
            return 1.0
        unique = len(set(actions))
        return unique / max(1, len(actions))

    @staticmethod
    def _zero_bullet_ratio(window: List[Dict[str, Any]]) -> float:
        if not window:
            return 0.0
        zeros = sum(1 for entry in window if int(entry.get("bullets", 0) or 0) == 0)
        return zeros / len(window)
