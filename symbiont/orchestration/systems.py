from __future__ import annotations

from typing import Iterable, Dict, Any


def _extract_rogue_scores(history: Iterable[Dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for entry in history:
        result = entry.get("result") if isinstance(entry, dict) else None
        if isinstance(result, dict):
            raw = result.get("rogue_score")
            if raw is None:
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            if score < 0:
                score = 0.0
            if score > 1:
                score = 1.0
            scores.append(score)
    return scores


def forecast_rogue_drift(
    history: Iterable[Dict[str, Any]],
    *,
    drift_rate: float = 0.05,
    horizon: int = 50,
) -> list[float]:
    """Predict rogue-score drift using a simple exponential smoothing model."""

    scores = _extract_rogue_scores(history)
    if not scores or horizon <= 0:
        return []

    rate = max(0.0, float(drift_rate))
    level = max(scores)
    forecast: list[float] = []
    for _ in range(horizon):
        level = min(1.0, level + rate * (1.0 - level))
        forecast.append(round(level, 4))
    return forecast


def governance_snapshot(
    history: Iterable[Dict[str, Any]],
    *,
    drift_rate: float = 0.05,
    horizon: int = 50,
    alert_threshold: float = 0.6,
) -> Dict[str, Any]:
    scores = _extract_rogue_scores(history)
    if not scores:
        return {}
    forecast = forecast_rogue_drift(history, drift_rate=drift_rate, horizon=horizon)
    if not forecast:
        return {}
    baseline = round(max(scores), 4)
    threshold = max(0.0, min(1.0, float(alert_threshold)))
    alert = baseline >= threshold or any(value >= threshold for value in forecast)
    return {
        "rogue_baseline": baseline,
        "rogue_forecast": forecast,
        "drift_rate": round(float(drift_rate), 4),
        "alert_threshold": threshold,
        "alert": alert,
    }
