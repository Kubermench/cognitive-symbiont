from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

from prometheus_client import Gauge, start_http_server

from symbiont.llm.budget import TokenBudget  # noqa: F401  (ensure dependency accounted for)

BUDGET_USED = Gauge("symbiont_budget_used_tokens", "Tokens consumed per budget label", ["label"])
BUDGET_LIMIT = Gauge("symbiont_budget_limit_tokens", "Token budget limit per label", ["label"])
BUDGET_LATENCY = Gauge("symbiont_budget_latency_seconds", "Average LLM latency per budget label", ["label"])
ROGUE_SCORE = Gauge("symbiont_rogue_score", "Rogue baseline and forecast metrics", ["metric"])
ROGUE_ALERT = Gauge("symbiont_rogue_alert", "1 when rogue alert triggered")


def _load_config(config_path: Path) -> Dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _snapshot_budget(data_root: Path) -> Dict[str, Dict[str, Any]]:
    token_dir = data_root / "token_budget"
    if not token_dir.exists():
        return {}
    snapshots: Dict[str, Dict[str, Any]] = {}
    for path in token_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = payload.get("label") or path.stem
        snapshots[label] = payload
    return snapshots


def _load_budget_history(data_root: Path, limit: int = 500) -> List[Dict[str, Any]]:
    history_path = data_root / "token_budget" / "history.jsonl"
    if not history_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except Exception:
                    continue
    except Exception:
        return []
    if limit <= 0:
        return rows
    return rows[-limit:]


def _latency_by_label(events: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for event in events:
        label = event.get("label")
        latency = event.get("latency_seconds")
        if not label or latency is None:
            continue
        try:
            buckets[label].append(float(latency))
        except (TypeError, ValueError):
            continue
    return {label: statistics.mean(values) for label, values in buckets.items() if values}


def _latest_governance_snapshot(data_root: Path) -> Dict[str, Any]:
    graphs_dir = data_root / "artifacts" / "graphs"
    if not graphs_dir.exists():
        return {}
    for path in sorted(graphs_dir.glob("graph_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        governance = payload.get("governance")
        if governance:
            return governance
    return {}


def serve_metrics(config_path: str, *, port: int = 8001, interval: int = 5) -> None:
    cfg = _load_config(Path(config_path))
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent).resolve()
    start_http_server(port)
    while True:
        snapshots = _snapshot_budget(data_root)
        for label, payload in snapshots.items():
            used = float(payload.get("used", 0) or 0)
            limit = float(payload.get("limit", 0) or 0)
            BUDGET_USED.labels(label=label).set(used)
            BUDGET_LIMIT.labels(label=label).set(limit)
        history = _load_budget_history(data_root)
        latency_map = _latency_by_label(history)
        for label, latency in latency_map.items():
            BUDGET_LATENCY.labels(label=label).set(latency)

        governance = _latest_governance_snapshot(data_root)
        if governance:
            baseline = float(governance.get("rogue_baseline", 0.0) or 0.0)
            forecast_values = [baseline]
            for value in governance.get("rogue_forecast", []):
                try:
                    forecast_values.append(float(value))
                except (TypeError, ValueError):
                    continue
            forecast_max = max(forecast_values)
            threshold = float(governance.get("alert_threshold", 0.0) or 0.0)
            ROGUE_SCORE.labels(metric="baseline").set(baseline)
            ROGUE_SCORE.labels(metric="forecast_max").set(forecast_max)
            ROGUE_SCORE.labels(metric="threshold").set(threshold)
            ROGUE_ALERT.set(1.0 if governance.get("alert") else 0.0)
        else:
            ROGUE_ALERT.set(0.0)
        time.sleep(max(1, interval))
