from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

from prometheus_client import Gauge, Counter, Histogram, Summary, start_http_server, generate_latest, CONTENT_TYPE_LATEST

from symbiont.llm.budget import TokenBudget  # noqa: F401  (ensure dependency accounted for)

# Token budget metrics
BUDGET_USED = Gauge("symbiont_budget_used_tokens", "Tokens consumed per budget label", ["label"])
BUDGET_LIMIT = Gauge("symbiont_budget_limit_tokens", "Token budget limit per label", ["label"])
BUDGET_LATENCY = Gauge("symbiont_budget_latency_seconds", "Average LLM latency per budget label", ["label"])

# Rogue detection metrics
ROGUE_SCORE = Gauge("symbiont_rogue_score", "Rogue baseline and forecast metrics", ["metric"])
ROGUE_ALERT = Gauge("symbiont_rogue_alert", "1 when rogue alert triggered")

# System health metrics
DAEMON_STATUS = Gauge("symbiont_daemon_status", "Daemon status (1=running, 0=stopped)")
DAEMON_UPTIME = Gauge("symbiont_daemon_uptime_seconds", "Daemon uptime in seconds")
DAEMON_LAST_HEARTBEAT = Gauge("symbiont_daemon_last_heartbeat_seconds", "Last daemon heartbeat timestamp")
ACTIVE_WORKERS = Gauge("symbiont_active_workers", "Number of active swarm workers")
TOTAL_WORKERS = Gauge("symbiont_total_workers", "Total number of swarm workers")

# LLM operation metrics
LLM_REQUESTS_TOTAL = Counter("symbiont_llm_requests_total", "Total LLM requests", ["provider", "model", "outcome"])
LLM_REQUEST_DURATION = Histogram("symbiont_llm_request_duration_seconds", "LLM request duration", ["provider", "model"])
LLM_TOKENS_TOTAL = Counter("symbiont_llm_tokens_total", "Total tokens processed", ["provider", "model", "type"])

# Initiative metrics
INITIATIVE_PROPOSALS_TOTAL = Counter("symbiont_initiative_proposals_total", "Total initiative proposals", ["reason", "outcome"])
INITIATIVE_CYCLES_TOTAL = Counter("symbiont_initiative_cycles_total", "Total initiative cycles", ["outcome"])
INITIATIVE_CYCLE_DURATION = Histogram("symbiont_initiative_cycle_duration_seconds", "Initiative cycle duration")

# Memory and storage metrics
MEMORY_OPERATIONS_TOTAL = Counter("symbiont_memory_operations_total", "Total memory operations", ["operation", "outcome"])
MEMORY_SIZE_BYTES = Gauge("symbiont_memory_size_bytes", "Memory storage size in bytes")
DATABASE_CONNECTIONS = Gauge("symbiont_database_connections", "Number of active database connections")

# Error and retry metrics
RETRY_ATTEMPTS_TOTAL = Counter("symbiont_retry_attempts_total", "Total retry attempts", ["operation", "outcome"])
ERRORS_TOTAL = Counter("symbiont_errors_total", "Total errors", ["error_type", "component"])

# Custom metrics for specific features
FORESIGHT_RUNS_TOTAL = Counter("symbiont_foresight_runs_total", "Total foresight runs", ["outcome"])
FORESIGHT_RELEVANCE_SCORE = Gauge("symbiont_foresight_relevance_score", "Foresight relevance score")
TRANSCRIPT_INGESTIONS_TOTAL = Counter("symbiont_transcript_ingestions_total", "Total transcript ingestions", ["outcome"])
TRANSCRIPT_SIZE_BYTES = Histogram("symbiont_transcript_size_bytes", "Transcript size in bytes")


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


def update_system_metrics(cfg: Dict[str, Any]) -> None:
    """Update system health metrics."""
    try:
        from ..initiative.state import get_state_store
        store = get_state_store(cfg)
        
        # Daemon status
        daemon_status = store.load_daemon()
        if daemon_status:
            DAEMON_STATUS.set(1.0 if daemon_status.get("status") == "running" else 0.0)
            DAEMON_LAST_HEARTBEAT.set(float(daemon_status.get("last_heartbeat_ts", 0)))
            
            # Calculate uptime
            started_ts = daemon_status.get("details", {}).get("started_ts", 0)
            if started_ts > 0:
                uptime = int(time.time()) - started_ts
                DAEMON_UPTIME.set(float(uptime))
        else:
            DAEMON_STATUS.set(0.0)
            DAEMON_LAST_HEARTBEAT.set(0.0)
            DAEMON_UPTIME.set(0.0)
        
        # Worker metrics
        workers = store.list_swarm_workers()
        active_workers = len([w for w in workers if w.get("status") == "active"])
        TOTAL_WORKERS.set(float(len(workers)))
        ACTIVE_WORKERS.set(float(active_workers))
        
    except Exception:
        # Set default values on error
        DAEMON_STATUS.set(0.0)
        DAEMON_LAST_HEARTBEAT.set(0.0)
        DAEMON_UPTIME.set(0.0)
        TOTAL_WORKERS.set(0.0)
        ACTIVE_WORKERS.set(0.0)


def update_llm_metrics(history: List[Dict[str, Any]]) -> None:
    """Update LLM operation metrics from history."""
    for event in history:
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        outcome = event.get("outcome", "unknown")
        latency = event.get("latency_seconds", 0)
        prompt_tokens = event.get("prompt_tokens", 0)
        response_tokens = event.get("response_tokens", 0)
        
        # Count requests
        LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, outcome=outcome).inc()
        
        # Record duration
        if latency > 0:
            LLM_REQUEST_DURATION.labels(provider=provider, model=model).observe(latency)
        
        # Count tokens
        if prompt_tokens > 0:
            LLM_TOKENS_TOTAL.labels(provider=provider, model=model, type="prompt").inc(prompt_tokens)
        if response_tokens > 0:
            LLM_TOKENS_TOTAL.labels(provider=provider, model=model, type="response").inc(response_tokens)


def serve_metrics(config_path: str, *, port: int = 8001, interval: int = 5) -> None:
    """Serve metrics on HTTP endpoint with comprehensive monitoring."""
    cfg = _load_config(Path(config_path))
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent).resolve()
    start_http_server(port)
    
    print(f"Metrics server started on port {port}")
    print(f"Access metrics at: http://localhost:{port}/metrics")
    
    while True:
        try:
            # Update budget metrics
            snapshots = _snapshot_budget(data_root)
            for label, payload in snapshots.items():
                used = float(payload.get("used", 0) or 0)
                limit = float(payload.get("limit", 0) or 0)
                BUDGET_USED.labels(label=label).set(used)
                BUDGET_LIMIT.labels(label=label).set(limit)
            
            # Update latency metrics
            history = _load_budget_history(data_root)
            latency_map = _latency_by_label(history)
            for label, latency in latency_map.items():
                BUDGET_LATENCY.labels(label=label).set(latency)
            
            # Update LLM metrics
            update_llm_metrics(history)
            
            # Update system health metrics
            update_system_metrics(cfg)
            
            # Update rogue detection metrics
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
            
        except Exception as e:
            print(f"Error updating metrics: {e}")
        
        time.sleep(max(1, interval))


def get_metrics_response() -> tuple[str, str]:
    """Get Prometheus metrics response for HTTP endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
