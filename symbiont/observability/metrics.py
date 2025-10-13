from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

from prometheus_client import Gauge, Counter, Histogram, start_http_server

from symbiont.llm.budget import TokenBudget  # noqa: F401  (ensure dependency accounted for)

# Token budget metrics
BUDGET_USED = Gauge("symbiont_budget_used_tokens", "Tokens consumed per budget label", ["label"])
BUDGET_LIMIT = Gauge("symbiont_budget_limit_tokens", "Token budget limit per label", ["label"])
BUDGET_LATENCY = Gauge("symbiont_budget_latency_seconds", "Average LLM latency per budget label", ["label"])
BUDGET_USAGE_PERCENT = Gauge("symbiont_budget_usage_percent", "Budget usage percentage", ["label"])

# LLM performance metrics
LLM_REQUESTS_TOTAL = Counter("symbiont_llm_requests_total", "Total LLM requests", ["provider", "model", "outcome"])
LLM_REQUEST_DURATION = Histogram("symbiont_llm_request_duration_seconds", "LLM request duration", ["provider", "model"])
LLM_TOKENS_CONSUMED = Counter("symbiont_llm_tokens_consumed_total", "Total tokens consumed", ["provider", "model", "type"])

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge("symbiont_circuit_breaker_state", "Circuit breaker state (0=closed, 1=half-open, 2=open)", ["name"])
CIRCUIT_BREAKER_FAILURES = Counter("symbiont_circuit_breaker_failures_total", "Circuit breaker failure count", ["name"])
CIRCUIT_BREAKER_SUCCESS_RATE = Gauge("symbiont_circuit_breaker_success_rate", "Circuit breaker success rate", ["name"])

# Daemon metrics
DAEMON_STATUS = Gauge("symbiont_daemon_status", "Daemon status (1=running, 0=stopped)", ["node_id"])
DAEMON_UPTIME_SECONDS = Gauge("symbiont_daemon_uptime_seconds", "Daemon uptime in seconds", ["node_id"])
DAEMON_LAST_ACTIVITY = Gauge("symbiont_daemon_last_activity_timestamp", "Last daemon activity timestamp", ["node_id", "activity_type"])

# System resource metrics
SYSTEM_LOAD_AVERAGE = Gauge("symbiont_system_load_average", "System load average", ["node_id"])
SYSTEM_MEMORY_USAGE = Gauge("symbiont_system_memory_usage_percent", "System memory usage percentage", ["node_id"])

# Rogue score metrics
ROGUE_SCORE = Gauge("symbiont_rogue_score", "Rogue baseline and forecast metrics", ["metric"])
ROGUE_ALERT = Gauge("symbiont_rogue_alert", "1 when rogue alert triggered")

# Initiative metrics
INITIATIVE_PROPOSALS_TOTAL = Counter("symbiont_initiative_proposals_total", "Total initiative proposals", ["node_id", "outcome"])
INITIATIVE_CYCLE_DURATION = Histogram("symbiont_initiative_cycle_duration_seconds", "Initiative cycle duration", ["node_id"])


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
    """Serve Prometheus metrics with comprehensive Symbiont observability."""
    cfg = _load_config(Path(config_path))
    data_root = Path(cfg.get("data_root") or Path(cfg.get("db_path", "./data/symbiont.db")).parent).resolve()
    
    print(f"Starting Symbiont metrics server on port {port}")
    print(f"Data root: {data_root}")
    print(f"Update interval: {interval}s")
    
    start_http_server(port)
    
    while True:
        try:
            _update_all_metrics(cfg, data_root)
        except Exception as exc:
            print(f"Error updating metrics: {exc}")
        
        time.sleep(max(1, interval))


def _update_all_metrics(cfg: Dict[str, Any], data_root: Path) -> None:
    """Update all Prometheus metrics."""
    
    # Update budget metrics
    _update_budget_metrics(data_root)
    
    # Update LLM performance metrics
    _update_llm_metrics(data_root)
    
    # Update circuit breaker metrics
    _update_circuit_breaker_metrics()
    
    # Update daemon metrics
    _update_daemon_metrics(cfg)
    
    # Update rogue score metrics
    _update_rogue_metrics(data_root)


def _update_budget_metrics(data_root: Path) -> None:
    """Update token budget related metrics."""
    snapshots = _snapshot_budget(data_root)
    history = _load_budget_history(data_root)
    
    # Current budget status
    for label, payload in snapshots.items():
        used = float(payload.get("used", 0) or 0)
        limit = float(payload.get("limit", 0) or 0)
        
        BUDGET_USED.labels(label=label).set(used)
        BUDGET_LIMIT.labels(label=label).set(limit)
        
        # Usage percentage
        if limit > 0:
            usage_pct = (used / limit) * 100
            BUDGET_USAGE_PERCENT.labels(label=label).set(usage_pct)
    
    # Latency metrics from history
    latency_map = _latency_by_label(history)
    for label, latency in latency_map.items():
        BUDGET_LATENCY.labels(label=label).set(latency)


def _update_llm_metrics(data_root: Path) -> None:
    """Update LLM performance metrics from history."""
    history = _load_budget_history(data_root, limit=1000)
    
    # Count requests by provider/model/outcome
    request_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    token_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for event in history:
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        outcome = event.get("outcome", "unknown")
        
        # Count requests
        request_counts[provider][model][outcome] += 1
        
        # Count tokens
        prompt_tokens = event.get("prompt_tokens", 0)
        response_tokens = event.get("response_tokens", 0)
        
        token_counts[provider][model]["prompt"] += prompt_tokens
        token_counts[provider][model]["response"] += response_tokens
        
        # Record latency
        latency = event.get("latency_seconds")
        if latency is not None:
            try:
                LLM_REQUEST_DURATION.labels(provider=provider, model=model).observe(float(latency))
            except Exception:
                pass
    
    # Update counters (note: Prometheus counters should only increase, so we set to cumulative values)
    for provider in request_counts:
        for model in request_counts[provider]:
            for outcome in request_counts[provider][model]:
                count = request_counts[provider][model][outcome]
                LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, outcome=outcome)._value._value = count
    
    for provider in token_counts:
        for model in token_counts[provider]:
            for token_type in token_counts[provider][model]:
                count = token_counts[provider][model][token_type]
                LLM_TOKENS_CONSUMED.labels(provider=provider, model=model, type=token_type)._value._value = count


def _update_circuit_breaker_metrics() -> None:
    """Update circuit breaker metrics."""
    try:
        from ..tools.retry_utils import list_circuit_breakers
        
        circuit_breakers = list_circuit_breakers()
        
        for name, metrics in circuit_breakers.items():
            state = metrics.get("state", "closed")
            success_rate = metrics.get("success_rate", 0.0)
            failure_count = metrics.get("failure_count", 0)
            
            # Map state to numeric value
            state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
            
            CIRCUIT_BREAKER_STATE.labels(name=name).set(state_value)
            CIRCUIT_BREAKER_SUCCESS_RATE.labels(name=name).set(success_rate)
            CIRCUIT_BREAKER_FAILURES.labels(name=name)._value._value = failure_count
            
    except Exception:
        # Circuit breaker metrics not available
        pass


def _update_daemon_metrics(cfg: Dict[str, Any]) -> None:
    """Update daemon status metrics."""
    try:
        from ..initiative.daemon import get_status
        
        status = get_status(cfg)
        node_id = status.get("node_id", "unknown")
        
        # Daemon running status
        running = 1.0 if status.get("daemon_running", False) else 0.0
        DAEMON_STATUS.labels(node_id=node_id).set(running)
        
        # Uptime
        started_ts = status.get("daemon_started_ts", 0)
        if started_ts > 0:
            uptime = time.time() - started_ts
            DAEMON_UPTIME_SECONDS.labels(node_id=node_id).set(uptime)
        
        # Last activity timestamps
        last_check = status.get("last_check_ts", 0)
        last_proposal = status.get("last_proposal_ts", 0)
        
        if last_check > 0:
            DAEMON_LAST_ACTIVITY.labels(node_id=node_id, activity_type="check").set(last_check)
        if last_proposal > 0:
            DAEMON_LAST_ACTIVITY.labels(node_id=node_id, activity_type="proposal").set(last_proposal)
        
        # System resources
        load_avg = status.get("load_avg", 0.0)
        memory_usage = status.get("memory_usage", 0.0)
        
        if load_avg > 0:
            SYSTEM_LOAD_AVERAGE.labels(node_id=node_id).set(load_avg)
        if memory_usage > 0:
            SYSTEM_MEMORY_USAGE.labels(node_id=node_id).set(memory_usage * 100)  # Convert to percentage
            
    except Exception:
        # Daemon metrics not available
        pass


def _update_rogue_metrics(data_root: Path) -> None:
    """Update rogue score metrics."""
    governance = _latest_governance_snapshot(data_root)
    
    if governance:
        baseline = float(governance.get("rogue_baseline", 0.0) or 0.0)
        forecast_values = [baseline]
        
        for value in governance.get("rogue_forecast", []):
            try:
                forecast_values.append(float(value))
            except (TypeError, ValueError):
                continue
        
        forecast_max = max(forecast_values) if forecast_values else baseline
        threshold = float(governance.get("alert_threshold", 0.0) or 0.0)
        
        ROGUE_SCORE.labels(metric="baseline").set(baseline)
        ROGUE_SCORE.labels(metric="forecast_max").set(forecast_max)
        ROGUE_SCORE.labels(metric="threshold").set(threshold)
        ROGUE_ALERT.set(1.0 if governance.get("alert") else 0.0)
    else:
        ROGUE_ALERT.set(0.0)
