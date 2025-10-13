"""Comprehensive metrics collection system for Symbiont.

This module provides centralized metrics collection that feeds into both
the Streamlit dashboard and Prometheus endpoints, ensuring consistent
observability across all monitoring interfaces.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of all Symbiont metrics at a point in time."""
    
    timestamp: float
    
    # Token budget metrics
    budget_snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    budget_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # LLM performance metrics
    llm_latency_stats: Dict[str, float] = field(default_factory=dict)
    llm_success_rates: Dict[str, float] = field(default_factory=dict)
    llm_request_counts: Dict[str, int] = field(default_factory=dict)
    
    # Circuit breaker metrics
    circuit_breaker_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Daemon metrics
    daemon_status: Dict[str, Any] = field(default_factory=dict)
    cluster_status: Dict[str, Any] = field(default_factory=dict)
    
    # Rogue score metrics
    rogue_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # System metrics
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Centralized metrics collector for Symbiont observability."""
    
    def __init__(self, data_root: Path, config: Optional[Dict[str, Any]] = None):
        self.data_root = data_root
        self.config = config or {}
        
        # Metrics storage
        self.snapshots: deque[MetricsSnapshot] = deque(maxlen=1000)
        self.last_collection_time = 0.0
        
        # Performance tracking
        self.collection_durations: deque[float] = deque(maxlen=100)
        
        logger.info("MetricsCollector initialized with data_root: %s", data_root)
    
    def collect_all_metrics(self) -> MetricsSnapshot:
        """Collect all available metrics and return a snapshot."""
        start_time = time.time()
        
        try:
            snapshot = MetricsSnapshot(timestamp=start_time)
            
            # Collect budget metrics
            snapshot.budget_snapshots = self._collect_budget_snapshots()
            snapshot.budget_history = self._collect_budget_history()
            
            # Collect LLM performance metrics
            snapshot.llm_latency_stats = self._collect_llm_latency_stats(snapshot.budget_history)
            snapshot.llm_success_rates = self._collect_llm_success_rates(snapshot.budget_history)
            snapshot.llm_request_counts = self._collect_llm_request_counts(snapshot.budget_history)
            
            # Collect circuit breaker metrics
            snapshot.circuit_breaker_states = self._collect_circuit_breaker_metrics()
            
            # Collect daemon metrics
            snapshot.daemon_status = self._collect_daemon_metrics()
            snapshot.cluster_status = self._collect_cluster_metrics()
            
            # Collect rogue score metrics
            snapshot.rogue_metrics = self._collect_rogue_metrics()
            
            # Collect system metrics
            snapshot.system_metrics = self._collect_system_metrics()
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self.last_collection_time = start_time
            
            # Track collection performance
            collection_duration = time.time() - start_time
            self.collection_durations.append(collection_duration)
            
            logger.debug("Metrics collection completed in %.3fs", collection_duration)
            
            return snapshot
            
        except Exception as exc:
            logger.error("Failed to collect metrics: %s", exc)
            # Return empty snapshot on failure
            return MetricsSnapshot(timestamp=start_time)
    
    def _collect_budget_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Collect current budget snapshots."""
        token_dir = self.data_root / "token_budget"
        if not token_dir.exists():
            return {}
        
        snapshots = {}
        for path in token_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                label = payload.get("label") or path.stem
                snapshots[label] = payload
            except Exception:
                continue
        
        return snapshots
    
    def _collect_budget_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Collect budget history events."""
        history_path = self.data_root / "token_budget" / "history.jsonl"
        if not history_path.exists():
            return []
        
        events = []
        try:
            with history_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        
        return events[-limit:] if limit > 0 else events
    
    def _collect_llm_latency_stats(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Collect LLM latency statistics by label."""
        latency_buckets: Dict[str, List[float]] = defaultdict(list)
        
        for event in history:
            label = event.get("label")
            latency = event.get("latency_seconds")
            
            if label and latency is not None:
                try:
                    latency_buckets[label].append(float(latency))
                except (TypeError, ValueError):
                    continue
        
        # Calculate statistics
        stats = {}
        for label, latencies in latency_buckets.items():
            if latencies:
                stats[f"{label}_mean"] = statistics.mean(latencies)
                stats[f"{label}_median"] = statistics.median(latencies)
                stats[f"{label}_p95"] = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies)
                stats[f"{label}_min"] = min(latencies)
                stats[f"{label}_max"] = max(latencies)
        
        return stats
    
    def _collect_llm_success_rates(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Collect LLM success rates by provider/model."""
        outcome_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for event in history:
            provider = event.get("provider", "unknown")
            model = event.get("model", "unknown")
            outcome = event.get("outcome", "unknown")
            
            key = f"{provider}_{model}"
            outcome_counts[key][outcome] += 1
        
        success_rates = {}
        for key, outcomes in outcome_counts.items():
            total = sum(outcomes.values())
            successes = outcomes.get("ok", 0)
            
            if total > 0:
                success_rates[key] = successes / total
        
        return success_rates
    
    def _collect_llm_request_counts(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Collect LLM request counts by various dimensions."""
        counts = defaultdict(int)
        
        for event in history:
            provider = event.get("provider", "unknown")
            model = event.get("model", "unknown")
            outcome = event.get("outcome", "unknown")
            label = event.get("label", "unknown")
            
            counts[f"total"] += 1
            counts[f"provider_{provider}"] += 1
            counts[f"model_{model}"] += 1
            counts[f"outcome_{outcome}"] += 1
            counts[f"label_{label}"] += 1
            counts[f"{provider}_{model}"] += 1
        
        return dict(counts)
    
    def _collect_circuit_breaker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect circuit breaker metrics."""
        try:
            from ..tools.retry_utils import list_circuit_breakers
            return list_circuit_breakers()
        except Exception:
            return {}
    
    def _collect_daemon_metrics(self) -> Dict[str, Any]:
        """Collect daemon status metrics."""
        try:
            from ..initiative.daemon import get_status
            return get_status(self.config)
        except Exception:
            return {}
    
    def _collect_cluster_metrics(self) -> Dict[str, Any]:
        """Collect cluster-wide metrics."""
        try:
            from ..initiative.heartbeat import get_heartbeat_manager
            
            heartbeat_manager = get_heartbeat_manager(self.config)
            return heartbeat_manager.get_cluster_status()
        except Exception:
            return {}
    
    def _collect_rogue_metrics(self) -> Dict[str, Any]:
        """Collect rogue score and governance metrics."""
        try:
            from .metrics import _latest_governance_snapshot
            return _latest_governance_snapshot(self.data_root)
        except Exception:
            return {}
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        metrics = {}
        
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            metrics.update({
                "cpu_percent": cpu_percent,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
                "cpu_count": psutil.cpu_count(),
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.update({
                "memory_total_bytes": memory.total,
                "memory_used_bytes": memory.used,
                "memory_available_bytes": memory.available,
                "memory_percent": memory.percent,
            })
            
            # Disk metrics
            disk = psutil.disk_usage(str(self.data_root))
            metrics.update({
                "disk_total_bytes": disk.total,
                "disk_used_bytes": disk.used,
                "disk_free_bytes": disk.free,
                "disk_percent": (disk.used / disk.total) * 100,
            })
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics.update({
                "process_memory_rss": process_memory.rss,
                "process_memory_vms": process_memory.vms,
                "process_cpu_percent": process.cpu_percent(),
                "process_threads": process.num_threads(),
            })
            
        except ImportError:
            # psutil not available
            metrics["psutil_available"] = False
        except Exception as exc:
            logger.debug("Failed to collect system metrics: %s", exc)
            metrics["collection_error"] = str(exc)
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the metrics collector itself."""
        if not self.collection_durations:
            return {}
        
        durations = list(self.collection_durations)
        
        return {
            "collection_count": len(durations),
            "avg_duration": statistics.mean(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "last_collection": self.last_collection_time,
            "snapshots_stored": len(self.snapshots),
        }
    
    def export_metrics_json(self, include_history: bool = False) -> Dict[str, Any]:
        """Export all metrics as JSON for external consumption."""
        latest_snapshot = self.snapshots[-1] if self.snapshots else None
        
        if not latest_snapshot:
            return {"error": "No metrics available"}
        
        export_data = {
            "timestamp": latest_snapshot.timestamp,
            "collection_performance": self.get_performance_summary(),
            "budget": {
                "snapshots": latest_snapshot.budget_snapshots,
                "history_count": len(latest_snapshot.budget_history),
            },
            "llm": {
                "latency_stats": latest_snapshot.llm_latency_stats,
                "success_rates": latest_snapshot.llm_success_rates,
                "request_counts": latest_snapshot.llm_request_counts,
            },
            "circuit_breakers": latest_snapshot.circuit_breaker_states,
            "daemon": latest_snapshot.daemon_status,
            "cluster": latest_snapshot.cluster_status,
            "rogue": latest_snapshot.rogue_metrics,
            "system": latest_snapshot.system_metrics,
        }
        
        if include_history:
            export_data["history"] = {
                "snapshots_count": len(self.snapshots),
                "budget_events": latest_snapshot.budget_history,
            }
        
        return export_data


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(
    data_root: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    
    if _metrics_collector is None:
        if data_root is None:
            data_root = Path("data")
        
        _metrics_collector = MetricsCollector(data_root, config)
    
    return _metrics_collector


def collect_and_export_metrics(
    data_root: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    export_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect all metrics and optionally export to file."""
    collector = get_metrics_collector(data_root, config)
    snapshot = collector.collect_all_metrics()
    
    export_data = collector.export_metrics_json(include_history=True)
    
    if export_path:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        logger.info("Metrics exported to %s", export_path)
    
    return export_data