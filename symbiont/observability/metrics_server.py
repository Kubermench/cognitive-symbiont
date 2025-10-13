"""Enhanced metrics server with comprehensive Prometheus metrics."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from prometheus_client import (
    Counter, Gauge, Histogram, Info, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily
import threading

logger = logging.getLogger(__name__)


class SymbiontMetricsCollector:
    """Custom Prometheus metrics collector for Symbiont."""
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._cache = {}
        self._cache_ttl = 30  # 30 seconds
        self._last_update = 0
    
    def _load_budget_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load token budget history."""
        history_path = self.data_root / "token_budget" / "history.jsonl"
        if not history_path.exists():
            return []
        
        events: List[Dict[str, Any]] = []
        try:
            with history_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Failed to load budget history: %s", e)
            return []
        
        events.sort(key=lambda e: e.get("ts", 0))
        return events[-limit:] if limit > 0 else events
    
    def _load_budget_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Load current budget snapshots."""
        token_dir = self.data_root / "token_budget"
        if not token_dir.exists():
            return {}
        
        snapshots: Dict[str, Dict[str, Any]] = {}
        for path in token_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                label = payload.get("label") or path.stem
                snapshots[label] = payload
            except Exception as e:
                logger.warning("Failed to load snapshot %s: %s", path, e)
                continue
        
        return snapshots
    
    def _load_daemon_state(self) -> Dict[str, Any]:
        """Load daemon state information."""
        state_path = self.data_root / "initiative" / "state.db"
        if not state_path.exists():
            return {}
        
        try:
            import sqlite3
            with sqlite3.connect(str(state_path)) as conn:
                cursor = conn.execute(
                    "SELECT node_id, status, last_check_ts, last_proposal_ts FROM daemon_state ORDER BY updated_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "node_id": row[0],
                        "status": row[1],
                        "last_check_ts": row[2],
                        "last_proposal_ts": row[3],
                    }
        except Exception as e:
            logger.warning("Failed to load daemon state: %s", e)
        
        return {}
    
    def _load_swarm_workers(self) -> List[Dict[str, Any]]:
        """Load swarm worker information."""
        state_path = self.data_root / "initiative" / "state.db"
        if not state_path.exists():
            return []
        
        try:
            import sqlite3
            with sqlite3.connect(str(state_path)) as conn:
                cursor = conn.execute(
                    "SELECT worker_id, node_id, status, active_goal, updated_at FROM swarm_workers WHERE updated_at > ?",
                    (int(time.time()) - 3600,)  # Last hour
                )
                workers = []
                for row in cursor.fetchall():
                    workers.append({
                        "worker_id": row[0],
                        "node_id": row[1],
                        "status": row[2],
                        "active_goal": row[3],
                        "updated_at": row[4],
                    })
                return workers
        except Exception as e:
            logger.warning("Failed to load swarm workers: %s", e)
        
        return []
    
    def _update_cache(self):
        """Update cached data if needed."""
        now = time.time()
        if now - self._last_update < self._cache_ttl:
            return
        
        try:
            self._cache = {
                "budget_history": self._load_budget_history(),
                "budget_snapshots": self._load_budget_snapshots(),
                "daemon_state": self._load_daemon_state(),
                "swarm_workers": self._load_swarm_workers(),
                "timestamp": now,
            }
            self._last_update = now
        except Exception as e:
            logger.error("Failed to update metrics cache: %s", e)
    
    def collect(self):
        """Collect all metrics."""
        self._update_cache()
        
        # Token budget metrics
        yield from self._collect_token_metrics()
        
        # Daemon metrics
        yield from self._collect_daemon_metrics()
        
        # Swarm metrics
        yield from self._collect_swarm_metrics()
        
        # System metrics
        yield from self._collect_system_metrics()
    
    def _collect_token_metrics(self):
        """Collect token-related metrics."""
        snapshots = self._cache.get("budget_snapshots", {})
        history = self._cache.get("budget_history", [])
        
        # Budget usage gauges
        budget_used = GaugeMetricFamily(
            "symbiont_budget_used_tokens",
            "Tokens consumed per budget label",
            labels=["label"]
        )
        budget_limit = GaugeMetricFamily(
            "symbiont_budget_limit_tokens", 
            "Token budget limit per label",
            labels=["label"]
        )
        budget_remaining = GaugeMetricFamily(
            "symbiont_budget_remaining_tokens",
            "Remaining tokens per budget label",
            labels=["label"]
        )
        
        for label, data in snapshots.items():
            used = data.get("used", 0)
            limit = data.get("limit", 0)
            remaining = max(limit - used, 0) if limit > 0 else 0
            
            budget_used.add_metric([label], used)
            budget_limit.add_metric([label], limit)
            budget_remaining.add_metric([label], remaining)
        
        yield budget_used
        yield budget_limit
        yield budget_remaining
        
        # Latency metrics
        latency_histogram = HistogramMetricFamily(
            "symbiont_llm_latency_seconds",
            "LLM request latency distribution",
            labels=["label", "provider", "model"]
        )
        
        # Group events by label, provider, model
        latency_buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for event in history:
            if event.get("outcome") == "ok":
                label = event.get("label", "unknown")
                provider = event.get("provider", "unknown")
                model = event.get("model", "unknown")
                latency = event.get("latency_seconds")
                
                if latency is not None:
                    latency_buckets[label][provider][model].append(latency)
        
        # Create histogram buckets
        for label, providers in latency_buckets.items():
            for provider, models in providers.items():
                for model, latencies in models.items():
                    if latencies:
                        # Create histogram with standard buckets
                        buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
                        bucket_counts = [0] * len(buckets)
                        total_count = len(latencies)
                        
                        for latency in latencies:
                            for i, bucket in enumerate(buckets):
                                if latency <= bucket:
                                    bucket_counts[i] += 1
                                    break
                        
                        # Add cumulative counts
                        cumulative = 0
                        for i, count in enumerate(bucket_counts):
                            cumulative += count
                            latency_histogram.add_metric(
                                [label, provider, model],
                                buckets[i],
                                cumulative
                            )
        
        yield latency_histogram
        
        # Request counters
        request_total = CounterMetricFamily(
            "symbiont_requests_total",
            "Total number of requests",
            labels=["label", "provider", "model", "outcome"]
        )
        
        # Count requests by outcome
        request_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        
        for event in history:
            label = event.get("label", "unknown")
            provider = event.get("provider", "unknown")
            model = event.get("model", "unknown")
            outcome = event.get("outcome", "unknown")
            
            request_counts[label][provider][model][outcome] += 1
        
        for label, providers in request_counts.items():
            for provider, models in providers.items():
                for model, outcomes in models.items():
                    for outcome, count in outcomes.items():
                        request_total.add_metric([label, provider, model, outcome], count)
        
        yield request_total
    
    def _collect_daemon_metrics(self):
        """Collect daemon-related metrics."""
        daemon_state = self._cache.get("daemon_state", {})
        
        # Daemon status
        daemon_status = GaugeMetricFamily(
            "symbiont_daemon_status",
            "Daemon status (1=running, 0=stopped)",
            labels=["node_id"]
        )
        
        if daemon_state:
            node_id = daemon_state.get("node_id", "unknown")
            status = daemon_state.get("status", "unknown")
            is_running = 1 if status == "running" else 0
            daemon_status.add_metric([node_id], is_running)
        
        yield daemon_status
        
        # Last check time
        last_check = GaugeMetricFamily(
            "symbiont_daemon_last_check_timestamp",
            "Timestamp of last daemon check",
            labels=["node_id"]
        )
        
        if daemon_state and daemon_state.get("last_check_ts"):
            node_id = daemon_state.get("node_id", "unknown")
            last_check.add_metric([node_id], daemon_state["last_check_ts"])
        
        yield last_check
    
    def _collect_swarm_metrics(self):
        """Collect swarm-related metrics."""
        workers = self._cache.get("swarm_workers", [])
        
        # Worker count by status
        worker_count = GaugeMetricFamily(
            "symbiont_swarm_workers_total",
            "Number of swarm workers by status",
            labels=["status", "node_id"]
        )
        
        worker_counts = defaultdict(lambda: defaultdict(int))
        for worker in workers:
            status = worker.get("status", "unknown")
            node_id = worker.get("node_id", "unknown")
            worker_counts[status][node_id] += 1
        
        for status, nodes in worker_counts.items():
            for node_id, count in nodes.items():
                worker_count.add_metric([status, node_id], count)
        
        yield worker_count
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # Data directory size
        data_size = GaugeMetricFamily(
            "symbiont_data_directory_size_bytes",
            "Size of data directory in bytes"
        )
        
        try:
            total_size = sum(f.stat().st_size for f in self.data_root.rglob('*') if f.is_file())
            data_size.add_metric([], total_size)
        except Exception:
            data_size.add_metric([], 0)
        
        yield data_size
        
        # File counts
        file_count = GaugeMetricFamily(
            "symbiont_data_files_total",
            "Number of files in data directory",
            labels=["type"]
        )
        
        try:
            json_files = len(list(self.data_root.rglob("*.json")))
            jsonl_files = len(list(self.data_root.rglob("*.jsonl")))
            db_files = len(list(self.data_root.rglob("*.db")))
            
            file_count.add_metric(["json"], json_files)
            file_count.add_metric(["jsonl"], jsonl_files)
            file_count.add_metric(["db"], db_files)
        except Exception:
            file_count.add_metric(["json"], 0)
            file_count.add_metric(["jsonl"], 0)
            file_count.add_metric(["db"], 0)
        
        yield file_count


class MetricsServer:
    """HTTP server for serving Prometheus metrics."""
    
    def __init__(self, data_root: Path, port: int = 8001):
        self.data_root = data_root
        self.port = port
        self.registry = CollectorRegistry()
        self.collector = SymbiontMetricsCollector(data_root)
        self.registry.register(self.collector)
        self._server = None
        self._thread = None
    
    def start(self):
        """Start the metrics server."""
        from prometheus_client import start_http_server
        
        self._server = start_http_server(self.port, registry=self.registry)
        logger.info("Metrics server started on port %d", self.port)
    
    def stop(self):
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Metrics server stopped")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


def create_metrics_endpoint(data_root: Path, port: int = 8001) -> MetricsServer:
    """Create and configure a metrics server."""
    return MetricsServer(data_root, port)


def serve_metrics_standalone(data_root: Path, port: int = 8001):
    """Run metrics server as standalone process."""
    import signal
    import sys
    
    server = create_metrics_endpoint(data_root, port)
    
    def signal_handler(sig, frame):
        logger.info("Received signal %d, shutting down...", sig)
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
        logger.info("Metrics server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Symbiont Metrics Server")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Data root directory")
    parser.add_argument("--port", type=int, default=8001, help="Port to serve metrics on")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    serve_metrics_standalone(args.data_root, args.port)