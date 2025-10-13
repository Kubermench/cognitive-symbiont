"""Enhanced budget monitoring with alerts and trend analysis.

This module provides comprehensive budget monitoring including:
- Real-time usage tracking
- Trend analysis and forecasting
- Alert thresholds and notifications
- Historical analysis and reporting
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BudgetAlert:
    """Represents a budget alert condition."""
    
    label: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    threshold_value: float
    current_value: float
    timestamp: float
    alert_type: str  # "usage", "rate", "forecast", "anomaly"


@dataclass
class BudgetTrend:
    """Represents budget usage trends."""
    
    label: str
    hourly_rate: float
    daily_rate: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    confidence: float
    forecast_exhaustion: Optional[datetime] = None


@dataclass
class BudgetMonitorConfig:
    """Configuration for budget monitoring."""
    
    # Alert thresholds (as percentages)
    warning_threshold: float = 75.0
    error_threshold: float = 90.0
    critical_threshold: float = 95.0
    
    # Rate-based alerts (tokens per hour)
    high_usage_rate_threshold: float = 1000.0
    anomaly_detection_enabled: bool = True
    
    # Trend analysis
    trend_window_hours: int = 24
    forecast_enabled: bool = True
    
    # Alert cooldown (seconds)
    alert_cooldown: int = 300


class BudgetMonitor:
    """Enhanced budget monitor with alerts and trend analysis."""
    
    def __init__(
        self,
        data_root: Path,
        config: Optional[BudgetMonitorConfig] = None,
    ):
        self.data_root = data_root
        self.config = config or BudgetMonitorConfig()
        
        # Alert tracking
        self.active_alerts: Dict[str, BudgetAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, float] = {}
        
        # Trend tracking
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.trend_cache: Dict[str, BudgetTrend] = {}
        self.last_trend_update = 0.0
        
        logger.info("BudgetMonitor initialized with data_root: %s", data_root)
    
    def load_current_budgets(self) -> Dict[str, Dict[str, Any]]:
        """Load current budget snapshots."""
        token_dir = self.data_root / "token_budget"
        if not token_dir.exists():
            return {}
        
        snapshots = {}
        for path in token_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                label = payload.get("label") or path.stem
                snapshots[label] = payload
            except Exception as exc:
                logger.debug("Failed to load budget snapshot %s: %s", path, exc)
        
        return snapshots
    
    def load_budget_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load budget history from JSONL file."""
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
        except Exception as exc:
            logger.warning("Failed to load budget history: %s", exc)
            return []
        
        return events[-limit:] if limit > 0 else events
    
    def update_usage_history(self, snapshots: Dict[str, Dict[str, Any]]) -> None:
        """Update usage history for trend analysis."""
        now = time.time()
        
        for label, snapshot in snapshots.items():
            used = snapshot.get("used", 0)
            limit = snapshot.get("limit", 0)
            
            usage_point = {
                "timestamp": now,
                "used": used,
                "limit": limit,
                "usage_percent": (used / limit * 100) if limit > 0 else 0,
            }
            
            self.usage_history[label].append(usage_point)
    
    def analyze_trends(self) -> Dict[str, BudgetTrend]:
        """Analyze usage trends for all budgets."""
        now = time.time()
        
        # Only update trends periodically
        if now - self.last_trend_update < 60:  # Update every minute
            return self.trend_cache
        
        trends = {}
        
        for label, history in self.usage_history.items():
            if len(history) < 2:
                continue
            
            trend = self._analyze_single_trend(label, list(history))
            if trend:
                trends[label] = trend
        
        self.trend_cache = trends
        self.last_trend_update = now
        return trends
    
    def _analyze_single_trend(self, label: str, history: List[Dict[str, Any]]) -> Optional[BudgetTrend]:
        """Analyze trend for a single budget."""
        if len(history) < 2:
            return None
        
        # Calculate usage rates
        recent_points = history[-min(24, len(history)):]  # Last 24 points
        
        if len(recent_points) < 2:
            return None
        
        # Calculate hourly and daily rates
        time_span = recent_points[-1]["timestamp"] - recent_points[0]["timestamp"]
        if time_span <= 0:
            return None
        
        usage_change = recent_points[-1]["used"] - recent_points[0]["used"]
        
        # Convert to hourly rate
        hourly_rate = (usage_change / time_span) * 3600 if time_span > 0 else 0
        daily_rate = hourly_rate * 24
        
        # Determine trend direction
        if abs(hourly_rate) < 1:  # Less than 1 token per hour
            trend_direction = "stable"
            confidence = 0.9
        elif hourly_rate > 0:
            trend_direction = "increasing"
            confidence = min(0.95, abs(hourly_rate) / 100)  # Higher rate = higher confidence
        else:
            trend_direction = "decreasing"
            confidence = min(0.95, abs(hourly_rate) / 100)
        
        # Forecast exhaustion if trend is increasing
        forecast_exhaustion = None
        if trend_direction == "increasing" and hourly_rate > 0:
            current_usage = recent_points[-1]["used"]
            limit = recent_points[-1]["limit"]
            
            if limit > 0 and current_usage < limit:
                remaining = limit - current_usage
                hours_to_exhaustion = remaining / hourly_rate
                
                if hours_to_exhaustion > 0:
                    forecast_exhaustion = datetime.now() + timedelta(hours=hours_to_exhaustion)
        
        return BudgetTrend(
            label=label,
            hourly_rate=hourly_rate,
            daily_rate=daily_rate,
            trend_direction=trend_direction,
            confidence=confidence,
            forecast_exhaustion=forecast_exhaustion,
        )
    
    def check_alerts(self, snapshots: Dict[str, Dict[str, Any]]) -> List[BudgetAlert]:
        """Check for alert conditions and return active alerts."""
        alerts = []
        now = time.time()
        
        for label, snapshot in snapshots.items():
            used = snapshot.get("used", 0)
            limit = snapshot.get("limit", 0)
            
            if limit <= 0:
                continue  # No limit set, no alerts
            
            usage_percent = (used / limit) * 100
            
            # Check usage threshold alerts
            alert_key = f"{label}_usage"
            
            # Determine severity based on usage
            severity = None
            if usage_percent >= self.config.critical_threshold:
                severity = "critical"
            elif usage_percent >= self.config.error_threshold:
                severity = "error"
            elif usage_percent >= self.config.warning_threshold:
                severity = "warning"
            
            if severity:
                # Check cooldown
                last_alert_time = self.last_alert_times.get(alert_key, 0)
                if now - last_alert_time >= self.config.alert_cooldown:
                    alert = BudgetAlert(
                        label=label,
                        severity=severity,
                        message=f"Budget '{label}' usage is {usage_percent:.1f}% ({used:,}/{limit:,} tokens)",
                        threshold_value=getattr(self.config, f"{severity}_threshold"),
                        current_value=usage_percent,
                        timestamp=now,
                        alert_type="usage",
                    )
                    
                    alerts.append(alert)
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    self.last_alert_times[alert_key] = now
            else:
                # Clear alert if usage has dropped
                self.active_alerts.pop(alert_key, None)
        
        # Check rate-based alerts
        if self.config.high_usage_rate_threshold > 0:
            trends = self.analyze_trends()
            
            for label, trend in trends.items():
                if trend.hourly_rate > self.config.high_usage_rate_threshold:
                    alert_key = f"{label}_rate"
                    last_alert_time = self.last_alert_times.get(alert_key, 0)
                    
                    if now - last_alert_time >= self.config.alert_cooldown:
                        alert = BudgetAlert(
                            label=label,
                            severity="warning",
                            message=f"High usage rate for '{label}': {trend.hourly_rate:.1f} tokens/hour",
                            threshold_value=self.config.high_usage_rate_threshold,
                            current_value=trend.hourly_rate,
                            timestamp=now,
                            alert_type="rate",
                        )
                        
                        alerts.append(alert)
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        self.last_alert_times[alert_key] = now
        
        # Check forecast alerts
        if self.config.forecast_enabled:
            trends = self.analyze_trends()
            
            for label, trend in trends.items():
                if trend.forecast_exhaustion:
                    hours_until_exhaustion = (trend.forecast_exhaustion - datetime.now()).total_seconds() / 3600
                    
                    if hours_until_exhaustion <= 24:  # Alert if exhaustion within 24 hours
                        alert_key = f"{label}_forecast"
                        last_alert_time = self.last_alert_times.get(alert_key, 0)
                        
                        if now - last_alert_time >= self.config.alert_cooldown:
                            severity = "error" if hours_until_exhaustion <= 6 else "warning"
                            
                            alert = BudgetAlert(
                                label=label,
                                severity=severity,
                                message=f"Budget '{label}' forecast to exhaust in {hours_until_exhaustion:.1f} hours",
                                threshold_value=24.0,
                                current_value=hours_until_exhaustion,
                                timestamp=now,
                                alert_type="forecast",
                            )
                            
                            alerts.append(alert)
                            self.active_alerts[alert_key] = alert
                            self.alert_history.append(alert)
                            self.last_alert_times[alert_key] = now
        
        return alerts
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a comprehensive summary report."""
        snapshots = self.load_current_budgets()
        trends = self.analyze_trends()
        alerts = self.check_alerts(snapshots)
        
        # Calculate summary statistics
        total_used = sum(s.get("used", 0) for s in snapshots.values())
        total_limit = sum(s.get("limit", 0) for s in snapshots.values() if s.get("limit", 0) > 0)
        
        active_budgets = len(snapshots)
        budgets_with_limits = sum(1 for s in snapshots.values() if s.get("limit", 0) > 0)
        
        # Alert summary
        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.severity] += 1
        
        # Trend summary
        increasing_trends = sum(1 for t in trends.values() if t.trend_direction == "increasing")
        decreasing_trends = sum(1 for t in trends.values() if t.trend_direction == "decreasing")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tokens_used": total_used,
                "total_tokens_limit": total_limit,
                "overall_usage_percent": (total_used / total_limit * 100) if total_limit > 0 else 0,
                "active_budgets": active_budgets,
                "budgets_with_limits": budgets_with_limits,
            },
            "alerts": {
                "total_active": len(alerts),
                "by_severity": dict(alert_counts),
                "details": [
                    {
                        "label": alert.label,
                        "severity": alert.severity,
                        "message": alert.message,
                        "type": alert.alert_type,
                    }
                    for alert in alerts
                ],
            },
            "trends": {
                "total_analyzed": len(trends),
                "increasing": increasing_trends,
                "decreasing": decreasing_trends,
                "stable": len(trends) - increasing_trends - decreasing_trends,
                "details": [
                    {
                        "label": trend.label,
                        "hourly_rate": trend.hourly_rate,
                        "daily_rate": trend.daily_rate,
                        "direction": trend.trend_direction,
                        "confidence": trend.confidence,
                        "forecast_exhaustion": trend.forecast_exhaustion.isoformat() if trend.forecast_exhaustion else None,
                    }
                    for trend in trends.values()
                ],
            },
            "budgets": snapshots,
        }
    
    def save_report(self, report: Optional[Dict[str, Any]] = None) -> Path:
        """Save a budget monitoring report."""
        if report is None:
            report = self.get_summary_report()
        
        reports_dir = self.data_root / "reports" / "budget"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"budget_report_{timestamp}.json"
        
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        
        logger.info("Budget report saved to %s", report_path)
        return report_path
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle and return results."""
        try:
            # Load current data
            snapshots = self.load_current_budgets()
            
            # Update usage history for trend analysis
            self.update_usage_history(snapshots)
            
            # Analyze trends
            trends = self.analyze_trends()
            
            # Check for alerts
            alerts = self.check_alerts(snapshots)
            
            # Log any new alerts
            for alert in alerts:
                alert_key = f"{alert.label}_{alert.alert_type}"
                if alert_key not in [a.label + "_" + a.alert_type for a in self.alert_history]:
                    logger.warning(
                        "Budget alert [%s]: %s",
                        alert.severity.upper(),
                        alert.message,
                    )
            
            # Generate summary
            summary = self.get_summary_report()
            
            return {
                "status": "success",
                "alerts_count": len(alerts),
                "trends_count": len(trends),
                "budgets_count": len(snapshots),
                "summary": summary,
            }
            
        except Exception as exc:
            logger.error("Budget monitoring cycle failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "alerts_count": 0,
                "trends_count": 0,
                "budgets_count": 0,
            }


def create_budget_alerts_config(
    warning_pct: float = 75.0,
    error_pct: float = 90.0,
    critical_pct: float = 95.0,
    high_rate_threshold: float = 1000.0,
) -> BudgetMonitorConfig:
    """Create a budget monitoring configuration with sensible defaults."""
    return BudgetMonitorConfig(
        warning_threshold=warning_pct,
        error_threshold=error_pct,
        critical_threshold=critical_pct,
        high_usage_rate_threshold=high_rate_threshold,
        anomaly_detection_enabled=True,
        trend_window_hours=24,
        forecast_enabled=True,
        alert_cooldown=300,  # 5 minutes
    )


def run_budget_monitoring_daemon(
    data_root: Path,
    config: Optional[BudgetMonitorConfig] = None,
    interval_seconds: int = 60,
) -> None:
    """Run budget monitoring as a daemon process."""
    monitor = BudgetMonitor(data_root, config)
    
    logger.info("Starting budget monitoring daemon (interval: %ds)", interval_seconds)
    
    try:
        while True:
            start_time = time.time()
            
            result = monitor.run_monitoring_cycle()
            
            if result["status"] == "success":
                logger.info(
                    "Budget monitoring cycle completed: %d budgets, %d alerts, %d trends",
                    result["budgets_count"],
                    result["alerts_count"],
                    result["trends_count"],
                )
            else:
                logger.error("Budget monitoring cycle failed: %s", result.get("error"))
            
            # Sleep for the remaining interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("Budget monitoring daemon stopped by user")
    except Exception as exc:
        logger.error("Budget monitoring daemon crashed: %s", exc)
        raise