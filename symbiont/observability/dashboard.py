"""Enhanced observability dashboard with comprehensive metrics and charts.

This module provides a first-class dashboard for monitoring token usage,
system health, and operational metrics.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .metrics import serve_metrics
from ..llm.budget import TokenBudget
from ..initiative.state import get_state_store


class ObservabilityDashboard:
    """Enhanced observability dashboard for Symbiont."""

    def __init__(self, config_path: str = "./configs/config.yaml"):
        self.config_path = config_path
        self.data_root = self._get_data_root()
        self.state_store = get_state_store()

    def _get_data_root(self) -> Path:
        """Get the data root directory from config."""
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            db_path = config.get("db_path", "./data/symbiont.db")
            return Path(db_path).parent
        except Exception:
            return Path("./data")

    def load_budget_history(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load token budget history."""
        history_path = self.data_root / "token_budget" / "history.jsonl"
        if not history_path.exists():
            return []
        
        events = []
        try:
            with history_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:
            pass
        
        return events[-limit:] if limit > 0 else events

    def load_budget_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Load current budget snapshots."""
        token_dir = self.data_root / "token_budget"
        if not token_dir.exists():
            return {}
        
        snapshots = {}
        for path in token_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                label = data.get("label", path.stem)
                snapshots[label] = data
            except Exception:
                continue
        
        return snapshots

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            # Get daemon status
            daemon_status = self.state_store.load_daemon()
            
            # Get swarm workers
            workers = self.state_store.list_swarm_workers()
            
            # Calculate health metrics
            now = int(time.time())
            health = {
                "daemon_running": daemon_status and daemon_status.get("status") == "running",
                "daemon_last_heartbeat": daemon_status.get("last_heartbeat_ts", 0) if daemon_status else 0,
                "active_workers": len([w for w in workers if w.get("status") == "active"]),
                "total_workers": len(workers),
                "system_uptime": now - (daemon_status.get("details", {}).get("started_ts", now) if daemon_status else now),
                "last_check": daemon_status.get("last_check_ts", 0) if daemon_status else 0,
                "last_proposal": daemon_status.get("last_proposal_ts", 0) if daemon_status else 0,
            }
            
            return health
        except Exception:
            return {
                "daemon_running": False,
                "daemon_last_heartbeat": 0,
                "active_workers": 0,
                "total_workers": 0,
                "system_uptime": 0,
                "last_check": 0,
                "last_proposal": 0,
            }

    def render_dashboard(self) -> None:
        """Render the main observability dashboard."""
        st.set_page_config(
            page_title="Symbiont Observability Dashboard",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 Symbiont Observability Dashboard")
        st.markdown("---")
        
        # System Health Overview
        self._render_system_health()
        
        # Token Usage Charts
        self._render_token_usage_charts()
        
        # Performance Metrics
        self._render_performance_metrics()
        
        # System Activity
        self._render_system_activity()
        
        # Configuration
        self._render_configuration()

    def _render_system_health(self) -> None:
        """Render system health overview."""
        st.header("📊 System Health")
        
        health = self.get_system_health()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if health["daemon_running"] else "🔴"
            st.metric(
                "Daemon Status",
                f"{status_color} {'Running' if health['daemon_running'] else 'Stopped'}"
            )
        
        with col2:
            st.metric(
                "Active Workers",
                f"{health['active_workers']}/{health['total_workers']}"
            )
        
        with col3:
            uptime_hours = health["system_uptime"] / 3600
            st.metric(
                "System Uptime",
                f"{uptime_hours:.1f} hours"
            )
        
        with col4:
            last_heartbeat = health["daemon_last_heartbeat"]
            if last_heartbeat > 0:
                heartbeat_age = int(time.time()) - last_heartbeat
                st.metric(
                    "Last Heartbeat",
                    f"{heartbeat_age}s ago"
                )
            else:
                st.metric("Last Heartbeat", "Never")

    def _render_token_usage_charts(self) -> None:
        """Render token usage charts."""
        st.header("💰 Token Usage Analytics")
        
        # Load data
        history = self.load_budget_history()
        snapshots = self.load_budget_snapshots()
        
        if not history and not snapshots:
            st.warning("No token usage data available")
            return
        
        # Token usage over time
        if history:
            self._render_token_usage_over_time(history)
        
        # Current budget status
        if snapshots:
            self._render_budget_status(snapshots)
        
        # Token usage by provider
        if history:
            self._render_usage_by_provider(history)

    def _render_token_usage_over_time(self, history: List[Dict[str, Any]]) -> None:
        """Render token usage over time chart."""
        st.subheader("Token Usage Over Time")
        
        # Prepare data
        df_data = []
        for event in history:
            df_data.append({
                "timestamp": datetime.fromtimestamp(event.get("ts", 0)),
                "label": event.get("label", "unknown"),
                "prompt_tokens": event.get("prompt_tokens", 0),
                "response_tokens": event.get("response_tokens", 0),
                "total_tokens": event.get("prompt_tokens", 0) + event.get("response_tokens", 0),
                "provider": event.get("provider", "unknown"),
                "outcome": event.get("outcome", "unknown"),
                "latency": event.get("latency_seconds", 0)
            })
        
        if not df_data:
            st.warning("No historical data available")
            return
        
        # Create charts
        tab1, tab2, tab3 = st.tabs(["Token Usage", "Latency", "Success Rate"])
        
        with tab1:
            # Token usage by label
            fig = px.line(
                df_data,
                x="timestamp",
                y="total_tokens",
                color="label",
                title="Total Tokens Used Over Time",
                labels={"total_tokens": "Tokens", "timestamp": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Latency over time
            fig = px.scatter(
                df_data,
                x="timestamp",
                y="latency",
                color="provider",
                title="LLM Latency Over Time",
                labels={"latency": "Latency (seconds)", "timestamp": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Success rate over time
            success_data = []
            for event in df_data:
                success_data.append({
                    "timestamp": event["timestamp"],
                    "label": event["label"],
                    "success": 1 if event["outcome"] == "ok" else 0
                })
            
            fig = px.line(
                success_data,
                x="timestamp",
                y="success",
                color="label",
                title="Success Rate Over Time",
                labels={"success": "Success (1=Success, 0=Failure)", "timestamp": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_budget_status(self, snapshots: Dict[str, Dict[str, Any]]) -> None:
        """Render current budget status."""
        st.subheader("Current Budget Status")
        
        # Prepare data for budget gauge
        budget_data = []
        for label, snapshot in snapshots.items():
            used = snapshot.get("used", 0)
            limit = snapshot.get("limit", 0)
            percentage = (used / limit * 100) if limit > 0 else 0
            
            budget_data.append({
                "label": label,
                "used": used,
                "limit": limit,
                "percentage": percentage,
                "remaining": max(0, limit - used)
            })
        
        if not budget_data:
            st.warning("No budget data available")
            return
        
        # Create budget gauges
        cols = st.columns(len(budget_data))
        for i, data in enumerate(budget_data):
            with cols[i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=data["percentage"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{data['label']} Budget"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    def _render_usage_by_provider(self, history: List[Dict[str, Any]]) -> None:
        """Render usage breakdown by provider."""
        st.subheader("Usage by Provider")
        
        # Aggregate data by provider
        provider_data = defaultdict(lambda: {"total_tokens": 0, "requests": 0, "success_rate": 0})
        
        for event in history:
            provider = event.get("provider", "unknown")
            provider_data[provider]["total_tokens"] += event.get("prompt_tokens", 0) + event.get("response_tokens", 0)
            provider_data[provider]["requests"] += 1
            if event.get("outcome") == "ok":
                provider_data[provider]["success_rate"] += 1
        
        # Calculate success rates
        for provider in provider_data:
            if provider_data[provider]["requests"] > 0:
                provider_data[provider]["success_rate"] = (
                    provider_data[provider]["success_rate"] / provider_data[provider]["requests"] * 100
                )
        
        # Create charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Token usage pie chart
            providers = list(provider_data.keys())
            tokens = [provider_data[p]["total_tokens"] for p in providers]
            
            fig = px.pie(
                values=tokens,
                names=providers,
                title="Token Usage by Provider"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate bar chart
            success_rates = [provider_data[p]["success_rate"] for p in providers]
            
            fig = px.bar(
                x=providers,
                y=success_rates,
                title="Success Rate by Provider",
                labels={"x": "Provider", "y": "Success Rate (%)"}
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_performance_metrics(self) -> None:
        """Render performance metrics."""
        st.header("⚡ Performance Metrics")
        
        history = self.load_budget_history()
        if not history:
            st.warning("No performance data available")
            return
        
        # Calculate performance metrics
        latencies = [event.get("latency_seconds", 0) for event in history if event.get("latency_seconds")]
        success_events = [event for event in history if event.get("outcome") == "ok"]
        
        if latencies:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Latency", f"{sum(latencies) / len(latencies):.2f}s")
            
            with col2:
                st.metric("Max Latency", f"{max(latencies):.2f}s")
            
            with col3:
                st.metric("Min Latency", f"{min(latencies):.2f}s")
            
            with col4:
                success_rate = len(success_events) / len(history) * 100 if history else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")

    def _render_system_activity(self) -> None:
        """Render system activity logs."""
        st.header("📋 System Activity")
        
        # Get recent activity from state store
        try:
            workers = self.state_store.list_swarm_workers()
            
            if workers:
                st.subheader("Active Workers")
                for worker in workers:
                    with st.expander(f"Worker: {worker.get('worker_id', 'unknown')}"):
                        st.json(worker)
            else:
                st.info("No active workers")
                
        except Exception as e:
            st.error(f"Error loading system activity: {e}")

    def _render_configuration(self) -> None:
        """Render configuration information."""
        st.header("⚙️ Configuration")
        
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            st.subheader("Current Configuration")
            st.json(config)
            
        except Exception as e:
            st.error(f"Error loading configuration: {e}")


def main():
    """Main entry point for the dashboard."""
    dashboard = ObservabilityDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()