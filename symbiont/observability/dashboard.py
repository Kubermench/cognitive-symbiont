"""Enhanced observability dashboard for Symbiont with real-time metrics.

This module provides a comprehensive Streamlit-based dashboard for monitoring:
- Token budget usage and trends
- LLM latency and performance metrics
- Circuit breaker status and health
- System resource utilization
- Rogue score tracking and alerts
- Initiative daemon status
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..llm.budget import TokenBudget
from ..initiative.daemon import get_status
from ..tools.retry_utils import list_circuit_breakers
from .metrics import _load_budget_history, _snapshot_budget, _latest_governance_snapshot


class SymbiontDashboard:
    """Main dashboard class for Symbiont observability."""
    
    def __init__(self, config_path: Optional[str] = None, data_root: Optional[Path] = None):
        self.config_path = config_path
        self.data_root = data_root or Path("data")
        self.refresh_interval = 5  # seconds
        
        # Initialize session state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = 0
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path:
            return {}
        
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    
    def get_budget_data(self) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Get current budget snapshots and history."""
        snapshots = _snapshot_budget(self.data_root)
        history = _load_budget_history(self.data_root, limit=1000)
        return snapshots, history
    
    def get_circuit_breaker_data(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status data."""
        try:
            return list_circuit_breakers()
        except Exception:
            return {}
    
    def get_daemon_status(self) -> Dict[str, Any]:
        """Get initiative daemon status."""
        try:
            config = self.load_config()
            return get_status(config)
        except Exception:
            return {}
    
    def render_header(self):
        """Render dashboard header."""
        st.set_page_config(
            page_title="Symbiont Observatory",
            page_icon="🔭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔭 Symbiont Observatory")
        st.markdown("Real-time monitoring and observability for your Symbiont instance")
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("🔄 Refresh Now"):
                st.session_state.last_refresh = 0
                st.experimental_rerun()
        
        with col2:
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh", 
                value=st.session_state.auto_refresh
            )
        
        with col3:
            if st.session_state.auto_refresh:
                time_since_refresh = time.time() - st.session_state.last_refresh
                if time_since_refresh > self.refresh_interval:
                    st.session_state.last_refresh = time.time()
                    st.experimental_rerun()
                
                next_refresh = max(0, self.refresh_interval - time_since_refresh)
                st.info(f"⏱️ Next auto-refresh in {next_refresh:.1f}s")
    
    def render_overview_metrics(self):
        """Render overview metrics cards."""
        st.subheader("📊 System Overview")
        
        # Get data
        snapshots, history = self.get_budget_data()
        circuit_breakers = self.get_circuit_breaker_data()
        daemon_status = self.get_daemon_status()
        
        # Calculate summary metrics
        total_tokens_used = sum(s.get("used", 0) for s in snapshots.values())
        total_tokens_limit = sum(s.get("limit", 0) for s in snapshots.values() if s.get("limit", 0) > 0)
        
        active_budgets = len(snapshots)
        
        circuit_breaker_issues = sum(
            1 for cb in circuit_breakers.values() 
            if cb.get("state") != "closed"
        )
        
        daemon_running = daemon_status.get("daemon_running", False)
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Tokens Used",
                value=f"{total_tokens_used:,}",
                delta=None
            )
        
        with col2:
            if total_tokens_limit > 0:
                usage_pct = (total_tokens_used / total_tokens_limit) * 100
                st.metric(
                    label="Budget Usage",
                    value=f"{usage_pct:.1f}%",
                    delta=f"{total_tokens_used:,}/{total_tokens_limit:,}"
                )
            else:
                st.metric(
                    label="Budget Usage",
                    value="Unlimited",
                    delta=f"{total_tokens_used:,} tokens"
                )
        
        with col3:
            st.metric(
                label="Active Budgets",
                value=active_budgets,
                delta=None
            )
        
        with col4:
            cb_color = "🔴" if circuit_breaker_issues > 0 else "🟢"
            st.metric(
                label="Circuit Breakers",
                value=f"{cb_color} {len(circuit_breakers) - circuit_breaker_issues}/{len(circuit_breakers)}",
                delta=f"{circuit_breaker_issues} issues" if circuit_breaker_issues > 0 else "All healthy"
            )
        
        with col5:
            daemon_color = "🟢" if daemon_running else "🔴"
            st.metric(
                label="Daemon Status",
                value=f"{daemon_color} {'Running' if daemon_running else 'Stopped'}",
                delta=None
            )
    
    def render_token_budget_section(self):
        """Render token budget monitoring section."""
        st.subheader("💰 Token Budget Monitoring")
        
        snapshots, history = self.get_budget_data()
        
        if not snapshots and not history:
            st.info("No token budget data available yet. Start using Symbiont to see metrics here.")
            return
        
        # Budget overview table
        if snapshots:
            st.write("**Current Budget Status**")
            
            budget_data = []
            for label, snapshot in snapshots.items():
                used = snapshot.get("used", 0)
                limit = snapshot.get("limit", 0)
                
                budget_data.append({
                    "Label": label,
                    "Used": f"{used:,}",
                    "Limit": f"{limit:,}" if limit > 0 else "Unlimited",
                    "Usage %": f"{(used/limit)*100:.1f}%" if limit > 0 else "N/A",
                    "Remaining": f"{max(0, limit-used):,}" if limit > 0 else "Unlimited"
                })
            
            df_budgets = pd.DataFrame(budget_data)
            st.dataframe(df_budgets, use_container_width=True)
        
        # Historical usage charts
        if history:
            st.write("**Usage Trends**")
            
            # Convert history to DataFrame
            df_history = pd.DataFrame(history)
            df_history['timestamp'] = pd.to_datetime(df_history['ts'], unit='s')
            
            # Token usage over time
            fig_usage = px.line(
                df_history,
                x='timestamp',
                y='prompt_tokens',
                color='label',
                title="Token Usage Over Time",
                labels={'prompt_tokens': 'Tokens', 'timestamp': 'Time'}
            )
            fig_usage.update_layout(height=400)
            st.plotly_chart(fig_usage, use_container_width=True)
            
            # Latency trends
            if 'latency_seconds' in df_history.columns:
                fig_latency = px.line(
                    df_history,
                    x='timestamp',
                    y='latency_seconds',
                    color='label',
                    title="LLM Latency Trends",
                    labels={'latency_seconds': 'Latency (seconds)', 'timestamp': 'Time'}
                )
                fig_latency.update_layout(height=400)
                st.plotly_chart(fig_latency, use_container_width=True)
            
            # Provider/model breakdown
            if 'provider' in df_history.columns:
                provider_usage = df_history.groupby(['provider', 'model'])['prompt_tokens'].sum().reset_index()
                
                fig_providers = px.pie(
                    provider_usage,
                    values='prompt_tokens',
                    names='provider',
                    title="Token Usage by Provider"
                )
                st.plotly_chart(fig_providers, use_container_width=True)
    
    def render_circuit_breaker_section(self):
        """Render circuit breaker monitoring section."""
        st.subheader("⚡ Circuit Breaker Status")
        
        circuit_breakers = self.get_circuit_breaker_data()
        
        if not circuit_breakers:
            st.info("No circuit breaker data available.")
            return
        
        # Circuit breaker status cards
        for name, metrics in circuit_breakers.items():
            state = metrics.get("state", "unknown")
            success_rate = metrics.get("success_rate", 0) * 100
            total_calls = metrics.get("total_calls", 0)
            failure_count = metrics.get("failure_count", 0)
            
            # Color coding based on state
            if state == "closed":
                state_color = "🟢"
                state_text = "Healthy"
            elif state == "half_open":
                state_color = "🟡"
                state_text = "Recovering"
            else:  # open
                state_color = "🔴"
                state_text = "Failed"
            
            with st.expander(f"{state_color} {name} - {state_text}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col2:
                    st.metric("Total Calls", f"{total_calls:,}")
                
                with col3:
                    st.metric("Failures", f"{failure_count:,}")
                
                with col4:
                    st.metric("State", state.title())
                
                # Recent activity if available
                recent_success_rate = metrics.get("recent_success_rate", 0) * 100
                if recent_success_rate != success_rate:
                    st.info(f"Recent success rate: {recent_success_rate:.1f}%")
    
    def render_daemon_status_section(self):
        """Render daemon status monitoring section."""
        st.subheader("🤖 Initiative Daemon Status")
        
        daemon_status = self.get_daemon_status()
        
        if not daemon_status:
            st.warning("Unable to retrieve daemon status.")
            return
        
        # Main status
        running = daemon_status.get("daemon_running", False)
        status_color = "🟢" if running else "🔴"
        status_text = "Running" if running else "Stopped"
        
        st.write(f"**Status**: {status_color} {status_text}")
        
        # Detailed information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**System Information**")
            
            node_id = daemon_status.get("node_id", "Unknown")
            pid = daemon_status.get("daemon_pid", 0)
            started_ts = daemon_status.get("daemon_started_ts", 0)
            
            st.write(f"- **Node ID**: {node_id}")
            st.write(f"- **PID**: {pid}")
            
            if started_ts > 0:
                started_time = datetime.fromtimestamp(started_ts)
                uptime = datetime.now() - started_time
                st.write(f"- **Started**: {started_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- **Uptime**: {uptime}")
            
            # System resources
            load_avg = daemon_status.get("load_avg", 0)
            memory_usage = daemon_status.get("memory_usage", 0)
            
            if load_avg > 0:
                st.write(f"- **Load Average**: {load_avg:.2f}")
            if memory_usage > 0:
                st.write(f"- **Memory Usage**: {memory_usage:.1%}")
        
        with col2:
            st.write("**Activity**")
            
            last_check = daemon_status.get("last_check_ts", 0)
            last_proposal = daemon_status.get("last_proposal_ts", 0)
            
            if last_check > 0:
                check_time = datetime.fromtimestamp(last_check)
                check_ago = datetime.now() - check_time
                st.write(f"- **Last Check**: {check_ago} ago")
            
            if last_proposal > 0:
                proposal_time = datetime.fromtimestamp(last_proposal)
                proposal_ago = datetime.now() - proposal_time
                st.write(f"- **Last Proposal**: {proposal_ago} ago")
            
            # Cluster information
            cluster_info = daemon_status.get("cluster", {})
            if cluster_info:
                cluster_size = cluster_info.get("cluster_size", 0)
                active_workers = len(cluster_info.get("active_workers", []))
                
                st.write(f"- **Cluster Size**: {cluster_size} nodes")
                st.write(f"- **Active Workers**: {active_workers}")
    
    def render_rogue_monitoring_section(self):
        """Render rogue score monitoring section."""
        st.subheader("🚨 Rogue Score Monitoring")
        
        governance = _latest_governance_snapshot(self.data_root)
        
        if not governance:
            st.info("No governance/rogue score data available yet.")
            return
        
        # Current rogue metrics
        baseline = governance.get("rogue_baseline", 0.0)
        forecast = governance.get("rogue_forecast", [])
        threshold = governance.get("alert_threshold", 0.6)
        alert_active = governance.get("alert", False)
        
        # Alert status
        if alert_active:
            st.error(f"🚨 **ROGUE ALERT ACTIVE** - Baseline score {baseline:.3f} exceeds threshold {threshold:.3f}")
        else:
            st.success(f"✅ Rogue score within safe limits ({baseline:.3f} < {threshold:.3f})")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Rogue Baseline",
                value=f"{baseline:.3f}",
                delta=f"Threshold: {threshold:.3f}"
            )
        
        with col2:
            if forecast:
                forecast_max = max(forecast)
                st.metric(
                    label="Forecast Peak",
                    value=f"{forecast_max:.3f}",
                    delta=f"{len(forecast)} points"
                )
        
        with col3:
            alert_color = "🔴" if alert_active else "🟢"
            st.metric(
                label="Alert Status",
                value=f"{alert_color} {'Active' if alert_active else 'Clear'}",
                delta=None
            )
        
        # Forecast visualization
        if forecast:
            fig_rogue = go.Figure()
            
            # Baseline line
            fig_rogue.add_hline(
                y=baseline,
                line_dash="solid",
                line_color="blue",
                annotation_text=f"Baseline: {baseline:.3f}"
            )
            
            # Threshold line
            fig_rogue.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Alert Threshold: {threshold:.3f}"
            )
            
            # Forecast line
            fig_rogue.add_trace(go.Scatter(
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orange', width=2)
            ))
            
            fig_rogue.update_layout(
                title="Rogue Score Forecast",
                yaxis_title="Rogue Score",
                xaxis_title="Time Steps",
                height=400
            )
            
            st.plotly_chart(fig_rogue, use_container_width=True)
    
    def render_alerts_section(self):
        """Render alerts and notifications section."""
        st.subheader("🔔 Alerts & Notifications")
        
        alerts = []
        
        # Check for various alert conditions
        snapshots, _ = self.get_budget_data()
        circuit_breakers = self.get_circuit_breaker_data()
        daemon_status = self.get_daemon_status()
        governance = _latest_governance_snapshot(self.data_root)
        
        # Budget alerts
        for label, snapshot in snapshots.items():
            used = snapshot.get("used", 0)
            limit = snapshot.get("limit", 0)
            
            if limit > 0:
                usage_pct = (used / limit) * 100
                if usage_pct > 90:
                    alerts.append({
                        "severity": "error",
                        "type": "Budget",
                        "message": f"Budget '{label}' is {usage_pct:.1f}% full ({used:,}/{limit:,} tokens)"
                    })
                elif usage_pct > 75:
                    alerts.append({
                        "severity": "warning",
                        "type": "Budget",
                        "message": f"Budget '{label}' is {usage_pct:.1f}% full ({used:,}/{limit:,} tokens)"
                    })
        
        # Circuit breaker alerts
        for name, metrics in circuit_breakers.items():
            state = metrics.get("state", "closed")
            if state != "closed":
                success_rate = metrics.get("success_rate", 0) * 100
                alerts.append({
                    "severity": "error" if state == "open" else "warning",
                    "type": "Circuit Breaker",
                    "message": f"Circuit breaker '{name}' is {state} (success rate: {success_rate:.1f}%)"
                })
        
        # Daemon alerts
        if not daemon_status.get("daemon_running", False):
            alerts.append({
                "severity": "error",
                "type": "Daemon",
                "message": "Initiative daemon is not running"
            })
        
        # Rogue score alerts
        if governance and governance.get("alert", False):
            baseline = governance.get("rogue_baseline", 0.0)
            threshold = governance.get("alert_threshold", 0.6)
            alerts.append({
                "severity": "error",
                "type": "Rogue Score",
                "message": f"Rogue baseline {baseline:.3f} exceeds threshold {threshold:.3f}"
            })
        
        # Display alerts
        if not alerts:
            st.success("🟢 All systems healthy - no active alerts")
        else:
            for alert in alerts:
                severity = alert["severity"]
                alert_type = alert["type"]
                message = alert["message"]
                
                if severity == "error":
                    st.error(f"🔴 **{alert_type}**: {message}")
                elif severity == "warning":
                    st.warning(f"🟡 **{alert_type}**: {message}")
                else:
                    st.info(f"🔵 **{alert_type}**: {message}")
    
    def render_sidebar(self):
        """Render sidebar with configuration and controls."""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # Data source configuration
            st.write("**Data Source**")
            data_root_input = st.text_input(
                "Data Root Path",
                value=str(self.data_root),
                help="Path to Symbiont data directory"
            )
            
            if data_root_input != str(self.data_root):
                self.data_root = Path(data_root_input)
            
            # Refresh settings
            st.write("**Refresh Settings**")
            new_interval = st.slider(
                "Auto-refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=self.refresh_interval
            )
            
            if new_interval != self.refresh_interval:
                self.refresh_interval = new_interval
            
            # Export options
            st.write("**Export Options**")
            
            if st.button("📊 Export Metrics"):
                snapshots, history = self.get_budget_data()
                
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "snapshots": snapshots,
                    "history": history[-100:],  # Last 100 events
                    "circuit_breakers": self.get_circuit_breaker_data(),
                    "daemon_status": self.get_daemon_status(),
                }
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"symbiont_metrics_{int(time.time())}.json",
                    mime="application/json"
                )
            
            # System information
            st.write("**System Info**")
            st.write(f"Dashboard Version: 1.0")
            st.write(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    def run(self):
        """Run the complete dashboard."""
        self.render_header()
        self.render_sidebar()
        
        # Main content
        self.render_overview_metrics()
        st.divider()
        
        self.render_token_budget_section()
        st.divider()
        
        self.render_circuit_breaker_section()
        st.divider()
        
        self.render_daemon_status_section()
        st.divider()
        
        self.render_rogue_monitoring_section()
        st.divider()
        
        self.render_alerts_section()


def main():
    """Main entry point for the dashboard."""
    dashboard = SymbiontDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()