"""Enhanced dashboard with token charts, alerts, and comprehensive metrics."""

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


def load_budget_history(data_root: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    """Load token budget history from JSONL file."""
    history_path = data_root / "token_budget" / "history.jsonl"
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
    except Exception:
        return []
    
    # Sort by timestamp and limit
    events.sort(key=lambda e: e.get("ts", 0))
    return events[-limit:] if limit > 0 else events


def load_budget_snapshots(data_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load current budget snapshots."""
    token_dir = data_root / "token_budget"
    if not token_dir.exists():
        return {}
    
    snapshots: Dict[str, Dict[str, Any]] = {}
    for path in token_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            label = payload.get("label") or path.stem
            snapshots[label] = payload
        except Exception:
            continue
    
    return snapshots


def create_token_usage_chart(events: List[Dict[str, Any]], time_window_hours: int = 24) -> go.Figure:
    """Create cumulative token usage chart."""
    if not events:
        return go.Figure()
    
    # Filter events by time window
    cutoff_time = time.time() - (time_window_hours * 3600)
    recent_events = [e for e in events if e.get("ts", 0) >= cutoff_time]
    
    if not recent_events:
        return go.Figure()
    
    # Group by label and calculate cumulative usage
    label_data = defaultdict(list)
    for event in recent_events:
        label = event.get("label", "unknown")
        ts = event.get("ts", 0)
        prompt_tokens = event.get("prompt_tokens", 0)
        response_tokens = event.get("response_tokens", 0)
        total_tokens = prompt_tokens + response_tokens
        
        if event.get("outcome") == "ok":
            label_data[label].append((ts, total_tokens))
    
    # Create cumulative data for each label
    fig = go.Figure()
    
    for label, data_points in label_data.items():
        if not data_points:
            continue
            
        # Sort by timestamp
        data_points.sort(key=lambda x: x[0])
        
        # Calculate cumulative usage
        cumulative = 0
        timestamps = []
        cumulative_tokens = []
        
        for ts, tokens in data_points:
            cumulative += tokens
            timestamps.append(datetime.fromtimestamp(ts))
            cumulative_tokens.append(cumulative)
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cumulative_tokens,
            mode='lines+markers',
            name=label,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=f"Token Usage Over Time (Last {time_window_hours} hours)",
        xaxis_title="Time",
        yaxis_title="Cumulative Tokens",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_latency_chart(events: List[Dict[str, Any]], time_window_hours: int = 24) -> go.Figure:
    """Create latency distribution chart."""
    if not events:
        return go.Figure()
    
    # Filter events by time window
    cutoff_time = time.time() - (time_window_hours * 3600)
    recent_events = [e for e in events if e.get("ts", 0) >= cutoff_time and e.get("outcome") == "ok"]
    
    if not recent_events:
        return go.Figure()
    
    # Group by label
    label_latencies = defaultdict(list)
    for event in recent_events:
        label = event.get("label", "unknown")
        latency = event.get("latency_seconds")
        if latency is not None:
            label_latencies[label].append(latency)
    
    # Create box plot
    fig = go.Figure()
    
    for label, latencies in label_latencies.items():
        if latencies:
            fig.add_trace(go.Box(
                y=latencies,
                name=label,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
    
    fig.update_layout(
        title=f"Latency Distribution (Last {time_window_hours} hours)",
        yaxis_title="Latency (seconds)",
        xaxis_title="Label",
        showlegend=True
    )
    
    return fig


def create_usage_breakdown_chart(snapshots: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Create current usage breakdown pie chart."""
    if not snapshots:
        return go.Figure()
    
    labels = []
    used_values = []
    limit_values = []
    
    for label, data in snapshots.items():
        used = data.get("used", 0)
        limit = data.get("limit", 0)
        
        if limit > 0:  # Only include budgets with limits
            labels.append(label)
            used_values.append(used)
            limit_values.append(limit)
    
    if not labels:
        return go.Figure()
    
    # Create subplot with two pie charts
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Tokens Used", "Budget Limits")
    )
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=used_values,
        name="Used",
        hovertemplate="<b>%{label}</b><br>Used: %{value}<br>Percentage: %{percent}<extra></extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=limit_values,
        name="Limits",
        hovertemplate="<b>%{label}</b><br>Limit: %{value}<br>Percentage: %{percent}<extra></extra>"
    ), row=1, col=2)
    
    fig.update_layout(
        title="Token Budget Breakdown",
        showlegend=True,
        height=400
    )
    
    return fig


def create_throughput_chart(events: List[Dict[str, Any]], time_window_hours: int = 24) -> go.Figure:
    """Create requests per hour throughput chart."""
    if not events:
        return go.Figure()
    
    # Filter events by time window
    cutoff_time = time.time() - (time_window_hours * 3600)
    recent_events = [e for e in events if e.get("ts", 0) >= cutoff_time]
    
    if not recent_events:
        return go.Figure()
    
    # Group by hour and label
    hourly_data = defaultdict(lambda: defaultdict(int))
    
    for event in recent_events:
        ts = event.get("ts", 0)
        label = event.get("label", "unknown")
        hour = datetime.fromtimestamp(ts).replace(minute=0, second=0, microsecond=0)
        hourly_data[hour][label] += 1
    
    # Create traces for each label
    fig = go.Figure()
    
    all_labels = set()
    for hour_data in hourly_data.values():
        all_labels.update(hour_data.keys())
    
    for label in sorted(all_labels):
        hours = []
        counts = []
        
        for hour in sorted(hourly_data.keys()):
            hours.append(hour)
            counts.append(hourly_data[hour].get(label, 0))
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=counts,
            mode='lines+markers',
            name=label,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=f"Request Throughput (Last {time_window_hours} hours)",
        xaxis_title="Time",
        yaxis_title="Requests per Hour",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def check_budget_alerts(snapshots: Dict[str, Dict[str, Any]], alert_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Check for budget alerts and return alert information."""
    alerts = []
    
    for label, data in snapshots.items():
        used = data.get("used", 0)
        limit = data.get("limit", 0)
        
        if limit > 0:
            usage_ratio = used / limit
            if usage_ratio >= alert_threshold:
                alerts.append({
                    "label": label,
                    "used": used,
                    "limit": limit,
                    "usage_ratio": usage_ratio,
                    "severity": "critical" if usage_ratio >= 0.95 else "warning",
                    "message": f"Budget {label} is {usage_ratio:.1%} full ({used}/{limit} tokens)"
                })
    
    return sorted(alerts, key=lambda x: x["usage_ratio"], reverse=True)


def check_latency_alerts(events: List[Dict[str, Any]], 
                        latency_threshold: float = 10.0,
                        time_window_hours: int = 1) -> List[Dict[str, Any]]:
    """Check for latency alerts."""
    alerts = []
    
    # Filter recent events
    cutoff_time = time.time() - (time_window_hours * 3600)
    recent_events = [e for e in events if e.get("ts", 0) >= cutoff_time and e.get("outcome") == "ok"]
    
    if not recent_events:
        return alerts
    
    # Group by label
    label_latencies = defaultdict(list)
    for event in recent_events:
        label = event.get("label", "unknown")
        latency = event.get("latency_seconds")
        if latency is not None:
            label_latencies[label].append(latency)
    
    # Check for high latency
    for label, latencies in label_latencies.items():
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            if avg_latency > latency_threshold:
                alerts.append({
                    "label": label,
                    "avg_latency": avg_latency,
                    "max_latency": max_latency,
                    "request_count": len(latencies),
                    "severity": "warning" if avg_latency < latency_threshold * 2 else "critical",
                    "message": f"High latency for {label}: avg={avg_latency:.2f}s, max={max_latency:.2f}s"
                })
    
    return sorted(alerts, key=lambda x: x["avg_latency"], reverse=True)


def render_enhanced_dashboard(data_root: Path, 
                            time_window_hours: int = 24,
                            budget_alert_threshold: float = 0.8,
                            latency_alert_threshold: float = 10.0) -> str:
    """Render enhanced dashboard with charts and alerts."""
    
    # Load data
    events = load_budget_history(data_root)
    snapshots = load_budget_snapshots(data_root)
    
    # Check alerts
    budget_alerts = check_budget_alerts(snapshots, budget_alert_threshold)
    latency_alerts = check_latency_alerts(events, latency_alert_threshold)
    
    # Generate dashboard content
    content = []
    
    # Header
    content.extend([
        "# Enhanced Symbiont Dashboard",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Time Window:** Last {time_window_hours} hours",
        f"**Data Root:** `{data_root}`",
        ""
    ])
    
    # Alerts section
    if budget_alerts or latency_alerts:
        content.extend([
            "## 🚨 Alerts",
            ""
        ])
        
        if budget_alerts:
            content.append("### Budget Alerts")
            for alert in budget_alerts:
                severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
                content.append(f"- {severity_icon} **{alert['label']}**: {alert['message']}")
            content.append("")
        
        if latency_alerts:
            content.append("### Latency Alerts")
            for alert in latency_alerts:
                severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
                content.append(f"- {severity_icon} **{alert['label']}**: {alert['message']}")
            content.append("")
    
    # Summary statistics
    total_events = len(events)
    successful_events = len([e for e in events if e.get("outcome") == "ok"])
    total_tokens = sum(e.get("prompt_tokens", 0) + e.get("response_tokens", 0) for e in events if e.get("outcome") == "ok")
    
    content.extend([
        "## 📊 Summary Statistics",
        "",
        f"- **Total Events:** {total_events:,}",
        f"- **Successful Events:** {successful_events:,} ({successful_events/total_events*100:.1f}%)" if total_events > 0 else "- **Successful Events:** 0",
        f"- **Total Tokens Used:** {total_tokens:,}",
        f"- **Active Budgets:** {len(snapshots)}",
        ""
    ])
    
    # Budget status table
    if snapshots:
        content.extend([
            "## 💰 Budget Status",
            "",
            "| Label | Used | Limit | Usage % | Status |",
            "|-------|------|-------|---------|--------|"
        ])
        
        for label, data in sorted(snapshots.items()):
            used = data.get("used", 0)
            limit = data.get("limit", 0)
            usage_pct = (used / limit * 100) if limit > 0 else 0
            status = "🟢 OK" if usage_pct < 80 else "🟡 WARN" if usage_pct < 95 else "🔴 CRIT"
            content.append(f"| {label} | {used:,} | {limit:,} | {usage_pct:.1f}% | {status} |")
        
        content.append("")
    
    # Chart placeholders (would be rendered by Streamlit)
    content.extend([
        "## 📈 Charts",
        "",
        "### Token Usage Over Time",
        "![Token Usage Chart](token_usage_chart.png)",
        "",
        "### Latency Distribution", 
        "![Latency Chart](latency_chart.png)",
        "",
        "### Usage Breakdown",
        "![Usage Breakdown](usage_breakdown_chart.png)",
        "",
        "### Request Throughput",
        "![Throughput Chart](throughput_chart.png)",
        ""
    ])
    
    return "\n".join(content)


def create_streamlit_dashboard(data_root: Path):
    """Create interactive Streamlit dashboard."""
    
    st.set_page_config(
        page_title="Symbiont Enhanced Dashboard",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Symbiont Enhanced Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    time_window = st.sidebar.slider("Time Window (hours)", 1, 168, 24)
    budget_threshold = st.sidebar.slider("Budget Alert Threshold", 0.5, 1.0, 0.8)
    latency_threshold = st.sidebar.slider("Latency Alert Threshold (s)", 1.0, 30.0, 10.0)
    
    # Load data
    events = load_budget_history(data_root)
    snapshots = load_budget_snapshots(data_root)
    
    # Alerts
    budget_alerts = check_budget_alerts(snapshots, budget_threshold)
    latency_alerts = check_latency_alerts(events, latency_threshold)
    
    # Display alerts
    if budget_alerts or latency_alerts:
        st.header("🚨 Alerts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if budget_alerts:
                st.subheader("Budget Alerts")
                for alert in budget_alerts:
                    if alert["severity"] == "critical":
                        st.error(f"🔴 **{alert['label']}**: {alert['message']}")
                    else:
                        st.warning(f"🟡 **{alert['label']}**: {alert['message']}")
        
        with col2:
            if latency_alerts:
                st.subheader("Latency Alerts")
                for alert in latency_alerts:
                    if alert["severity"] == "critical":
                        st.error(f"🔴 **{alert['label']}**: {alert['message']}")
                    else:
                        st.warning(f"🟡 **{alert['label']}**: {alert['message']}")
    
    # Summary metrics
    st.header("📊 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(events)
    successful_events = len([e for e in events if e.get("outcome") == "ok"])
    total_tokens = sum(e.get("prompt_tokens", 0) + e.get("response_tokens", 0) for e in events if e.get("outcome") == "ok")
    
    with col1:
        st.metric("Total Events", f"{total_events:,}")
    
    with col2:
        success_rate = (successful_events / total_events * 100) if total_events > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        st.metric("Total Tokens", f"{total_tokens:,}")
    
    with col4:
        st.metric("Active Budgets", len(snapshots))
    
    # Charts
    st.header("📈 Charts")
    
    # Token usage chart
    st.subheader("Token Usage Over Time")
    token_chart = create_token_usage_chart(events, time_window)
    st.plotly_chart(token_chart, use_container_width=True)
    
    # Latency chart
    st.subheader("Latency Distribution")
    latency_chart = create_latency_chart(events, time_window)
    st.plotly_chart(latency_chart, use_container_width=True)
    
    # Usage breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Usage Breakdown")
        breakdown_chart = create_usage_breakdown_chart(snapshots)
        st.plotly_chart(breakdown_chart, use_container_width=True)
    
    with col2:
        st.subheader("Request Throughput")
        throughput_chart = create_throughput_chart(events, time_window)
        st.plotly_chart(throughput_chart, use_container_width=True)
    
    # Budget status table
    if snapshots:
        st.header("💰 Budget Status")
        
        budget_data = []
        for label, data in snapshots.items():
            used = data.get("used", 0)
            limit = data.get("limit", 0)
            usage_pct = (used / limit * 100) if limit > 0 else 0
            budget_data.append({
                "Label": label,
                "Used": f"{used:,}",
                "Limit": f"{limit:,}",
                "Usage %": f"{usage_pct:.1f}%",
                "Status": "🟢 OK" if usage_pct < 80 else "🟡 WARN" if usage_pct < 95 else "🔴 CRIT"
            })
        
        st.dataframe(budget_data, use_container_width=True)