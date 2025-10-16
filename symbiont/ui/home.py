from __future__ import annotations
import os, json, sqlite3, time, threading, statistics
from pathlib import Path
from typing import Any
import streamlit as st
from symbiont.memory.db import MemoryDB
from symbiont.memory import retrieval
from symbiont.tools import scriptify, repo_scan
from symbiont.initiative import daemon as initiative
from symbiont.ports import browser as browser_port
from symbiont.agents.registry import AgentRegistry
from symbiont.orchestration.graph import GraphSpec, GraphRunner


def _rel(ts: int) -> str:
    if not ts:
        return "—"
    d = int(time.time()) - int(ts)
    if d < 60:
        return f"{d}s ago"
    if d < 3600:
        return f"{d//60}m ago"
    if d < 86400:
        return f"{d//3600}h ago"
    return f"{d//86400}d ago"


def _latest_artifact(db_path: str, kind: str):
    with sqlite3.connect(db_path) as c:
        row = c.execute(
            "SELECT path, summary, created_at FROM artifacts WHERE type=? ORDER BY id DESC LIMIT 1",
            (kind,),
        ).fetchone()
    return (row[0], row[1], row[2]) if row else (None, None, None)

def _latest_apply_script(db_path: str):
    with sqlite3.connect(db_path) as c:
        row = c.execute(
            "SELECT path, created_at FROM artifacts WHERE type='script' AND path LIKE '%apply_%' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _latest_governance_snapshot(db_path: str):
    graphs_dir = Path(db_path).resolve().parent / "artifacts" / "graphs"
    if not graphs_dir.exists():
        return None
    graph_files = sorted(
        (p for p in graphs_dir.glob("graph_*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in graph_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        snapshot = data.get("governance")
        if snapshot:
            return path, snapshot
    return None


def _latest_simulation(db_path: str):
    sim_dir = Path(db_path).resolve().parent / "artifacts" / "graphs" / "simulations"
    if not sim_dir.exists():
        return None
    sim_files = sorted(
        (p for p in sim_dir.glob("simulation_*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in sim_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("result", {}).get("stats"):
            return path, data
    return None


def _sd_runs_history(db_path: str, limit: int = 5):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT goal, label, horizon, timestep, stats_json, plot_path, created_at "
                "FROM sd_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except Exception:
        return []
    history = []
    for row in rows:
        try:
            stats = json.loads(row["stats_json"] or "{}")
        except json.JSONDecodeError:
            stats = {}
        record = dict(row)
        record["stats"] = stats
        history.append(record)
    return history


def _paused_graph_states(limit: int = 20):
    states_dir = Path("data/evolution")
    if not states_dir.exists():
        return []
    entries = sorted(
        (p for p in states_dir.glob("graph_state_*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    paused = []
    for path in entries[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("awaiting_human"):
            paused.append((path, data))
    return paused


def _load_budget_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
        except Exception:
            continue
    except Exception:
        return []
    return rows


def _build_budget_chart(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    if not rows:
        return {}
    rows_sorted = sorted(rows, key=lambda r: r.get("ts", 0))
    labels = sorted({row.get("label") for row in rows_sorted if row.get("label")})
    if not labels:
        return {}
    cumulative = {label: 0.0 for label in labels}
    series = {label: [] for label in labels}
    for row in rows_sorted:
        label = row.get("label")
        if label not in cumulative:
            continue
        tokens = float(row.get("prompt_tokens", 0) or 0) + float(row.get("response_tokens", 0) or 0)
        cumulative[label] += tokens
        for lbl in labels:
            series[lbl].append(cumulative[lbl])
    return series


def _upsert_env_var(key: str, value: str, env_path: Path | None = None) -> None:
    if not value:
        return
    env_path = env_path or Path(".env")
    lines: list[str] = []
    found = False
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            if not raw or raw.strip().startswith("#"):
                lines.append(raw)
                continue
            name, _, remainder = raw.partition("=")
            if name == key:
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(raw)
    if not found:
        lines.append(f"{key}={value}")
    text = "\n".join(lines).rstrip("\n") + "\n"
    env_path.write_text(text, encoding="utf-8")


def render_home(cfg: dict, db: MemoryDB):
    st.title("🧠 Cognitive Symbiont — Homebase (v3.6)")
    st.autorefresh(interval=15000, key="home_autorefresh")
    st.markdown(
        "Welcome! Symbiont drafts small, reviewable changes for you. **You stay in control**—every script pauses for approval and every step is logged so you can undo it later."
    )
    st.info(
        "**Three quick steps:** 1) Click *Run a cycle* to let Symbiont suggest a tiny fix. 2) Read the plan it generates. 3) If it looks good, press *Run safely* and confirm."
    )
    with st.expander("What happens behind the scenes?", expanded=False):
        st.markdown(
            "- **Team of roles:** the scout gathers context, the architect drafts the change, and the critic double-checks it.\n"
            "- **Memory:** notes, beliefs, and past scripts are saved so future plans stay aligned with your goals.\n"
            "- **Guard rails:** scripts never run automatically; file writes, command runs, and web fetches all wait for you to approve them and are recorded in the audit log."
        )

    tab_quick, tab_dash, tab_art, tab_ep, tab_settings, tab_sandbox = st.tabs(
        ["Quickstart Wizard", "Dashboard", "Outputs & Scripts", "Episodes & Tasks", "Settings", "Sandbox"]
    )

    with tab_quick:
        st.subheader("Guided Setup")
        st.markdown(
            "Configure Symbiont's essentials without editing YAML by hand."
            " Choose your LLM mode, optional cloud fallback, and whether to enable external research hunts."
        )

        llm_cfg = cfg.get("llm", {})
        current_mode = llm_cfg.get("mode", "local").lower()
        mode = st.radio(
            "LLM mode",
            options=["local", "hybrid", "cloud"],
            index=["local", "hybrid", "cloud"].index(current_mode if current_mode in {"local", "hybrid", "cloud"} else "local"),
            horizontal=True,
        )
        provider = st.selectbox(
            "Local provider",
            options=["ollama", "cmd", "none"],
            index=["ollama", "cmd", "none"].index(llm_cfg.get("provider", "ollama") if llm_cfg.get("provider", "ollama") in {"ollama", "cmd", "none"} else "ollama"),
            help="Choose the primary local provider. 'cmd' lets you supply a custom command fallback.",
        )
        model = st.text_input("Local model ID", llm_cfg.get("model", "phi3:mini"))
        timeout = st.slider("Timeout (seconds)", min_value=10, max_value=900, value=int(llm_cfg.get("timeout_seconds", 600)), step=10)

        cloud_cfg = llm_cfg.get("cloud", {}) if isinstance(llm_cfg.get("cloud"), dict) else {}
        cloud_provider = st.text_input("Cloud provider", cloud_cfg.get("provider", "openai"), disabled=mode == "local")
        cloud_model = st.text_input("Cloud model", cloud_cfg.get("model", cloud_cfg.get("model_name", "gpt-4o-mini")), disabled=mode == "local")
        api_key_env = cloud_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key_value = st.text_input(
            f"{api_key_env} (optional; saved to .env)",
            value="",
            type="password",
            help="Leave blank to keep existing key."
        )
        save_api_key = st.checkbox("Update .env with the provided API key", value=bool(api_key_value))

        retr_cfg = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval"), dict) else {}
        external_cfg = retr_cfg.get("external", {}) if isinstance(retr_cfg.get("external"), dict) else {}
        external_enabled = st.checkbox(
            "Enable external research hunts (arXiv, Semantic Scholar)",
            value=bool(external_cfg.get("enabled", False)),
        )
        max_items = st.number_input(
            "Max external items per query",
            min_value=1,
            max_value=20,
            value=int(external_cfg.get("max_items", 6)),
            step=1,
            disabled=not external_enabled,
        )
        min_relevance = st.slider(
            "Minimum relevance threshold",
            min_value=0.1,
            max_value=1.0,
            value=float(external_cfg.get("min_relevance", 0.7)),
            step=0.05,
            disabled=not external_enabled,
        )

        if st.button("Apply Quickstart Settings", type="primary"):
            try:
                cfg.setdefault("llm", {})
                cfg["llm"]["mode"] = mode
                cfg["llm"]["provider"] = provider
                cfg["llm"]["model"] = model
                cfg["llm"]["timeout_seconds"] = timeout
                cfg["llm"].setdefault("cloud", {})
                cfg["llm"]["cloud"]["provider"] = cloud_provider or "openai"
                cfg["llm"]["cloud"]["model"] = cloud_model or "gpt-4o-mini"
                cfg["llm"]["cloud"]["api_key_env"] = api_key_env
                retrieval_section = cfg.setdefault("retrieval", {})
                external_section = retrieval_section.setdefault("external", {})
                external_section["enabled"] = bool(external_enabled)
                external_section["max_items"] = int(max_items)
                external_section["min_relevance"] = float(min_relevance)
                env_path = Path(".env")
                if save_api_key and api_key_value:
                    _upsert_env_var(api_key_env, api_key_value, env_path=env_path)
                with open("./configs/config.yaml", "w", encoding="utf-8") as handle:
                    import yaml as _y

                    _y.safe_dump(cfg, handle, sort_keys=False)
                st.success("Quickstart settings saved.")
            except Exception as exc:
                st.error(f"Failed to persist settings: {exc}")

    with tab_dash:
        st.markdown("**How it works** — scout → architect → critic → plan + safe script (never auto‑runs). You approve; everything is logged and reversible.")
        with st.expander("Learn more (council, memory, guards)"):
            st.markdown("- Council: scout gathers, architect proposes, critic vets.\n- Memory: beliefs + claims + allowlisted notes.\n- Guards: fs_write / proc_run / net_read need explicit approval; audits record every decision.")
        ui_cfg = cfg.get("ui", {})
        confirm_intent = bool(ui_cfg.get("confirm_intent", False))
        beginner_mode = bool(ui_cfg.get("beginner_mode", False))
        cols = st.columns(3)
        with cols[0]:
            info = repo_scan.inspect_repo(".")
            det = info["detected"]
            repostat = f"{'🟢' if det['git'] else '⚪️'} git · {'🟢' if det['python'] else '⚪️'} python · {'🟢' if det['node'] else '⚪️'} node · {'🟢' if det['editorconfig'] else '🔴'} editorconfig"
            st.markdown(f"**Repo**: {repostat}")
        with cols[1]:
            stt = initiative.get_status(cfg)
            st.markdown(f"**Initiative**: {'🟢 running' if stt.get('daemon_running') else '⚪️ stopped'} · last prop {_rel(stt.get('last_proposal_ts',0))}")
        with cols[2]:
            with sqlite3.connect(cfg["db_path"]) as c:
                ce = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                ca = c.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            st.markdown(f"**Memory**: episodes={ce} · artifacts={ca}")

        data_root_value = cfg.get("data_root")
        if data_root_value:
            data_root = Path(data_root_value)
        else:
            data_root = Path(cfg["db_path"]).resolve().parent

        governance_data = _latest_governance_snapshot(cfg["db_path"])
        if governance_data:
            graph_path, snapshot = governance_data
            st.subheader("Governance forecast")
            st.caption(
                f"Baseline rogue score {snapshot['rogue_baseline']:.2f}; forecasting {len(snapshot['rogue_forecast'])} future cycles from {graph_path.name}."
            )
            metrics_cols = st.columns(2)
            metrics_cols[0].metric("Rogue baseline", f"{snapshot['rogue_baseline']:.2f}")
            forecast_max = max(snapshot.get("rogue_forecast", [snapshot['rogue_baseline']]) or [snapshot['rogue_baseline']])
            metrics_cols[1].metric(
                "Forecast max",
                f"{forecast_max:.2f}",
                delta=f"threshold {snapshot['alert_threshold']:.2f}",
            )
            if snapshot.get("alert"):
                st.warning(
                    f"Rogue score exceeds {snapshot['alert_threshold']:.2f}. Review guards before approving further automation."
                )
            st.line_chart(snapshot["rogue_forecast"])

        sim_data = _latest_simulation(cfg["db_path"])
        if sim_data:
            sim_path, payload = sim_data
            result = payload.get("result", {})
            stats = result.get("stats", {})
            st.subheader("System dynamics preview")
            st.caption(
                f"Latest projection '{payload.get('label', 'baseline')}' captured in {sim_path.name}; horizon {len(result.get('trajectory', []))} steps."
            )
            cols = st.columns(2)
            with cols[0]:
                st.json(stats)
            plot_path = result.get("plot_path")
            if plot_path and Path(plot_path).exists():
                with cols[1]:
                    st.image(str(plot_path), caption="SD projection", use_column_width=True)

        history = _sd_runs_history(cfg["db_path"], limit=5)
        if history:
            with st.expander("Recent SD runs", expanded=False):
                for record in history:
                    ts = record.get("created_at") or 0
                    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) if ts else "—"
                    stats = record.get("stats", {})
                    headline = []
                    for key in ("rogue", "autonomy", "latency"):
                        if key in stats and isinstance(stats[key], dict):
                            last_value = stats[key].get("last")
                            if isinstance(last_value, (int, float)):
                                headline.append(f"{key}={last_value:.2f}")
                    st.markdown(
                        f"**{record['goal']}** — {record['label']} (h={record['horizon']}, dt={record['timestep']:.2f})"
                        f"<br/>`{when} UTC`<br/>Stats: {', '.join(headline) if headline else '(none)' }",
                        unsafe_allow_html=True,
                    )
                    if record.get("plot_path") and Path(record["plot_path"]).exists():
                        st.caption(f"plot: {record['plot_path']}")
                    st.markdown("---")

        token_dir = data_root / "token_budget"
        if token_dir.exists():
            snapshots = {}
            for path in sorted(token_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                label = payload.get("label") or path.stem
                snapshots[label] = payload
            if snapshots:
                total_limit = sum(float(snap.get("limit", 0) or 0) for snap in snapshots.values())
                total_used = sum(float(snap.get("used", 0) or 0) for snap in snapshots.values())
                st.subheader("Token budgets")
                st.caption(
                    f"Used {total_used:.0f} tokens across {len(snapshots)} trackers (limit {total_limit or '∞'})"
                )

                history_rows = _load_budget_history(token_dir / "history.jsonl")
                latency_map: dict[str, float] = {}
                if history_rows:
                    latencies: dict[str, list[float]] = {}
                    for row in history_rows:
                        label = row.get("label")
                        latency = row.get("latency_seconds")
                        if not label or latency is None:
                            continue
                        try:
                            latencies.setdefault(label, []).append(float(latency))
                        except (TypeError, ValueError):
                            continue
                    for label, values in latencies.items():
                        if values:
                            latency_map[label] = statistics.mean(values)

                gauge_columns = st.columns(min(3, len(snapshots)))
                alert_labels = []
                for idx, (label, payload) in enumerate(sorted(snapshots.items())):
                    used = float(payload.get("used", 0) or 0)
                    limit = float(payload.get("limit", 0) or 0)
                    ratio = (used / limit) if limit else None
                    col = gauge_columns[idx % len(gauge_columns)]
                    col.metric(f"{label}", f"{used:.0f}", delta=f"limit {limit:.0f}" if limit else None)
                    if limit:
                        col.progress(min(1.0, used / limit), text=f"{used:.0f}/{limit:.0f} tokens")
                    if label in latency_map:
                        col.caption(f"avg latency {latency_map[label]:.2f}s")
                    if ratio is not None and ratio >= 0.8:
                        alert_labels.append((label, ratio))

                if alert_labels:
                    alerts = ", ".join(f"{label}: {ratio:.0%}" for label, ratio in alert_labels)
                    st.warning(f"Usage nearing limits — {alerts}", icon="⚠️")

                chart_series = _build_budget_chart(history_rows)
                if chart_series:
                    st.line_chart(chart_series, height=220)
                if history_rows:
                    with st.expander("Recent token events", expanded=False):
                        preview = [
                            {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.get("ts", 0))),
                                "label": event.get("label"),
                                "tokens": (event.get("prompt_tokens", 0) or 0)
                                + (event.get("response_tokens", 0) or 0),
                                "latency": round(float(event.get("latency_seconds", 0) or 0), 3),
                                "outcome": event.get("outcome"),
                            }
                            for event in history_rows[-20:]
                        ]
                        st.table(preview)
            else:
                configured_limit = cfg.get("max_tokens")
                if configured_limit:
                    st.info(f"Token budget limit {configured_limit} (no usage recorded yet)")
        else:
            configured_limit = cfg.get("max_tokens")
            if configured_limit:
                st.info(f"Token budget limit {configured_limit} (no usage recorded yet)")
        
        paused_states = _paused_graph_states()
        if paused_states:
            with st.expander("Human-in-loop queue", expanded=False):
                labels = {
                    f"{path.name} — {data.get('goal', '(no goal)')}": (path, data)
                    for path, data in paused_states
                }
                choice = st.selectbox("Select a paused graph", list(labels.keys()))
                selected_path, state_data = labels[choice]
                st.caption(f"Graph path: {state_data.get('graph_path','?')}")
                st.caption(f"Next node: {state_data.get('current_node','END')}")
                budget = state_data.get("token_budget") or {}
                if budget:
                    limit = budget.get("limit") or 0
                    used = budget.get("used") or 0
                    remaining = max(limit - used, 0) if limit else None
                    event_count = len(budget.get("events") or [])
                    limit_label = f"{limit}" if limit else "∞"
                    if remaining is not None:
                        st.caption(
                            f"Token budget: {used}/{limit_label} used (remaining {remaining}, events {event_count})"
                        )
                    else:
                        st.caption(
                            f"Token budget: {used}/{limit_label} used (events {event_count})"
                        )
                    with st.expander("Token usage log", expanded=False):
                        st.json(budget.get("events", [])[-5:])
                handoff = state_data.get("handoff") or {}
                if handoff:
                    st.markdown("**Handoff status**")
                    st.caption(
                        f"Status: {handoff.get('status','?')} · Assignee: {handoff.get('assignee','-')}"
                    )
                    st.json({k: v for k, v in handoff.items() if k not in {"payload"}}, expanded=False)
                    if handoff.get("status") == "pending":
                        outcome = st.selectbox(
                            "Outcome",
                            options=["success", "failure", "block"],
                            index=["success", "failure", "block"].index(
                                handoff.get("outcome") or "success"
                            ),
                        )
                        result_default = handoff.get("result") or {"verdict": "ok"}
                        result_text = st.text_area(
                            "Resolution JSON",
                            value=json.dumps(result_default, indent=2),
                            height=160,
                            key=f"handoff_result_{selected_path.name}",
                        )
                        note_text = st.text_area(
                            "Resolution note (optional)",
                            value=handoff.get("note", ""),
                            height=80,
                            key=f"handoff_note_{selected_path.name}",
                        )
                        if st.button("Mark handoff resolved", key=f"resolve_{selected_path.name}"):
                            try:
                                result_payload = json.loads(result_text) if result_text.strip() else {}
                            except json.JSONDecodeError as exc:
                                st.error(f"Invalid resolution JSON: {exc}")
                            else:
                                handoff_update = dict(handoff)
                                handoff_update.update(
                                    {
                                        "status": "resolved",
                                        "outcome": outcome,
                                        "result": result_payload,
                                        "note": note_text or None,
                                        "resolved_at": int(time.time()),
                                    }
                                )
                                state_data["handoff"] = handoff_update
                                state_data["awaiting_human"] = False
                                try:
                                    selected_path.write_text(
                                        json.dumps(state_data, indent=2),
                                        encoding="utf-8",
                                    )
                                    task_id = handoff_update.get("task_id")
                                    if task_id:
                                        try:
                                            db.update_task_status(
                                                int(task_id),
                                                status=outcome,
                                                result=json.dumps(result_payload),
                                            )
                                        except Exception as exc:
                                            st.warning(f"Task update failed: {exc}")
                                    st.success("Handoff marked resolved. You can now resume the graph.")
                                except Exception as exc:
                                    st.error(f"Failed to persist handoff resolution: {exc}")
                if state_data.get("history"):
                    st.markdown("**Latest node**")
                    last_entry = state_data["history"][-1]
                    st.json(last_entry, expanded=False)
                    default_text = json.dumps(last_entry.get("result", {}), indent=2)
                    edited = st.text_area(
                        "Edit latest result JSON",
                        value=default_text,
                        height=220,
                        key=f"edit_{selected_path.name}"
                    )
                    if st.button("Save edits", key=f"save_{selected_path.name}"):
                        try:
                            new_result = json.loads(edited)
                        except json.JSONDecodeError as exc:
                            st.error(f"Invalid JSON: {exc}")
                        else:
                            state_data["history"][-1]["result"] = new_result
                            selected_path.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
                            st.success("Saved edits.")
                if st.button("Resume graph", key=f"resume_{selected_path.name}"):
                    try:
                        graph_path = Path(state_data.get("graph_path") or "")
                        spec = GraphSpec.from_yaml(graph_path)
                        crew_cfg = Path(state_data.get("crew_config") or spec.crew_config)
                        registry = AgentRegistry.from_yaml(crew_cfg)
                        runner = GraphRunner(spec, registry, cfg, db, graph_path=graph_path)
                        result = runner.run(state_data.get("goal", ""), resume_state=selected_path)
                        if isinstance(result, dict) and result.get("status") == "paused":
                            st.warning(
                                f"Paused again at node {result.get('last_node','?')}."
                            )
                        else:
                            st.success(f"Graph completed. Transcript: {result}")
                    except Exception as exc:
                        st.error(f"Failed to resume graph: {exc}")

        st.subheader("Take an action")
        st.caption("Choose what you need right now. Symbiont handles the technical work and keeps a full paper trail.")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.caption("1. Ask Symbiont for a fresh idea")
            if st.button("Run a cycle", key="dash_run_cycle"):
                from symbiont.memory.db import MemoryDB as _DB
                _db = _DB(cfg["db_path"]) ; _db.ensure_schema()
                li = _db.latest_intent()
                goal = (li or {}).get('summary','Propose one 10-minute refactor for my repo')
                if not confirm_intent:
                    from symbiont.orchestrator import Orchestrator
                    res = Orchestrator(cfg).cycle(goal=goal)
                    st.success(res["decision"]["action"])
                    st.code(json.dumps(res["trace"], indent=2))
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute(
                            "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                            ("plan", "Run Cycle (auto)", goal, 1),
                        )
                else:
                    st.session_state['pending_action'] = ('cycle', None)
            if st.session_state.get('pending_action') == ('cycle', None) and confirm_intent:
                from symbiont.memory.db import MemoryDB as _DB
                _db = _DB(cfg["db_path"]) ; _db.ensure_schema()
                li = _db.latest_intent()
                st.warning(f"Confirm alignment with intent: '{(li or {}).get('summary','(none)')}'.")
                c1,c2 = st.columns(2)
                with c1:
                    if st.button("Confirm & Run", key="dash_cycle_confirm"):
                        from symbiont.orchestrator import Orchestrator
                        res = Orchestrator(cfg).cycle(goal=(li or {}).get('summary','Propose one 10-minute refactor for my repo'))
                        st.success(res["decision"]["action"])
                        st.code(json.dumps(res["trace"], indent=2))
                        with sqlite3.connect(cfg["db_path"]) as c:
                            c.execute(
                                "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                ("plan", "Run Cycle aligned with intent", (li or {}).get('summary',''), 1),
                            )
                        st.session_state['pending_action'] = None
                with c2:
                    if st.button("Not now", key="dash_cycle_cancel"):
                        st.session_state['pending_action'] = None
        with q2:
            st.caption("2. Trigger the initiative watch")
            if st.button("Propose now", key="dash_prop_now"):
                from symbiont.memory.db import MemoryDB as _DB
                _db = _DB(cfg["db_path"]) ; _db.ensure_schema()
                li = _db.latest_intent()
                if not confirm_intent:
                    res = initiative.propose_once(cfg, reason="dashboard")
                    st.success(f"Proposed. Episode {res.get('episode_id')}")
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute(
                            "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                            ("initiative", "Propose Now (auto)", (li or {}).get('summary',''), 1),
                        )
                else:
                    st.session_state['pending_action'] = ('propose', None)
            if st.session_state.get('pending_action') == ('propose', None) and confirm_intent:
                from symbiont.memory.db import MemoryDB as _DB
                _db = _DB(cfg["db_path"]) ; _db.ensure_schema()
                li = _db.latest_intent()
                st.warning(f"Confirm alignment with intent: '{(li or {}).get('summary','(none)')}'.")
                c1,c2 = st.columns(2)
                with c1:
                    if st.button("Confirm & Propose", key="dash_prop_confirm"):
                        res = initiative.propose_once(cfg, reason="dashboard")
                        st.success(f"Proposed. Episode {res.get('episode_id')}")
                        with sqlite3.connect(cfg["db_path"]) as c:
                            c.execute(
                                "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                ("initiative", "Propose Now aligned with intent", (li or {}).get('summary',''), 1),
                            )
                        st.session_state['pending_action'] = None
                with c2:
                    if st.button("Not now", key="dash_prop_cancel"):
                        st.session_state['pending_action'] = None
        with q3:
            st.caption("3. Refresh the knowledge index (advanced)")
            if not beginner_mode:
                if st.button("Rebuild memory", key="dash_rag"):
                    n = retrieval.build_indices(db)
                    st.success(f"Indexed {n} notes and messages.")
            else:
                st.caption("Toggle off Beginner Mode in Settings to see this option.")
        with q4:
            st.caption("4. Turn the last plan into a script (advanced)")
            if not beginner_mode and st.button("Draft last plan as script", key="dash_scriptify"):
                with sqlite3.connect(cfg["db_path"]) as c:
                    row = c.execute(
                        "SELECT content FROM messages WHERE role='architect' ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                if not row:
                    st.warning("No architect output found.")
                else:
                    bullets = json.loads(row[0]).get("bullets", [])
                    if not bullets:
                        st.warning("No bullets to scriptify.")
                    else:
                        st.warning("This writes a shell script to disk. Approve it only if the plan looks right.")
                        if st.button("Yes, create the script", key="dash_scriptify_confirm"):
                            path = scriptify.write_script(
                                bullets,
                                base_dir=os.path.join(
                                    os.path.dirname(cfg["db_path"]), "artifacts", "scripts"
                                ),
                            )
                            with sqlite3.connect(cfg["db_path"]) as c:
                                c.execute(
                                    "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                    ("fs_write", "Write apply script from bullets", "scriptify", 1),
                                )
                            st.success(f"Script saved at: {path}")

        st.subheader("Review the latest plan and script")
        st.caption("Read the plan to see what Symbiont wants to do. The script shows the exact commands it would run once you approve.")
        lp, lps, lpt = _latest_artifact(cfg["db_path"], "plan")
        ls, lst = _latest_apply_script(cfg["db_path"])  # prefer apply scripts
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Latest plan")
            if lp:
                st.write(f"{lp} · {_rel(lpt)}")
                try:
                    st.markdown(open(lp, "r", encoding="utf-8").read())
                except Exception as e:
                    st.error(str(e))
            else:
                st.info("No plan yet. Run a cycle to create one.")
        with c2:
            st.caption("Latest script (safe to run)")
            if ls and ls.endswith(".sh"):
                st.write(f"{ls} · {_rel(lst)}")
                content = open(ls, "r", encoding="utf-8").read()
                st.code("\n".join(content.splitlines()[:40]) or "(empty)")
                if "pending_run_script" not in st.session_state:
                    st.session_state["pending_run_script"] = None
                if st.session_state["pending_run_script"] != ls:
                    if st.button("Run Safely (guarded)", key="dash_run_guarded"):
                        st.session_state["pending_run_script"] = ls
                        st.warning("Confirm execution below. Review preview.")
                else:
                    # Show a fresh preview right next to the guard message so it's visible
                    st.code("\n".join(content.splitlines()[:40]) or "(empty)")
                    cols = st.columns(2)
                    with cols[0]:
                        if st.button("Confirm Execute Now", key="dash_confirm_exec"):
                            import subprocess
                            ts = int(time.time())
                            proc = subprocess.run(["bash", ls], capture_output=True, text=True)
                            out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
                            logs_dir = os.path.join(
                                os.path.dirname(cfg["db_path"]), "artifacts", "logs"
                            )
                            os.makedirs(logs_dir, exist_ok=True)
                            log_path = os.path.join(logs_dir, f"exec_{ts}.txt")
                            with open(log_path, "w", encoding="utf-8") as f:
                                f.write(out)
                            st.success(
                                f"Executed with code {proc.returncode}. Log: {log_path}"
                            )
                            with sqlite3.connect(cfg["db_path"]) as c:
                                c.execute(
                                    "INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))",
                                    (
                                        None,
                                        "log",
                                        log_path,
                                        f"Execution log for {os.path.basename(ls)}",
                                    ),
                                )
                                c.execute(
                                    "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                    (
                                        "proc_run",
                                        f"Run script {os.path.basename(ls)}",
                                        "\n".join(content.splitlines()[:20]),
                                        1,
                                    ),
                                )
                            st.session_state["pending_run_script"] = None
                    with cols[1]:
                        if st.button("Cancel", key="dash_cancel_exec"):
                            with sqlite3.connect(cfg["db_path"]) as c:
                                c.execute(
                                    "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                    ("proc_run", f"Run script {os.path.basename(ls)}", "cancel", 0),
                                )
                            st.session_state["pending_run_script"] = None
            else:
                st.info("No ready-to-run script yet.")
                if st.button("Generate Script from Last Plan", key="dash_generate_script"):
                    with sqlite3.connect(cfg["db_path"]) as c:
                        row = c.execute("SELECT content FROM messages WHERE role='architect' ORDER BY id DESC LIMIT 1").fetchone()
                    if not row:
                        st.warning("No plan found. Click 'Run Cycle' first.")
                    else:
                        try:
                            bullets = json.loads(row[0]).get("bullets", [])
                        except Exception:
                            bullets = []
                        if not bullets:
                            st.warning("Last plan had no concrete steps. Try 'Run Cycle' again.")
                        else:
                            path = scriptify.write_script(bullets, base_dir=os.path.join(os.path.dirname(cfg["db_path"]), "artifacts","scripts"))
                            st.success(f"Script saved at: {path}. You can now click 'Run Safely'.")

        st.subheader("Guard Console (last 10 decisions)")
        try:
            with sqlite3.connect(cfg["db_path"]) as c:
                rows = c.execute(
                    "SELECT capability, description, preview, approved, created_at FROM audits ORDER BY id DESC LIMIT 10"
                ).fetchall()
            view = []
            for cap, desc, prev, approved, ts in rows:
                status = "✅ allowed" if int(approved or 0) == 1 else "⛔ denied"
                when = _rel(ts)
                cap_name = {"fs_write":"Write files","proc_run":"Run commands","net_read":"Read from web"}.get(cap, cap)
                view.append({"capability": cap_name, "action": (desc or "")[:80], "details": (prev or "")[:60], "status": status, "when": when})
            st.dataframe(view, use_container_width=True)
        except Exception as e:
            st.error(str(e))

        # Revert latest rollback
        try:
            with sqlite3.connect(cfg["db_path"]) as c:
                r = c.execute("SELECT path FROM artifacts WHERE type='script' AND path LIKE '%rollback_%' ORDER BY id DESC LIMIT 1").fetchone()
            rb = r[0] if r else None
        except Exception:
            rb = None
        if rb and st.button("Revert Latest (rollback)", key="dash_revert"):
            st.session_state['pending_rollback'] = rb
        if rb and st.session_state.get('pending_rollback') == rb:
            st.warning(f"Confirm rollback script: {rb}")
            c1,c2 = st.columns(2)
            with c1:
                if st.button("Confirm Rollback Now", key="dash_revert_confirm"):
                    import subprocess
                    proc = subprocess.run(["bash", rb], capture_output=True, text=True)
                    st.success(f"Rollback exit code {proc.returncode}")
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute(
                            "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                            ("proc_run", f"Rollback {os.path.basename(rb)}", "", 1),
                        )
                    st.session_state['pending_rollback'] = None
            with c2:
                if st.button("Cancel", key="dash_revert_cancel"):
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute(
                            "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                            ("proc_run", f"Rollback {os.path.basename(rb)}", "cancel", 0),
                        )
                    st.session_state['pending_rollback'] = None

    with tab_art:
        st.subheader("Outputs & Scripts")
        with sqlite3.connect(cfg["db_path"]) as c:
            arts_all = [
                {
                    "id": r[0],
                    "task": r[1],
                    "type": r[2],
                    "path": r[3],
                    "summary": r[4],
                    "ts": r[5],
                }
                for r in c.execute(
                    "SELECT id,task_id,type,path,summary,created_at FROM artifacts ORDER BY id DESC LIMIT 200"
                )
            ]
        kinds = ["all", "plan", "script", "log"]
        kpick = st.selectbox("Filter by type", kinds, index=0, help="Pick what you want to see: plans, scripts, logs, or notes")
        arts = [a for a in arts_all if (kpick == "all" or a["type"] == kpick)]
        st.dataframe(arts, use_container_width=True)
        sel = [a["path"] for a in arts]
        pick = st.selectbox("Open item", sel if sel else [""], help="Select a file to preview")
        if pick:
            try:
                content = open(pick, "r", encoding="utf-8").read()
                if pick.endswith(".md"):
                    st.markdown(content)
                else:
                    st.code(content)
                if pick.endswith(".sh"):
                    st.info("Script controls")
                    if "pending_run_script" not in st.session_state:
                        st.session_state["pending_run_script"] = None
                    if st.session_state["pending_run_script"] != pick:
                        if st.button("Run Safely (guarded)"):
                            st.session_state["pending_run_script"] = pick
                            st.warning(
                                "Confirm execution below. Review script preview first."
                            )
                    else:
                        st.warning(
                            "Confirm: Running this may modify files. Ensure git is clean or you have backups."
                        )
                        st.code("\n".join(content.splitlines()[:40]) or "(empty)")
                        cols = st.columns(2)
                        with cols[0]:
                            if st.button("Confirm Execute Now"):
                                import subprocess
                                ts = int(time.time())
                                proc = subprocess.run(
                                    ["bash", pick], capture_output=True, text=True
                                )
                                out = proc.stdout + (
                                    "\n" + proc.stderr if proc.stderr else ""
                                )
                                logs_dir = os.path.join(
                                    os.path.dirname(cfg["db_path"]), "artifacts", "logs"
                                )
                                os.makedirs(logs_dir, exist_ok=True)
                                log_path = os.path.join(logs_dir, f"exec_{ts}.txt")
                                with open(log_path, "w", encoding="utf-8") as f:
                                    f.write(out)
                                st.success(
                                    f"Executed with code {proc.returncode}. Log: {log_path}"
                                )
                                with sqlite3.connect(cfg["db_path"]) as c:
                                    c.execute(
                                        "INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))",
                                        (
                                            None,
                                            "log",
                                            log_path,
                                            f"Execution log for {os.path.basename(pick)}",
                                        ),
                                    )
                                    c.execute(
                                        "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                        (
                                            "proc_run",
                                            f"Run script {os.path.basename(pick)}",
                                            "\n".join(content.splitlines()[:20]),
                                            1,
                                        ),
                                    )
                                st.session_state["pending_run_script"] = None
                        with cols[1]:
                            if st.button("Cancel"):
                                with sqlite3.connect(cfg["db_path"]) as c:
                                    c.execute(
                                        "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                                        ("proc_run", f"Run script {os.path.basename(pick)}", "cancel", 0),
                                    )
                                st.session_state["pending_run_script"] = None
            except Exception as e:
                st.error(str(e))

    with tab_ep:
        st.subheader("Episodes")
        with sqlite3.connect(cfg["db_path"]) as c:
            eps = [
                {"id": r[0], "title": r[1], "started": r[2], "status": r[3]}
                for r in c.execute(
                    "SELECT id,title,started_at,status FROM episodes ORDER BY id DESC LIMIT 50"
                )
            ]
        st.dataframe(eps, use_container_width=True)
        st.subheader("Tasks")
        with sqlite3.connect(cfg["db_path"]) as c:
            tasks = [
                {
                    "id": r[0],
                    "episode": r[1],
                    "desc": r[2],
                    "status": r[3],
                    "who": r[4],
                    "ts": r[5],
                }
                for r in c.execute(
                    "SELECT id,episode_id,description,status,assignee_role,created_at FROM tasks ORDER BY id DESC LIMIT 100"
                )
            ]
        st.dataframe(tasks, use_container_width=True)

    with tab_settings:
        st.subheader("LLM (Brain) Settings")
        prov = st.selectbox(
            "Provider",
            ["none", "ollama", "cmd"],
            index=["none", "ollama", "cmd"].index(cfg["llm"].get("provider", "none")),
        )
        model = st.text_input("Model", cfg["llm"].get("model", "phi3:mini"))
        cmd = st.text_input("Cmd (if provider=cmd)", cfg["llm"].get("cmd", ""))
        if st.button("Save LLM Settings"):
            cfg["llm"]["provider"] = prov
            cfg["llm"]["model"] = model
            cfg["llm"]["cmd"] = cmd
            with open("./configs/config.yaml", "w", encoding="utf-8") as f:
                import yaml as _y
                _y.safe_dump(cfg, f, sort_keys=False)
            st.success("Saved.")

        st.subheader("Search past notes & outputs (RAG)")
        with st.expander("What is RAG? (plain English)"):
            st.markdown("RAG = Retrieval‑Augmented Generation. Before writing a plan, I quickly look back through your saved notes, plans, and messages to pull in relevant context. This keeps me grounded in your project history instead of guessing.")
        if st.button("Rebuild Index"):
            n = retrieval.build_indices(db)
            st.success(f"Indexed {n} items.")

        st.subheader("UI Preferences")
        ui_cfg = cfg.get("ui", {})
        ui_confirm = st.checkbox("Ask me to confirm intent before each run", value=bool(ui_cfg.get("confirm_intent", False)))
        ui_beginner = st.checkbox("Beginner Mode (hide advanced controls)", value=bool(ui_cfg.get("beginner_mode", True)))
        if st.button("Save UI Preferences"):
            cfg.setdefault("ui", {})
            cfg["ui"]["confirm_intent"] = bool(ui_confirm)
            cfg["ui"]["beginner_mode"] = bool(ui_beginner)
            with open("./configs/config.yaml", "w", encoding='utf-8') as f:
                import yaml as _y
                _y.safe_dump(cfg, f, sort_keys=False)
            st.success("Saved.")

        st.subheader("Initiative Settings")
        ini = cfg.get("initiative", {})
        ini_enabled = st.checkbox("Enabled", value=ini.get("enabled", False))
        ini_timer = st.number_input(
            "Timer minutes", value=int(ini.get("timer_minutes", 120)), min_value=1, step=1
        )
        ini_idle = st.number_input(
            "Idle minutes", value=int(ini.get("idle_minutes", 120)), min_value=1, step=1
        )
        ini_mode = st.selectbox(
            "Trigger mode",
            ["idle_and_timer", "any"],
            index=["idle_and_timer", "any"].index(ini.get("trigger_mode", "idle_and_timer")),
        )
        if st.button("Save Initiative Settings"):
            cfg.setdefault("initiative", {})
            cfg["initiative"].update(
                {
                    "enabled": ini_enabled,
                    "timer_minutes": int(ini_timer),
                    "idle_minutes": int(ini_idle),
                    "trigger_mode": ini_mode,
                }
            )
            with open("./configs/config.yaml", "w", encoding="utf-8") as f:
                import yaml as _y
                _y.safe_dump(cfg, f, sort_keys=False)
            st.success("Saved.")
        if st.button("Propose Now (ignore watchers)"):
            res = initiative.propose_once(cfg, reason="ui-forced")
            st.success(f"Proposed. Episode {res.get('episode_id')}")
        st.caption("Daemon controls are session-local. Closing this page stops the thread.")
        status = initiative.get_status(cfg)
        last_check = (
            time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(status.get("last_check_ts", 0))
            )
            if status.get("last_check_ts")
            else "—"
        )
        last_prop = (
            time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(status.get("last_proposal_ts", 0)),
            )
            if status.get("last_proposal_ts")
            else "—"
        )
        st.write(f"Last check: {last_check}")
        st.write(
            f"Last proposal: {last_prop} · Running: {'yes' if status.get('daemon_running') else 'no'} (pid={status.get('daemon_pid') or '—'})"
        )
        if "daemon_thread" not in st.session_state:
            st.session_state["daemon_thread"] = None
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Daemon"):
                if (
                    st.session_state["daemon_thread"]
                    and st.session_state["daemon_thread"].is_alive()
                ):
                    st.info("Daemon already running.")
                else:
                    initiative.clear_stop()
                    t = threading.Thread(
                        target=initiative.daemon_loop,
                        args=(cfg,),
                        kwargs={"poll_seconds": 60},
                        daemon=True,
                    )
                    t.start()
                    st.session_state["daemon_thread"] = t
                    st.success("Daemon started.")
        with col2:
            if st.button("Stop Daemon"):
                initiative.request_stop()
                t = st.session_state.get("daemon_thread")
                if t and t.is_alive():
                    t.join(timeout=1.0)
                st.success("Stop requested.")

        st.subheader("Intent Checkpoint")
        intent = st.text_area(
            "Intent summary (aligns future proposals)",
            value="Propose one 10-minute refactor for my repo",
        )
        if st.button("Save Intent"):
            from symbiont.memory.db import MemoryDB as _DB
            _db = _DB(cfg["db_path"])
            _db.ensure_schema()
            _db.add_intent(episode_id=0, summary=intent)
            st.success("Saved intent summary.")

        st.subheader("Browser (read-only)")
        bcfg = cfg.get("browser", {"enabled": False, "allowlist": []})
        ben = st.checkbox("Enable browser (read-only)", value=bcfg.get("enabled", False))
        allow = st.text_area("Allowlist (one per line)", value="\n".join(bcfg.get("allowlist", [])))
        if st.button("Save Browser Settings"):
            cfg.setdefault("browser", {})
            cfg["browser"].update({"enabled": ben, "allowlist": [ln.strip() for ln in allow.splitlines() if ln.strip()]})
            with open("./configs/config.yaml", "w", encoding="utf-8") as f:
                import yaml as _y
                _y.safe_dump(cfg, f, sort_keys=False)
            st.success("Saved.")
        test_url = st.text_input("Fetch URL (must be allowlisted)", value=(bcfg.get("allowlist", [""]) or [""])[0])
        if st.button("Fetch to notes (guarded)"):
            try:
                st.warning("This will perform a network fetch (net_read). Confirm below.")
                if st.button("Confirm Fetch", key="confirm_fetch"):
                    path = browser_port.fetch_to_artifact(test_url, cfg)
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute(
                            "INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))",
                            (None, "note", path, f"Note from {test_url}"),
                        )
                        c.execute(
                            "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                            ("net_read", f"Fetch {test_url}", f"GET {test_url}", 1),
                        )
                    st.success(f"Saved note: {path}")
                else:
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute(
                            "INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)",
                            ("net_read", f"Fetch {test_url}", f"GET {test_url}", 0),
                        )
            except Exception as e:
                st.error(str(e))

        st.subheader("Voice Settings (experimental)")
        voice = cfg.get("voice", {"enabled": False, "stt": "none", "tts": "say", "wake_word": "symbiont"})
        v_enabled = st.checkbox("Enable voice", value=voice.get("enabled", False))
        v_tts = st.selectbox("TTS engine", ["say", "none"], index=["say", "none"].index(voice.get("tts", "say")))
        v_wake = st.text_input("Wake word", value=voice.get("wake_word", "symbiont"))
        if st.button("Save Voice Settings"):
            cfg.setdefault("voice", {})
            cfg["voice"].update({"enabled": v_enabled, "tts": v_tts, "wake_word": v_wake})
            with open("./configs/config.yaml", "w", encoding="utf-8") as f:
                import yaml as _y
                _y.safe_dump(cfg, f, sort_keys=False)
            st.success("Saved.")

        st.subheader("Autopilot Settings")
        ap_cfg = cfg.get("autopilot", {"push_enabled": False})
        ap_push = st.checkbox("Allow autopilot to push a branch to remote (opt‑in)", value=ap_cfg.get("push_enabled", False), help="When enabled, Autopilot may push branch symbiont/autopilot to your Git remote.")
        if st.button("Save Autopilot Settings"):
            cfg.setdefault("autopilot", {})
            cfg["autopilot"]["push_enabled"] = bool(ap_push)
            with open("./configs/config.yaml", "w", encoding='utf-8') as f:
                import yaml as _y
                _y.safe_dump(cfg, f, sort_keys=False)
            st.success("Saved.")

    with tab_sandbox:
        st.subheader("Sandbox (local CI-lite + staging)")
        st.caption("Tests → build → stage via Docker Compose. Daily deploy cap enforced.")
        sand_root = os.path.join(os.getcwd(), 'sandbox')
        exists = os.path.exists(sand_root)
        if not exists:
            st.warning("Sandbox not initialized.")
            if st.button("Initialize Sandbox Scaffold"):
                os.makedirs(os.path.join(sand_root, 'app'), exist_ok=True)
                os.makedirs(os.path.join(sand_root, 'tests'), exist_ok=True)
                os.makedirs(os.path.join(sand_root, 'scripts'), exist_ok=True)
                os.makedirs(os.path.join(sand_root, '.state'), exist_ok=True)
                open(os.path.join(sand_root,'requirements.txt'),'w').write('fastapi==0.115.0\nuvicorn[standard]==0.30.6\npytest==8.3.2\nhttpx==0.27.2\n')
                open(os.path.join(sand_root,'app','main.py'),'w').write('from fastapi import FastAPI\napp=FastAPI()\n@app.get("/healthz")\ndef healthz(): return {"status":"ok"}\n@app.get("/echo/{x}")\ndef echo(x:str): return {"echo":x}\n')
                open(os.path.join(sand_root,'tests','test_app.py'),'w').write('from fastapi.testclient import TestClient\nfrom app.main import app\nc=TestClient(app)\n\n\ndef test_health():\n    r=c.get("/healthz"); assert r.status_code==200 and r.json()["status"]=="ok"\n\n\ndef test_echo():\n    r=c.get("/echo/hi"); assert r.json()=={"echo":"hi"}\n')
                open(os.path.join(sand_root,'Dockerfile'),'w').write('FROM python:3.11-slim\nWORKDIR /srv\nCOPY requirements.txt /srv/requirements.txt\nRUN pip install --no-cache-dir -r /srv/requirements.txt\nCOPY app /srv/app\nEXPOSE 8000\nCMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]\n')
                open(os.path.join(sand_root,'docker-compose.yml'),'w').write('services:\n  sandbox-staging:\n    build: .\n    ports: ["8001:8000"]\n    # Resource limits (compose may not enforce strictly; adjust as needed)\n    deploy:\n      resources:\n        limits:\n          cpus: "0.50"\n          memory: 512M\n        reservations:\n          cpus: "0.25"\n          memory: 256M\n    healthcheck:\n      test: ["CMD-SHELL", "wget -qO- http://localhost:8000/healthz | grep -q ok"]\n      interval: 5s\n      timeout: 3s\n      retries: 10\n')
                open(os.path.join(sand_root,'scripts','ci.sh'),'w').write('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/.."\nCAP=${SANDBOX_MAX_DEPLOYS_PER_DAY:-5}\nTODAY="sandbox/.state/deploys-$(date +%F)"\ncount=0; [[ -f "$TODAY" ]] && count=$(cat "$TODAY")\nif (( count >= CAP )); then echo "[cap] daily deploy cap reached ($CAP)"; exit 2; fi\necho "[test] running pytest"\npython -m pip install -r requirements.txt >/dev/null\npython -m pytest -q\necho "[build] docker image"\ndocker build -t symbiont-sandbox:latest .\necho "[stage] up compose"\ndocker compose up -d\necho "[health] waiting for healthcheck"\nfor i in {1..30}; do\n  if curl -sf http://localhost:8001/healthz | grep -q ok; then ok=1; break; fi\n  sleep 1\ndone\nif [[ "${ok:-0}" != "1" ]]; then echo "[health] failed"; docker compose logs --no-color; docker compose down; exit 1; fi\necho "[ok] staged"\necho $((count+1)) > "$TODAY"\n')
                os.chmod(os.path.join(sand_root,'scripts','ci.sh'), 0o755)
                st.success("Sandbox scaffold initialized.")
        else:
            st.success("Sandbox detected at ./sandbox")
            # Cap remaining indicator
            try:
                cap = int(os.environ.get('SANDBOX_MAX_DEPLOYS_PER_DAY', '5'))
            except Exception:
                cap = 5
            from datetime import datetime
            today_file = os.path.join(sand_root, '.state', f"deploys-{datetime.utcnow().strftime('%Y-%m-%d')}")
            used = 0
            try:
                if os.path.exists(today_file):
                    used = int(open(today_file,'r').read().strip() or 0)
            except Exception:
                used = 0
            st.markdown(f"Deploys today: {used}/{cap}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run CI (test → build → stage)"):
                    import subprocess
                    out = st.empty()
                    out.write("Starting CI…")
                    proc = subprocess.Popen(["bash","scripts/ci.sh"], cwd=sand_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    logs = []
                    for line in proc.stdout:
                        logs.append(line.rstrip("\n"))
                        out.code("\n".join(logs[-200:]))
                    proc.wait()
                    # record last status
                    try:
                        os.makedirs(os.path.join(sand_root,'.state'), exist_ok=True)
                        ts = int(time.time())
                        with open(os.path.join(sand_root,'.state','ci_last_status'), 'w', encoding='utf-8') as f:
                            f.write(f"{ts},{proc.returncode}\n")
                        with open(os.path.join(sand_root,'.state','ci_last.log'), 'w', encoding='utf-8') as f:
                            f.write("\n".join(logs))
                    except Exception:
                        pass
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute("INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)", ("proc_run", "sandbox ci", "scripts/ci.sh", 1 if proc.returncode==0 else 0))
                    if proc.returncode==0:
                        st.success("Staged OK. Health at http://localhost:8001/healthz")
                    else:
                        st.error(f"CI failed (code {proc.returncode})")
            with col2:
                if st.button("Down / Rollback"):
                    import subprocess
                    proc = subprocess.run(["docker","compose","down"], cwd=sand_root, capture_output=True, text=True)
                    st.code(proc.stdout + ("\n"+proc.stderr if proc.stderr else ""))
                    with sqlite3.connect(cfg["db_path"]) as c:
                        c.execute("INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)", ("proc_run", "sandbox down", "docker compose down", 1 if proc.returncode==0 else 0))
            st.subheader("CI Status & Logs")
            try:
                stt_path = os.path.join(sand_root,'.state','ci_last_status')
                log_path = os.path.join(sand_root,'.state','ci_last.log')
                if os.path.exists(stt_path):
                    ts_s, code_s = open(stt_path,'r',encoding='utf-8').read().strip().split(',')
                    st.write(f"Last run: {_rel(int(ts_s))} · exit={code_s}")
                if os.path.exists(log_path) and st.button("Show last CI logs"):
                    st.code(open(log_path,'r',encoding='utf-8').read())
            except Exception:
                pass

            st.subheader("Autopilot")
            if st.button("Run Autopilot Now"):
                import subprocess
                env = os.environ.copy()
                try:
                    if cfg.get("autopilot", {}).get("push_enabled", False):
                        env["SYMBIONT_AUTOPILOT_PUSH"] = "1"
                except Exception:
                    pass
                ap = subprocess.run(["bash","scripts/autopilot.sh"], capture_output=True, text=True, env=env)
                st.code(ap.stdout + ("\n"+ap.stderr if ap.stderr else ""))
                with sqlite3.connect(cfg["db_path"]) as c:
                    c.execute("INSERT INTO audits (capability, description, preview, approved) VALUES (?,?,?,?)", ("proc_run", "autopilot run", "scripts/autopilot.sh", 1 if ap.returncode==0 else 0))
