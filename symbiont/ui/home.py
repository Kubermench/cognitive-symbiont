from __future__ import annotations
import os, json, sqlite3, time, threading
import streamlit as st
from symbiont.memory.db import MemoryDB
from symbiont.memory import retrieval
from symbiont.tools import scriptify, repo_scan
from symbiont.initiative import daemon as initiative
from symbiont.ports import browser as browser_port


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


def render_home(cfg: dict, db: MemoryDB):
    st.title("🧠 Cognitive Symbiont — Homebase (v2.4)")

    tab_dash, tab_art, tab_ep, tab_settings = st.tabs(
        ["Dashboard", "Artifacts", "Episodes/Tasks", "Settings"]
    )

    with tab_dash:
        cols = st.columns(3)
        with cols[0]:
            info = repo_scan.inspect_repo(".")
            det = info["detected"]
            st.markdown(
                f"**Repo**: git={'✅' if det['git'] else '❌'} · python={'✅' if det['python'] else '—'} · "
                f"node={'✅' if det['node'] else '—'} · editorconfig={'✅' if det['editorconfig'] else '❌'}"
            )
        with cols[1]:
            stt = initiative.get_status()
            st.markdown(
                f"**Initiative**: running={'🟢' if stt.get('daemon_running') else '⚪️'} · last prop {_rel(stt.get('last_proposal_ts',0))}"
            )
        with cols[2]:
            with sqlite3.connect(cfg["db_path"]) as c:
                ce = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                ca = c.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            st.markdown(f"**Memory**: episodes={ce} · artifacts={ca}")

        st.subheader("Quick Actions")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            if st.button("Run Cycle", key="dash_run_cycle"):
                st.session_state['pending_action'] = ('cycle', None)
            if st.session_state.get('pending_action') == ('cycle', None):
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
                    if st.button("Cancel", key="dash_cycle_cancel"):
                        st.session_state['pending_action'] = None
        with q2:
            if st.button("Propose Now", key="dash_prop_now"):
                st.session_state['pending_action'] = ('propose', None)
            if st.session_state.get('pending_action') == ('propose', None):
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
                    if st.button("Cancel", key="dash_prop_cancel"):
                        st.session_state['pending_action'] = None
        with q3:
            if st.button("Rebuild RAG", key="dash_rag"):
                n = retrieval.build_indices(db)
                st.info(f"Indexed {n} items.")
        with q4:
            if st.button("Scriptify Last Bullets", key="dash_scriptify"):
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
                        st.warning("This will write a script to disk (fs_write). Confirm to continue.")
                        if st.button("Confirm Scriptify", key="dash_scriptify_confirm"):
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

        st.subheader("Latest Artifacts")
        lp, lps, lpt = _latest_artifact(cfg["db_path"], "plan")
        ls, lss, lst = _latest_artifact(cfg["db_path"], "script")
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
            st.caption("Latest script")
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
                            st.session_state["pending_run_script"] = None
            else:
                st.info("No script yet. Use Scriptify or run a cycle.")

    with tab_art:
        st.subheader("Artifacts")
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
        kpick = st.selectbox("Filter by type", kinds, index=0)
        arts = [a for a in arts_all if (kpick == "all" or a["type"] == kpick)]
        st.dataframe(arts, use_container_width=True)
        sel = [a["path"] for a in arts]
        pick = st.selectbox("Open artifact", sel if sel else [""])
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
                        st.code("\n".join(content.splitlines()[:40]))
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
        st.subheader("LLM Settings")
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

        st.subheader("RAG")
        if st.button("Rebuild Index"):
            n = retrieval.build_indices(db)
            st.success(f"Indexed {n} items.")

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
        status = initiative.get_status()
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
