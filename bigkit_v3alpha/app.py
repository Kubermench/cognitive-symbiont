import os, json, time, sqlite3, streamlit as st, yaml
from pathlib import Path
from symbiont.memory.db import MemoryDB
from symbiont.memory import retrieval
from symbiont.memory import beliefs as belief_api
from symbiont.memory import graphrag
from symbiont.orchestrator import Orchestrator
from symbiont.initiative import daemon as initiative
from symbiont import guards as guard_mod

CFG_PATH = "./configs/config.yaml"
st.set_page_config(page_title="Symbiont BigKit v3.0-alpha", layout="wide")

def load_cfg():
    with open(CFG_PATH,"r",encoding="utf-8") as f: return yaml.safe_load(f)
def save_cfg(cfg):
    with open(CFG_PATH,"w",encoding="utf-8") as f: yaml.safe_dump(cfg,f,sort_keys=False)

cfg = load_cfg()
db = MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()

st.title("🧠 Symbiont BigKit v3.0-alpha")

tab_cycles, tab_memory, tab_beliefs, tab_governance, tab_foresight, tab_agency, tab_ports = st.tabs([
    "Cycles", "Memory", "Beliefs", "Governance", "Foresight", "Agency", "Ports"
])

with tab_cycles:
    st.subheader("Council Cycle")
    goal = st.text_input("Goal", "Propose one 10-minute refactor for my repo")
    if st.button("Run Cycle", key="bk_run_cycle"):
        res = Orchestrator(cfg).cycle(goal=goal)
        st.success(res["decision"]["action"])
        st.code(json.dumps(res["trace"], indent=2))

with tab_memory:
    st.subheader("Episodes / Tasks / Artifacts")
    with sqlite3.connect(cfg["db_path"]) as c:
        eps = [{"id":r[0],"title":r[1],"started":r[2],"status":r[3]} for r in c.execute("SELECT id,title,started_at,status FROM episodes ORDER BY id DESC LIMIT 50")]
        tasks = [{"id":r[0],"episode":r[1],"desc":r[2],"status":r[3],"who":r[4],"ts":r[5]} for r in c.execute("SELECT id,episode_id,description,status,assignee_role,created_at FROM tasks ORDER BY id DESC LIMIT 100")]
        arts = [{"id":r[0],"task":r[1],"type":r[2],"path":r[3],"summary":r[4],"ts":r[5]} for r in c.execute("SELECT id,task_id,type,path,summary,created_at FROM artifacts ORDER BY id DESC LIMIT 100")]
    st.write("Episodes")
    st.dataframe(eps, use_container_width=True)
    st.write("Tasks")
    st.dataframe(tasks, use_container_width=True)
    st.write("Artifacts")
    st.dataframe(arts, use_container_width=True)

with tab_beliefs:
    st.subheader("Beliefs & GraphRAG")
    with st.expander("Add statement", expanded=True):
        statement = st.text_input("Statement")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05, key="belief_conf")
        evidence_json = st.text_area("Evidence (JSON)", "", key="belief_evidence")
        if st.button("Add Belief", key="belief_add") and statement.strip():
            belief_api.add_belief(db, statement.strip(), confidence, evidence_json)
            st.success("Belief stored.")
            st.experimental_rerun()

    st.markdown("### Belief Statements")
    bl = belief_api.list_beliefs(db)
    if bl:
        st.dataframe(bl, use_container_width=True)
    else:
        st.info("No beliefs captured yet.")

    st.markdown("### GraphRAG Triples")
    claims = graphrag.list_claims(db, limit=100)
    if claims:
        st.dataframe([{**c, "source": c.get("source_url") or "-"} for c in claims], use_container_width=True)
        def _escape(val: str) -> str:
            return (val or "").replace("\"", "\\\"")

        edges = [
            f"\"{_escape(c['subject'])}\" -> \"{_escape(c['object'])}\" [label=\"{_escape(c['relation'])}\\n{c['importance']:.2f}\"]"
            for c in claims
        ]
        dot = "digraph beliefs {\n  rankdir=LR;\n  " + "\n  ".join(edges) + "\n}"
        st.graphviz_chart(dot)

        selected = st.selectbox(
            "Edit claim",
            claims,
            format_func=lambda c: f"#{c['id']} {c['subject']} {c['relation']} {c['object']} ({c['importance']:.2f})",
        )
        boost, lower, delete = st.columns(3)
        if boost.button("Boost", key="claim_boost"):
            graphrag.adjust_claim_importance(db, selected['id'], 0.1)
            st.experimental_rerun()
        if lower.button("Lower", key="claim_lower"):
            graphrag.adjust_claim_importance(db, selected['id'], -0.1)
            st.experimental_rerun()
        if delete.button("Delete", key="claim_delete"):
            graphrag.delete_claim(db, selected['id'])
            st.experimental_rerun()
    else:
        st.info("No GraphRAG triples captured yet. Use CLI `sym graph_add_claim` or Homebase beliefs form.")

with tab_agency:
    st.subheader("Initiative (run once)")
    if st.button("Propose Now (ignore watchers)", key="bk_prop_now"):
        res = initiative.propose_once(cfg, reason="bigkit-ui")
        st.success(f"Proposed. Episode {res.get('episode_id')}")
    st.caption("To use watchers/daemon, return to Homebase or CLI.")

with tab_ports:
    st.subheader("Ports (stubs)")
    st.write("CLI: run `python -m symbiont.cli --help`")
    st.write("Voice-mini: planned — wake word + local STT/TTS (read-only)")
    st.write("Browser: planned — read-only allowlist with source provenance")
    st.write("AR overlay: planned — next-action card prototype")


with tab_governance:
    st.subheader("Governance Dashboard")
    state_path = Path("data/evolution/state.json")
    if state_path.exists():
        history = json.loads(state_path.read_text()).get("history", [])
        st.write("Recent Actions")
        st.dataframe(history[-10:], use_container_width=True)
    else:
        st.info("No reflection history yet. Run a few cycles first.")
        history = []

    latest_apply = sorted(Path("data/artifacts/scripts").glob("apply_*.sh"))
    if latest_apply:
        script_report = guard_mod.analyze_script(latest_apply[-1])
        st.metric("Latest Rogue Score", script_report.get("rogue_score", 0.0))
        if script_report.get("issues"):
            for issue in script_report["issues"]:
                st.warning(f"{issue['reason']} (weight {issue['weight']})")
    else:
        st.write("No scripts analysed yet.")

    if st.button("Export Compliance Report"):
        reports_dir = Path("data/artifacts/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        report_path = reports_dir / f"governance_{timestamp}.md"
        body = ["# Governance Snapshot", f"Exported: {time.ctime(timestamp)}"]
        if latest_apply:
            body.append(f"Latest script: {latest_apply[-1]}")
            body.append(f"Rogue score: {script_report.get('rogue_score', 0.0)}")
        if history:
            body.append("## Recent History\n")
            body.append(json.dumps(history[-10:], indent=2))
        report_path.write_text("\n\n".join(body), encoding="utf-8")
        st.success(f"Compliance report saved to {report_path}")

with tab_foresight:
    st.subheader("Foresight Hunts")
    meta_dir = Path("data/artifacts/foresight/meta")
    records = []
    if meta_dir.exists():
        for path in sorted(meta_dir.glob("*_meta.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = payload.get("meta", {}) or {}
            breakdown = meta.get("source_breakdown", {}) if isinstance(meta.get("source_breakdown"), dict) else {}
            records.append(
                {
                    "Timestamp": payload.get("timestamp"),
                    "Topic": payload.get("topic"),
                    "Total": meta.get("total_candidates"),
                    "Dropped": meta.get("dropped_low_score"),
                    "Tokens": meta.get("token_delta"),
                    "Cost": meta.get("cost_estimate"),
                    "Sources": ", ".join(f"{src}:{info.get('count', 0)}" for src, info in breakdown.items()),
                    "Meta Path": str(path),
                    "Plot Path": meta.get("source_plot"),
                    "_breakdown": breakdown,
                }
            )

    if records:
        st.dataframe(records[::-1], use_container_width=True)
        chart_data = {
            "cost": [float(r.get("Cost") or 0.0) for r in records],
            "tokens": [float(r.get("Tokens") or 0.0) for r in records],
        }
        st.markdown("### Cost & Token Trend")
        st.line_chart(chart_data)

        latest = records[-1]
        st.markdown("### Latest Source Mix")
        breakdown = latest.get("_breakdown", {})
        if breakdown:
            breakdown_rows = [
                {
                    "source": src,
                    "count": info.get("count", 0),
                    "avg_score": info.get("avg_score", 0.0),
                }
                for src, info in breakdown.items()
            ]
            st.table(breakdown_rows)
        plot_path = latest.get("Plot Path")
        if plot_path and Path(plot_path).exists():
            st.image(str(plot_path), caption="Source distribution", use_column_width=True)
    else:
        st.info("No foresight hunts recorded yet.")
