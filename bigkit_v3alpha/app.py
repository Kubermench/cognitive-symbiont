import os, json, time, sqlite3, streamlit as st, yaml
from symbiont.memory.db import MemoryDB
from symbiont.memory import retrieval
from symbiont.memory import beliefs as belief_api
from symbiont.orchestrator import Orchestrator
from symbiont.initiative import daemon as initiative

CFG_PATH = "./configs/config.yaml"
st.set_page_config(page_title="Symbiont BigKit v3.0-alpha", layout="wide")

def load_cfg():
    with open(CFG_PATH,"r",encoding="utf-8") as f: return yaml.safe_load(f)
def save_cfg(cfg):
    with open(CFG_PATH,"w",encoding="utf-8") as f: yaml.safe_dump(cfg,f,sort_keys=False)

cfg = load_cfg()
db = MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()

st.title("🧠 Symbiont BigKit v3.0-alpha")

tab_cycles, tab_memory, tab_beliefs, tab_agency, tab_ports = st.tabs([
    "Cycles", "Memory", "Beliefs", "Agency", "Ports"
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
    st.subheader("Beliefs v1")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Add / Update")
        statement = st.text_input("Statement")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
        evidence_json = st.text_area("Evidence (JSON)", "")
        if st.button("Add Belief"):
            belief_api.add_belief(db, statement, confidence, evidence_json)
            st.success("Added.")
    with col2:
        st.write("Existing Beliefs")
        bl = belief_api.list_beliefs(db)
        st.dataframe(bl, use_container_width=True)

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

