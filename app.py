import os, json, sqlite3, streamlit as st, yaml
from symbiont.memory.db import MemoryDB
from symbiont.memory import retrieval
from symbiont.tools import scriptify

CFG_PATH = "./configs/config.yaml"
st.set_page_config(page_title="Cognitive Symbiont — Homebase", layout="wide")

def load_cfg():
    with open(CFG_PATH,"r",encoding="utf-8") as f: return yaml.safe_load(f)
def save_cfg(cfg):
    with open(CFG_PATH,"w",encoding="utf-8") as f: yaml.safe_dump(cfg,f,sort_keys=False)

cfg = load_cfg()
db = MemoryDB(db_path=cfg["db_path"]); db.ensure_schema()

st.title("🧠 Cognitive Symbiont — Homebase (v2.3)")
with st.sidebar:
    st.subheader("LLM Settings")
    prov = st.selectbox("Provider", ["none","ollama","cmd"], index=["none","ollama","cmd"].index(cfg["llm"].get("provider","none")))
    model = st.text_input("Model", cfg["llm"].get("model","phi3:mini"))
    cmd = st.text_input("Cmd (if provider=cmd)", cfg["llm"].get("cmd",""))
    if st.button("Save LLM Settings"):
        cfg["llm"]["provider"]=prov; cfg["llm"]["model"]=model; cfg["llm"]["cmd"]=cmd; save_cfg(cfg); st.success("Saved.")
    st.subheader("RAG")
    if st.button("Rebuild Index"):
        n = retrieval.build_indices(db)
        st.success(f"Indexed {n} items.")

goal = st.text_input("Goal", "Propose one 10-minute refactor for my repo")
if st.button("Run Cycle"):
    from symbiont.orchestrator import Orchestrator
    res = Orchestrator(cfg).cycle(goal=goal)
    st.success(res["decision"]["action"])
    st.code(json.dumps(res["trace"], indent=2))
    bullets = next((o["output"].get("bullets",[]) for o in res["trace"] if o["role"]=="architect"), [])
    if bullets and st.button("Generate Script from Bullets"):
        path = scriptify.write_script(bullets, base_dir=os.path.join(os.path.dirname(cfg["db_path"]), "artifacts","scripts"))
        st.info(f"Script saved at: {path}")

st.subheader("Episodes")
with sqlite3.connect(cfg["db_path"]) as c:
    eps = [{"id":r[0],"title":r[1],"started":r[2],"status":r[3]} for r in c.execute("SELECT id,title,started_at,status FROM episodes ORDER BY id DESC LIMIT 50")]
st.dataframe(eps, use_container_width=True)

st.subheader("Tasks")
with sqlite3.connect(cfg["db_path"]) as c:
    tasks = [{"id":r[0],"episode":r[1],"desc":r[2],"status":r[3],"who":r[4],"ts":r[5]} for r in c.execute("SELECT id,episode_id,description,status,assignee_role,created_at FROM tasks ORDER BY id DESC LIMIT 100")]
st.dataframe(tasks, use_container_width=True)

st.subheader("Artifacts")
with sqlite3.connect(cfg["db_path"]) as c:
    arts = [{"id":r[0],"task":r[1],"type":r[2],"path":r[3],"summary":r[4],"ts":r[5]} for r in c.execute("SELECT id,task_id,type,path,summary,created_at FROM artifacts ORDER BY id DESC LIMIT 100")]
st.dataframe(arts, use_container_width=True)
sel = [a["path"] for a in arts]
pick = st.selectbox("Open artifact", sel if sel else [""])
if pick:
    try:
        st.markdown(open(pick,"r",encoding="utf-8").read())
    except Exception as e:
        st.error(str(e))
