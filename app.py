import streamlit as st
import yaml
from symbiont.memory.db import MemoryDB
from symbiont.ui.home import render_home

CFG_PATH = "./configs/config.yaml"
st.set_page_config(page_title="Cognitive Symbiont — Homebase", layout="wide")

def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_cfg()
db = MemoryDB(db_path=cfg["db_path"])
db.ensure_schema()
render_home(cfg, db)
