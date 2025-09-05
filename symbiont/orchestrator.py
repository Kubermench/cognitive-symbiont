from __future__ import annotations
import os, json, time
from typing import List, Dict, Any
from rich import print as rprint
from .memory.db import MemoryDB
from .memory import retrieval
from .agents.subself import SubSelf
from .tools.files import ensure_dirs
from .tools import scriptify

class Orchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = MemoryDB(db_path=config["db_path"])
        with open(os.path.join(os.path.dirname(__file__),'roles','roles.yaml'),'r',encoding='utf-8') as f:
            import yaml; self.roles = yaml.safe_load(f).get('roles', [])
        self.subselves = [SubSelf(role=r, config=config) for r in self.roles]
        ensure_dirs([os.path.dirname(config['db_path'] or './data/symbiont.db')])

    def cycle(self, goal: str) -> Dict[str, Any]:
        self.db.ensure_schema()
        retrieval.build_indices(self.db, limit_if_new=256)
        eid = self.db.start_episode(title=f"Goal: {goal}")
        ctx = {"goal": goal, "episode_id": eid, "cwd": os.getcwd()}
        trace = []
        for sub in self.subselves:
            out = sub.run(context=ctx, memory=self.db)
            trace.append({"role": sub.name, "output": out})
            self.db.add_message(role=sub.name, content=json.dumps(out))
        decision = self._consensus(trace)
        self.db.add_task(episode_id=eid, description=decision["action"], status="proposed", assignee_role="architect")
        plan = self._write_plan_artifact(eid, goal, decision["action"], trace)
        bullets = next((o["output"].get("bullets", []) for o in trace if o["role"]=="architect"), [])
        if bullets:
            spath = scriptify.write_script(bullets, base_dir=os.path.join(os.path.dirname(self.config['db_path']), 'artifacts','scripts'))
            with self.db._conn() as c:
                c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'script',spath,'Script from bullets'))
        rprint("[bold green]Consensus Action:[/bold green]", decision["action"], "\n[dim]Saved:", plan)
        return {"episode_id": eid, "decision": decision, "trace": trace}

    def _consensus(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        arch = next((o for o in trace if o["role"]=="architect"), None)
        crit = next((o for o in trace if o["role"]=="critic"), None)
        if crit and crit["output"].get("verdict")=="block":
            fix = crit["output"].get("suggested_fix","narrow scope and re-run")
            return {"action": f"Revise plan: {fix}"}
        if arch: return {"action": arch["output"].get("next_step","No action")}
        return {"action": "No outputs"}

    def _write_plan_artifact(self, eid: int, goal: str, decision: str, trace: List[Dict[str, Any]]):
        bullets = next((o["output"].get("bullets", []) for o in trace if o["role"]=="architect"), [])
        arts_dir = os.path.join(os.path.dirname(self.config["db_path"]), "artifacts")
        os.makedirs(arts_dir, exist_ok=True)
        fpath = os.path.join(arts_dir, f"plan_{eid}_{int(time.time())}.md")
        blines = "\n".join([f"- {b}" for b in bullets]) if bullets else "- (no concrete bullets)"
        with open(fpath,"w",encoding="utf-8") as f:
            f.write(f"# One-Step Plan\nGoal: {goal}\n\n**Action**: {decision}\n\n## 3 bullets\n{blines}\n")
        with self.db._conn() as c:
            c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'plan',fpath,f"Plan for: {goal}"))
        return fpath
