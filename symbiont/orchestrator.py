from __future__ import annotations
import os, json, time
from typing import List, Dict, Any
from rich import print as rprint
from .memory.db import MemoryDB
from .memory import retrieval
from .memory import graphrag
from .agents.subself import SubSelf
from .tools.files import ensure_dirs
from .memory import beliefs as belief_api
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
            scripts_dir = os.path.join(os.path.dirname(self.config['db_path']), 'artifacts','scripts')
            spath = scriptify.write_script(bullets, base_dir=scripts_dir)
            rpath = scriptify.write_rollback_script(spath)
            with self.db._conn() as c:
                c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'script',spath,'Script from bullets'))
                c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'script',rpath,'Rollback script'))
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
        # beliefs/assumptions
        try:
            btop = belief_api.list_beliefs(self.db)[:3]
        except Exception:
            btop = []
        btext = "\n".join([f"- {b['statement']} (conf {b['confidence']:.2f})" for b in btop]) if btop else "- (none)"
        # sources (recent notes within 24h)
        notes = []
        try:
            with self.db._conn() as c:
                now = int(time.time())
                rows = c.execute(
                    "SELECT path, summary, created_at FROM artifacts WHERE type='note' AND created_at>=? ORDER BY id DESC LIMIT 3",
                    (now - 86400,),
                ).fetchall()
            for (p, s, ts) in rows:
                try:
                    first = open(p, 'r', encoding='utf-8').read().splitlines()[0]
                except Exception:
                    first = os.path.basename(p)
                notes.append(f"- {first} — {p}")
        except Exception:
            pass
        sources_text = "\n".join(notes) if notes else "- (none)"
        # relevant claims (GraphRAG-lite)
        try:
            claims = graphrag.query_claims(self.db, goal, limit=3)
            claims_text = "\n".join([f"- {c['subject']} {c['relation']} {c['object']} (imp {c['importance']:.2f})" for c in claims]) if claims else "- (none)"
        except Exception:
            claims_text = "- (none)"
        with open(fpath,"w",encoding="utf-8") as f:
            f.write(
                (
                    f"# One-Step Plan\n"
                    f"Goal: {goal}\n\n"
                    f"**Action**: {decision}\n\n"
                    f"## 3 bullets\n{blines}\n\n"
                    f"## Assumptions (beliefs)\n{btext}\n"
                    f"\n## Sources (recent notes)\n{sources_text}\n"
                    f"\n## Relevant Claims\n{claims_text}\n"
                )
            )
        with self.db._conn() as c:
            c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'plan',fpath,f"Plan for: {goal}"))
        return fpath
