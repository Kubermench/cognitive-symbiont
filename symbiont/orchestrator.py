from __future__ import annotations
import os, json, re, time
from pathlib import Path
from typing import List, Dict, Any
from rich import print as rprint
from .memory.db import MemoryDB
from .memory import retrieval
from .memory import graphrag
from .agents.subself import SubSelf
from .llm.budget import TokenBudget
from .agents.reflector import CycleReflector
from .agents.swarm import SwarmCoordinator
from .tools.files import ensure_dirs
from .memory import beliefs as belief_api
from .tools import scriptify

class Orchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = MemoryDB(db_path=config["db_path"])
        with open(os.path.join(os.path.dirname(__file__),'roles','roles.yaml'),'r',encoding='utf-8') as f:
            import yaml; self.roles = yaml.safe_load(f).get('roles', [])
        ensure_dirs([os.path.dirname(config['db_path'] or './data/symbiont.db')])
        self.reflector = CycleReflector(config)
        swarm_candidate = SwarmCoordinator(config)
        self.swarm = swarm_candidate if swarm_candidate.enabled else None
        self._rlhf_dir = Path(self.db.db_path).parent / "artifacts" / "rlhf"
        self._rlhf_dir.mkdir(parents=True, exist_ok=True)

    def cycle(self, goal: str) -> Dict[str, Any]:
        self.db.ensure_schema()
        retrieval.build_indices(self.db, limit_if_new=256)
        eid = self.db.start_episode(title=f"Goal: {goal}")
        ctx = {"goal": goal, "episode_id": eid, "cwd": os.getcwd()}
        limit = 0
        try:
            limit = int(self.config.get("max_tokens", 0) or 0)
        except (TypeError, ValueError):
            limit = 0
        data_root = Path(
            self.config.get("data_root")
            or Path(self.config.get("db_path", "./data/symbiont.db")).parent
        )
        sink_path = data_root / "token_budget" / f"cycle_{eid}.json"
        history_path = sink_path.parent / "history.jsonl"
        budget = TokenBudget(
            limit=limit,
            label=f"cycle:{eid}",
            sink_path=sink_path,
            history_path=history_path,
        )
        ctx["token_budget"] = budget
        subselves = [SubSelf(role=r, config=self.config, token_budget=budget) for r in self.roles]

        trace = []
        for sub in subselves:
            out = sub.run(context=ctx, memory=self.db)
            trace.append({"role": sub.name, "output": out})
            self.db.add_message(role=sub.name, content=json.dumps(out))
        decision = self._consensus(trace)
        self.db.add_task(episode_id=eid, description=decision["action"], status="proposed", assignee_role="architect")
        plan = self._write_plan_artifact(eid, goal, decision["action"], trace)
        bullets = next((o["output"].get("bullets", []) for o in trace if o["role"]=="architect"), [])
        if bullets:
            scripts_dir = os.path.join(os.path.dirname(self.config['db_path']), 'artifacts','scripts')
            spath = scriptify.write_script(bullets, base_dir=scripts_dir, db_path=self.config['db_path'], episode_id=eid)
            rpath = scriptify.write_rollback_script(spath)
            with self.db._conn() as c:
                c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'script',spath,'Script from bullets'))
                c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'script',rpath,'Rollback script'))
        reward = self._score_cycle(trace, goal)
        result = {"episode_id": eid, "decision": decision, "trace": trace, "reward": reward}
        try:
            self.reflector.observe_cycle(result)
        except Exception as exc:  # pragma: no cover - reflection should not break main flow
            rprint(f"[yellow]Reflection skipped:[/yellow] {exc}")
        if self.swarm:
            try:
                self.swarm.after_cycle(result, budget=budget)
            except Exception as exc:  # pragma: no cover
                rprint(f"[yellow]Swarm evolution skipped:[/yellow] {exc}")
        rprint("[bold green]Consensus Action:[/bold green]", decision["action"], "\n[dim]Saved:", plan)
        self._log_cycle_reward(eid, goal, reward, trace)
        return result

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
        # sources: capture recent notes and link them to this episode so we can attribute usage
        notes = []
        try:
            with self.db._conn() as c:
                now = int(time.time())
                rows = c.execute(
                    "SELECT id, path, summary FROM artifacts WHERE type='note' AND created_at>=? ORDER BY id DESC LIMIT 5",
                    (now - 86400,),
                ).fetchall()
                # Link to episode
                for (aid, p, s) in rows:
                    c.execute("INSERT INTO episode_artifacts (episode_id, artifact_id, linked_at) VALUES (?,?,strftime('%s','now'))", (eid, aid))
            # Build display list from linked notes
            with self.db._conn() as c:
                rows2 = c.execute(
                    "SELECT a.path FROM episode_artifacts ea JOIN artifacts a ON ea.artifact_id=a.id WHERE ea.episode_id=? ORDER BY ea.linked_at DESC LIMIT 5",
                    (eid,),
                ).fetchall()
            for (p,) in rows2:
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
                    f"\n## Sources (this cycle)\n{sources_text}\n"
                    f"\n## Relevant Claims\n{claims_text}\n"
                )
            )
        with self.db._conn() as c:
            c.execute("INSERT INTO artifacts (task_id,type,path,summary,created_at) VALUES (?,?,?,?,strftime('%s','now'))", (None,'plan',fpath,f"Plan for: {goal}"))
        return fpath

    def _score_cycle(self, trace: List[Dict[str, Any]], goal: str) -> float:
        reward = 0.6
        architect = next((o for o in trace if o["role"] == "architect"), None)
        critic = next((o for o in trace if o["role"] == "critic"), None)
        if architect:
            bullets = architect["output"].get("bullets", []) or []
            reward += 0.1 * min(3, len(bullets))
        if critic and critic["output"].get("verdict") == "block":
            reward -= 0.4
        if "rlhf" in goal.lower():
            reward += 0.05
        return round(max(0.0, min(1.0, reward)), 3)

    def _log_cycle_reward(
        self,
        episode_id: int,
        goal: str,
        reward: float,
        trace: List[Dict[str, Any]],
    ) -> None:
        payload = {
            "episode_id": episode_id,
            "goal": self._scrub_text(goal),
            "reward": reward,
            "timestamp": int(time.time()),
            "agents": [
                {
                    "role": entry["role"],
                    "score": entry["output"].get("score"),
                }
                for entry in trace
            ],
        }
        path = self._rlhf_dir / f"reward_{episode_id}.json"
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _scrub_text(self, text: str) -> str:
        if not text:
            return ""
        patterns = [
            r"sk-[a-zA-Z0-9]{20,}",
            r"[A-Za-z0-9_]+=\S+",
            r"(?i)password\s*[:=]\s*\S+",
        ]
        scrubbed = text
        for pattern in patterns:
            scrubbed = re.sub(pattern, "[redacted]", scrubbed)
        return scrubbed

    def train_from_rewards(self) -> Dict[str, Any]:
        """Aggregate reward history for lightweight RLHF fine-tuning."""

        records: List[Dict[str, Any]] = []
        for reward_file in sorted(self._rlhf_dir.glob("reward_*.json")):
            try:
                records.append(json.loads(reward_file.read_text(encoding="utf-8")))
            except Exception:
                continue
        if not records:
            return {"count": 0, "mean_reward": 0.0}
        mean_reward = sum(r.get("reward", 0.0) for r in records) / len(records)
        top = sorted(records, key=lambda r: r.get("reward", 0.0), reverse=True)[:5]
        return {
            "count": len(records),
            "mean_reward": round(mean_reward, 3),
            "top_examples": top,
        }
