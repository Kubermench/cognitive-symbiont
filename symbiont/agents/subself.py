from __future__ import annotations
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..tools import repo_scan
from ..llm.client import LLMClient
from ..llm.budget import TokenBudget
from ..memory import retrieval
import os, json, re

class SubSelf(BaseAgent):
    def __init__(
        self,
        role: Dict[str, Any],
        config: Dict[str, Any],
        *,
        llm_client: LLMClient | None = None,
        token_budget: TokenBudget | None = None,
    ):
        self.role = role
        self.name = role.get("name", "anon")
        self.goal = role.get("goal", "")
        self.style = role.get("style", "")
        self.llm = llm_client or LLMClient((config or {}).get("llm", {}))
        self.token_budget = token_budget

    def run(self, context: Dict[str, Any], memory) -> Dict[str, Any]:
        goal = context.get("goal","")
        cwd = context.get("cwd", os.getcwd())
        if self.name == "scout":
            return {"role": self.name, "bullets": self._scout(goal, memory)}
        if self.name == "architect":
            ns, bullets = self._architect(goal, memory, cwd)
            return {"role": self.name, "next_step": ns, "bullets": bullets}
        if self.name == "critic":
            v, fix = self._critic(goal, memory)
            return {"role": self.name, "verdict": v, "suggested_fix": fix}
        if self.name == "dynamics_scout":
            return self._dynamics_scout(goal, context, memory)
        if self.name == "sd_modeler":
            return self._sd_modeler(goal, context)
        if self.name == "strategist":
            return self._strategist(goal, context)
        return {"role": self.name, "note": "noop"}

    def _scout(self, goal: str, memory) -> List[str]:
        hints = []
        for r in retrieval.search(memory, goal, k=3):
            hints.append(f"mem:{r['kind']}→{r['preview'][:100]} (score={r['score']:.2f})")
        for m in memory.last_messages(limit=3):
            hints.append(f"recent:{m['role']}→{m['content'][:80]}")
        hints.append("constraint: ≤60s, one-step action")
        hints.append(f"goal-seen: {goal[:120]}")
        return hints

    def _architect(self, goal: str, memory, cwd: str):
        info = repo_scan.inspect_repo(cwd)
        suggestions = sorted(info["suggestions"], key=lambda s: (s["effort_min"], s["priority"]))
        top = suggestions[:5]
        heur = []
        for s in top[:3]:
            cmd = f" (cmd: {s['commands'][0]})" if s.get("commands") else ""
            heur.append(f"{s['title']} — {s['details']}{cmd}")
        next_step = f"Apply 3 quick refactors: {', '.join([s['title'] for s in top[:3]])}" if top else "No obvious quick refactors; run scan"
        prompt = self._bullets_prompt(goal, top)
        llm_out = (
            self.llm.generate(
                prompt,
                budget=self.token_budget,
                label=f"{self.name}:bullets",
            )
            if prompt
            else ""
        )
        parsed: List[str] = []
        if llm_out:
            for ln in llm_out.splitlines():
                line = ln.strip()
                if not line:
                    continue
                if line.startswith("- "):
                    parsed.append(line[2:])
                    continue
                # Accept numbered or bulleted variants (e.g., "1. ...", "1) ...", "* ...")
                m = re.match(r"^(?:\d+[.\)]\s*|[•*]\s*)(.+)", line)
                if m:
                    parsed.append(m.group(1).strip())
                    continue
                parsed.append(line)
        bullets = (parsed or heur)[:3]
        if parsed:
            next_step = f"Apply 3 quick refactors: {', '.join([b.split(' — ',1)[0] for b in bullets])}"
        return next_step, bullets

    def _critic(self, goal: str, memory):
        return ("block","Clarify deliverable + constraints.") if len(goal.split())<4 else ("ok","")

    def _bullets_prompt(self, goal: str, top):
        lines = [f"- {s['title']}: {s['details']}" for s in top]
        repo_ctx = "\n".join(lines) if lines else "No suggestions."
        return f"""You are a senior developer. Goal: "{goal}".
The repo has these low-effort suggestions (≤10 minutes each):
{repo_ctx}

Write EXACTLY 3 concise bullets. Format EACH:
- <Title> — <one-line details> (cmd: <one short shell command>)

Only bullets. No extra text.
"""

    def _dynamics_scout(self, goal: str, context: Dict[str, Any], memory) -> Dict[str, Any]:
        goal_hint = goal.lower()
        autonomy_start = 0.55 if "automation" in goal_hint else 0.45
        rogue_start = 0.22 if "guard" in goal_hint else 0.18
        blueprint = {
            "timestep": 1.0,
            "stocks": [
                {"name": "autonomy", "initial": autonomy_start, "min": 0.0, "max": 1.0},
                {"name": "rogue", "initial": rogue_start, "min": 0.0, "max": 1.0},
                {"name": "latency", "initial": 0.25, "min": 0.0, "max": 1.5},
                {"name": "knowledge", "initial": 0.5, "min": 0.0, "max": 2.0},
            ],
            "auxiliaries": [
                {"name": "proposal_rate", "expression": "0.35 + 0.4 * autonomy"},
                {"name": "guard_pressure", "expression": "max(0.0, rogue - 0.35)"},
                {"name": "latency_drag", "expression": "0.25 * latency"},
            ],
            "flows": [
                {"name": "autonomy_gain", "target": "autonomy", "expression": "0.08 * proposal_rate - 0.05 * guard_pressure"},
                {"name": "rogue_drift", "target": "rogue", "expression": "0.04 * autonomy - 0.07 * guard_pressure"},
                {"name": "latency_load", "target": "latency", "expression": "0.05 * proposal_rate - 0.06"},
                {"name": "knowledge_compound", "target": "knowledge", "expression": "0.07 * proposal_rate - 0.04 * guard_pressure"},
            ],
            "parameters": {
                "trend_autonomy": 0.1 if "swarm" in goal_hint else 0.05,
                "trend_latency": 0.03,
            },
        }
        context.setdefault("sd_blueprint", blueprint)
        return {
            "role": self.name,
            "sd_blueprint": blueprint,
            "notes": "Mapped baseline stocks (autonomy, rogue, latency, knowledge).",
        }

    def _sd_modeler(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        horizon = 80 if "swarm" in goal.lower() else 60
        blueprint = context.get("sd_blueprint") or {}
        return {
            "role": self.name,
            "simulate": True,
            "horizon": horizon,
            "noise": 0.02,
            "timestep": blueprint.get("timestep", 1.0),
        }

    def _strategist(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        projection = context.get("sd_projection") or {}
        stats = projection.get("stats", {})
        interventions: List[str] = []
        rogue = stats.get("rogue", {})
        autonomy = stats.get("autonomy", {})
        latency = stats.get("latency", {})

        if rogue.get("last", 0.0) > 0.6:
            interventions.append("Lower guard threshold by 0.1 and schedule audit cycle")
        if autonomy.get("last", 0.0) < autonomy.get("avg", 0.0):
            interventions.append("Inject new success exemplars into GraphRAG")
        if latency and latency.get("max", 0.0) > 0.9:
            interventions.append("Switch hybrid router bias to local for next 10 cycles")
        if not interventions:
            interventions.append("Proceed with guarded execution; monitor rogue trend every 10 cycles")

        summary = {
            "rogue_last": rogue.get("last"),
            "autonomy_last": autonomy.get("last"),
            "latency_peak": latency.get("max"),
        }
        return {
            "role": self.name,
            "interventions": interventions,
            "summary": summary,
        }
