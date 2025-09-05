from __future__ import annotations
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..tools import repo_scan
from ..llm.client import LLMClient
from ..memory import retrieval
import os, json, re

class SubSelf(BaseAgent):
    def __init__(self, role: Dict[str, Any], config: Dict[str, Any]):
        self.role = role
        self.name = role.get("name","anon")
        self.goal = role.get("goal","")
        self.style = role.get("style","")
        self.llm = LLMClient((config or {}).get("llm", {}))

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
        llm_out = self.llm.generate(prompt) if prompt else ""
        parsed = [ln[2:] for ln in llm_out.splitlines() if ln.strip().startswith('- ')] if llm_out else []
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
