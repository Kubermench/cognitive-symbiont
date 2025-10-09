from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
from .base_agent import BaseAgent
from ..tools import repo_scan, research
from ..llm.client import LLMClient
from ..llm.budget import TokenBudget
from ..memory import retrieval
import os, json, re
from collections import Counter

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
        self.config = config or {}
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
        if self.name == "loop_mapper":
            return self._loop_mapper(goal, context, memory)
        if self.name == "leverage_ranker":
            return self._leverage_ranker(goal, context)
        if self.name == "cynefin_classifier":
            return self._cynefin_classifier(goal, context)
        if self.name == "cynefin_planner":
            return self._cynefin_planner(goal, context)
        if self.name == "model_challenger":
            return self._model_challenger(goal, context)
        if self.name == "success_miner":
            return self._success_miner(goal, context)
        if self.name == "coupling_analyzer":
            return self._coupling_analyzer(goal, context)
        if self.name == "flow_analyzer":
            return self._flow_analyzer(goal, context)
        if self.name == "foresight_scout":
            return self._foresight_scout(goal, context)
        if self.name == "foresight_analyzer":
            return self._foresight_analyzer(goal, context)
        if self.name == "foresight_suggester":
            return self._foresight_suggester(goal, context)
        if self.name == "foresight_validator":
            return self._foresight_validator(goal, context)
        if self.name == "foresight_evolver":
            return self._foresight_evolver(goal, context)
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

    def _loop_mapper(self, goal: str, context: Dict[str, Any], memory) -> Dict[str, Any]:
        evidences = []
        for r in retrieval.search(memory, goal, k=3):
            evidences.append({"kind": r["kind"], "preview": r["preview"][:120], "score": r["score"]})
        prompt = (
            "You are a systems thinker mapping feedback loops for the following goal:\n"
            f"Goal: {goal}\n"
            f"Evidence: {json.dumps(evidences)[:600]}\n"
            "Return strict JSON with key 'loops' as a list. Each item must contain"
            " name, type ('reinforcing' or 'balancing'), stocks (list of strings),"
            " flows (list of strings), and note."
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:loops",
        ) or "{}"
        loops: List[Dict[str, Any]] = []
        try:
            payload = json.loads(raw)
            loops = payload.get("loops", []) if isinstance(payload, dict) else []
        except json.JSONDecodeError:
            loops = []
        clean_loops: List[Dict[str, Any]] = []
        for loop in loops:
            if not isinstance(loop, dict):
                continue
            clean_loops.append(
                {
                    "name": str(loop.get("name", "Unnamed Loop"))[:80],
                    "type": str(loop.get("type", "unknown")),
                    "stocks": [str(s) for s in loop.get("stocks", [])][:5],
                    "flows": [str(f) for f in loop.get("flows", [])][:5],
                    "note": str(loop.get("note", ""))[:200],
                }
            )
        if not clean_loops and evidences:
            clean_loops.append(
                {
                    "name": "Baseline Loop",
                    "type": "reinforcing",
                    "stocks": [goal[:40]],
                    "flows": [evidences[0]["preview"][:40]],
                    "note": "LLM parsing failed; placeholder from evidence",
                }
            )
        context["systems_loops"] = clean_loops
        return {"role": self.name, "loops": clean_loops, "evidence": evidences}

    def _leverage_ranker(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        loops = context.get("systems_loops", [])
        prompt = (
            "Rank leverage points for the goal using Meadows' framework."
            " Focus on paradigm or rule changes over parameters.\n"
            f"Goal: {goal}\n"
            f"Loops: {json.dumps(loops)[:800]}\n"
            "Return strict JSON with key 'leverage_points' as a list of objects"
            " each containing name, description, effort (1-5), impact (1-5)."
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:leverage",
        ) or "{}"
        leverage_points: List[Dict[str, Any]] = []
        try:
            payload = json.loads(raw)
            leverage_points = payload.get("leverage_points", []) if isinstance(payload, dict) else []
        except json.JSONDecodeError:
            leverage_points = []
        clean_points: List[Dict[str, Any]] = []
        for point in leverage_points:
            if not isinstance(point, dict):
                continue
            effort = float(point.get("effort", 3) or 3)
            impact = float(point.get("impact", 3) or 3)
            ratio = impact / effort if effort else impact
            clean_points.append(
                {
                    "name": str(point.get("name", "Leverage"))[:80],
                    "description": str(point.get("description", ""))[:200],
                    "effort": round(effort, 2),
                    "impact": round(impact, 2),
                    "ratio": round(ratio, 3),
                }
            )
        clean_points.sort(key=lambda p: p["ratio"], reverse=True)
        context["leverage_points"] = clean_points
        return {"role": self.name, "leverage_points": clean_points}

    def _cynefin_classifier(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        recent_signals = context.get("signals") or goal
        prompt = (
            "Classify the situation into a Cynefin domain (clear, complicated, complex, chaotic, disorder)."
            " Provide JSON with keys domain, reason, signals (list of 2 short bullet strings)."
            f" Situation: {recent_signals}"
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:cynefin",
        ) or "{}"
        domain = "disorder"
        reason = ""
        signals: List[str] = []
        try:
            data = json.loads(raw)
            domain = str(data.get("domain", domain)).lower()
            reason = str(data.get("reason", ""))
            raw_signals = data.get("signals", [])
            if isinstance(raw_signals, list):
                signals = [str(s)[:80] for s in raw_signals[:3]]
        except json.JSONDecodeError:
            reason = "LLM parse error"
        if domain not in {"clear", "obvious", "complicated", "complex", "chaotic", "disorder"}:
            domain = "disorder"
        if domain == "obvious":
            domain = "clear"
        context["cynefin_domain"] = domain
        context.setdefault("cynefin_signals", signals)
        context.setdefault("cynefin_reason", reason)
        return {"role": self.name, "domain": domain, "reason": reason, "signals": signals}

    def _cynefin_planner(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        domain = context.get("cynefin_domain", "disorder")
        signals = context.get("cynefin_signals", [])
        reason = context.get("cynefin_reason", "")
        rulebook = {
            "clear": "Sense → Categorize → Respond; apply best practice",
            "complicated": "Sense → Analyze → Respond; involve experts",
            "complex": "Probe → Sense → Respond; run safe-to-fail experiments",
            "chaotic": "Act → Sense → Respond; stabilize immediately",
            "disorder": "Gather more data to classify",
        }
        prompt = (
            "Given the Cynefin domain, suggest up to three concrete next actions."
            f" Domain: {domain}. Rule: {rulebook.get(domain, 'Assess further')}"
            f" Goal: {goal}. Signals: {signals}."
            " Return JSON with keys rule, actions (list of strings), probes (list of strings)."
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:actions",
        ) or "{}"
        rule = rulebook.get(domain, "Assess further")
        actions: List[str] = []
        probes: List[str] = []
        try:
            data = json.loads(raw)
            rule = str(data.get("rule", rule))
            if isinstance(data.get("actions"), list):
                actions = [str(a)[:120] for a in data["actions"][:3]]
            if isinstance(data.get("probes"), list):
                probes = [str(p)[:120] for p in data["probes"][:3]]
        except json.JSONDecodeError:
            actions = ["Document domain classification; schedule follow-up"]
        context["cynefin_rule"] = rule
        context["cynefin_actions"] = actions
        context["cynefin_probes"] = probes
        return {"role": self.name, "rule": rule, "actions": actions, "probes": probes}

    def _model_challenger(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "Challenge the mental model described. Provide JSON with keys model, counter_bet,"
            " experiment (short description), and signal (what to observe)."
            f" Mental model: {goal}"
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:challenge",
        ) or "{}"
        model = goal
        counter = ""
        experiment = ""
        signal = ""
        try:
            data = json.loads(raw)
            model = str(data.get("model", model))
            counter = str(data.get("counter_bet", counter))
            experiment = str(data.get("experiment", experiment))
            signal = str(data.get("signal", signal))
        except json.JSONDecodeError:
            counter = "Collect data to validate assumption"
            experiment = "Interview 3 users"
        context["mental_model"] = {
            "model": model,
            "counter_bet": counter,
            "experiment": experiment,
            "signal": signal,
        }
        return {"role": self.name, "model": model, "counter_bet": counter, "experiment": experiment, "signal": signal}

    def _success_miner(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "Produce a Safety-II success entry summarizing what went right, adaptations,"
            " subtle signals, and next improvement. Return JSON with keys what_went_right,"
            " adaptations, signals, next_step."
            f" Context: {goal}"
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:success",
        ) or "{}"
        entry = {
            "what_went_right": "",
            "adaptations": "",
            "signals": "",
            "next_step": "",
        }
        try:
            data = json.loads(raw)
            for key in entry:
                entry[key] = str(data.get(key, ""))
        except json.JSONDecodeError:
            entry["what_went_right"] = goal[:120]
        context["safety_entry"] = entry
        return {"role": self.name, **entry}

    def _coupling_analyzer(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        graph_path = Path(goal.strip()) if goal.strip() else Path("configs/crews.yaml")
        try:
            analysis = __import__("symbiont.tools.coupling_analyzer", fromlist=["analyze"]).analyze(graph_path)
        except Exception as exc:
            return {"role": self.name, "error": str(exc)}
        context["coupling_entries"] = analysis.get("entries", [])
        context["coupling_heat"] = analysis.get("heat", 0.0)
        return {
            "role": self.name,
            "entries": analysis.get("entries", []),
            "heat": analysis.get("heat", 0.0),
            "nodes": analysis.get("nodes", []),
        }

    def _flow_analyzer(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        import json

        flow_path = Path(goal.strip()) if goal.strip() else Path("systems/FlowMetrics.json")
        try:
            data = json.loads(flow_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        prompt = (
            "Given DORA-style metrics, identify risks and suggest actions."
            f" Metrics: {json.dumps(data)}"
            " Return JSON with keys summary, risks (list), actions (list)."
        )
        raw = self.llm.generate(
            prompt,
            budget=self.token_budget,
            label=f"{self.name}:flow",
        ) or "{}"
        summary = ""
        risks: List[str] = []
        actions: List[str] = []
        try:
            payload = json.loads(raw)
            summary = str(payload.get("summary", ""))
            if isinstance(payload.get("risks"), list):
                risks = [str(r)[:120] for r in payload["risks"][:5]]
            if isinstance(payload.get("actions"), list):
                actions = [str(a)[:120] for a in payload["actions"][:5]]
        except json.JSONDecodeError:
            summary = "Unable to parse LLM output"
        context["flow_analysis"] = {"summary": summary, "risks": risks, "actions": actions}
        return {"role": self.name, "summary": summary, "risks": risks, "actions": actions, "metrics": data}

    def _foresight_scout(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        budget = context.get("token_budget")
        before_tokens = budget.used if budget else None
        insights = research.scout_insights(self.llm, goal)
        if budget and before_tokens is not None:
            delta = max(0, budget.used - before_tokens)
            meta = insights.setdefault("meta", {}) if isinstance(insights, dict) else {}
            meta["token_delta"] = delta
            cost_rate = float(self.config.get("foresight", {}).get("token_cost_per_unit", 0.000002))
            meta["cost_estimate"] = round(delta * cost_rate, 6)
        context["foresight_sources"] = insights
        return {"role": self.name, **insights}

    def _foresight_analyzer(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        sources = context.get("foresight_sources") or research.scout_insights(self.llm, goal)
        analysis = research.analyze_insights(self.llm, sources.get("topic", goal), sources)
        context["foresight_analysis"] = analysis
        return {"role": self.name, **analysis}

    def _foresight_suggester(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = context.get("foresight_analysis") or {"highlight": goal, "implication": "Investigate"}
        proposal = research.draft_proposal(self.llm, analysis)
        if not proposal.get("proposal") or not proposal.get("diff"):
            topic = analysis.get("topic", goal) if isinstance(analysis, dict) else goal
            proposal = research.build_fallback_proposal(topic, analysis if isinstance(analysis, dict) else None)

        foresight_cfg = (self.config.get("foresight") or {})
        collaboration_cfg = foresight_cfg.get("collaboration") or {}
        sources = context.get("foresight_sources") or {}
        items = sources.get("items") if isinstance(sources, dict) else []
        peer_votes: List[Dict[str, Any]] = []
        approve_threshold = float(collaboration_cfg.get("approve_threshold", 0.5))
        if isinstance(items, list):
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("source", "")).lower() != "peer":
                    continue
                peer_name = str(entry.get("peer") or entry.get("source") or "peer")
                support = float(entry.get("peer_support", 0.0))
                vote_value = "approve" if support >= approve_threshold else "reject"
                peer_votes.append(
                    {
                        "peer": peer_name,
                        "vote": vote_value,
                        "support": round(support, 3),
                    }
                )

        if peer_votes:
            vote_counts = Counter(vote["vote"] for vote in peer_votes)
            top_vote, top_count = vote_counts.most_common(1)[0]
            total_votes = len(peer_votes)
            consensus_ratio = top_count / total_votes if total_votes else 0.0
            proposal["peer_votes"] = peer_votes
            proposal["peer_consensus"] = {
                "vote": top_vote,
                "confidence": round(consensus_ratio, 3),
                "total": total_votes,
            }
            proposal.setdefault("status", "consensus" if consensus_ratio >= 0.6 else "disputed")
            if consensus_ratio < 0.6:
                proposal["status"] = "disputed"
            elif top_vote != "approve":
                proposal["status"] = "rejected"
        context["foresight_proposal"] = proposal
        return {"role": self.name, **proposal}

    def _foresight_validator(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        proposal = context.get("foresight_proposal") or {"proposal": goal, "diff": "# noop"}
        validation = research.validate_proposal(self.llm, proposal)
        if not validation.get("tests"):
            validation = research.build_fallback_validation(proposal)
        context["foresight_validation"] = validation
        return {"role": self.name, **validation}

    def _foresight_evolver(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        proposal = context.get("foresight_proposal") or {}
        validation = context.get("foresight_validation") or {}
        approved = bool(validation.get("approve")) and float(validation.get("risk", 1.0)) <= 0.5
        result = {
            "approved": approved,
            "proposal": proposal,
            "validation": validation,
        }
        context["foresight_result"] = result
        return {"role": self.name, **result}
