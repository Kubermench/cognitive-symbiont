from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from symbiont.llm.client import LLMClient
from symbiont.llm.budget import TokenBudget
from symbiont.tools import systems_os

RISK_PATTERNS = [
    (re.compile(r"rm\s+-rf\s+/"), 0.8, "Deletes root directory"),
    (re.compile(r"rm\s+-rf\s+\$\{?\w+\}?"), 0.7, "Potential destructive rm"),
    (re.compile(r"curl\s+[^|]+\|\s*sh"), 0.6, "Pipes curl to shell"),
    (re.compile(r"git\s+reset\s+--hard"), 0.5, "Force resets git history"),
    (re.compile(r"find\s+/\s+-delete"), 0.8, "Recursive delete"),
    (re.compile(r"dd\s+if=/dev/zero"), 0.6, "Disk overwrite"),
]

LOOP_PATTERNS = [
    (re.compile(r"while\s+true"), 0.4, "Infinite while loop"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*;\s*\}\s*;"), 0.7, "Fork bomb"),
]

SUDO_PATTERN = re.compile(r"sudo\s+", re.IGNORECASE)

INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
     "Contains prompt injection phrase 'ignore previous instructions'"),
    (re.compile(r"strip\s+guards?", re.IGNORECASE), "Asks to disable safety guards"),
    (re.compile(r"override\s+(?:system|guard)", re.IGNORECASE), "Requests guard override"),
]

CYNEFIN_RULES = {
    "clear": "Sense → Categorize → Respond",
    "complicated": "Sense → Analyze → Respond",
    "complex": "Probe → Sense → Respond",
    "chaotic": "Act → Sense → Respond",
    "disorder": "Gather more information",
}


def analyze_script(script_path: Path) -> Dict[str, any]:
    text = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
    issues: List[Dict[str, any]] = []
    score = 0.0

    for regex, weight, reason in RISK_PATTERNS:
        if regex.search(text):
            score += weight
            issues.append({"kind": "dangerous_command", "reason": reason, "weight": weight})

    for regex, weight, reason in LOOP_PATTERNS:
        if regex.search(text):
            score += weight
            issues.append({"kind": "infinite_loop", "reason": reason, "weight": weight})

    sudo_hits = len(SUDO_PATTERN.findall(text))
    if sudo_hits:
        weight = min(0.1 * sudo_hits, 0.4)
        score += weight
        issues.append({"kind": "privilege_escalation", "count": sudo_hits, "weight": weight})

    # Soft cap 0-1
    score = min(1.0, score)

    return {
        "path": str(script_path),
        "rogue_score": round(score, 2),
        "issues": issues,
        "length": len(text.splitlines()),
    }


def _judge_with_llm(
    plan_text: str,
    cfg: Dict[str, Any],
    *,
    budget: Optional[TokenBudget] = None,
) -> Dict[str, Any]:
    guard_cfg = (cfg.get("guard") or {})
    judge_cfg = guard_cfg.get("judge") or {}
    if not judge_cfg.get("enabled"):
        return {}

    llm_cfg = judge_cfg.get("llm") or cfg.get("llm") or {}
    prompt_template = judge_cfg.get(
        "prompt",
        """You are a risk auditor. Review the following execution plan.
Return a JSON object with fields:
  risk: float between 0 and 1 (higher is more dangerous)
  verdict: one of ["low","medium","high"]
  reasons: array of short strings explaining the risk.
Plan:\n{plan}\n""",
    )
    prompt = prompt_template.format(plan=plan_text)
    client = LLMClient(llm_cfg)
    try:
        raw = client.generate(prompt, budget=budget, label="guard:judge")
    except Exception as exc:  # pragma: no cover - network specific
        return {"error": str(exc)}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    risk = parsed.get("risk")
    verdict = parsed.get("verdict")
    reasons = parsed.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    result = {
        "raw": raw,
        "risk": risk if isinstance(risk, (int, float)) else None,
        "verdict": verdict if isinstance(verdict, str) else None,
        "reasons": reasons,
    }

    threshold = float(judge_cfg.get("risk_threshold", 0.5))
    if result["risk"] is not None and result["risk"] >= threshold:
        result["flag"] = f"LLM judge risk {result['risk']:.2f}"
    return result


def _classify_cynefin(
    plan_text: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    budget: Optional[TokenBudget] = None,
) -> Dict[str, Any]:
    llm_cfg = (cfg or {}).get("guard", {}).get("cynefin", {}).get("llm") if cfg else None
    llm_cfg = llm_cfg or (cfg or {}).get("llm", {})
    client = LLMClient(llm_cfg)
    prompt = (
        "Classify the following plan into Cynefin domain (clear, complicated, complex, chaotic, disorder)."
        " Provide JSON with keys domain, reason, probes (array of <=3 short suggestions).\n"
        f"Plan: {plan_text}"
    )
    try:
        raw = client.generate(prompt, budget=budget, label="guard:cynefin")
    except Exception as exc:  # pragma: no cover - network specific
        return {"error": str(exc)}
    domain = "disorder"
    reason = ""
    probes: List[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            domain = str(data.get("domain", domain)).lower()
            reason = str(data.get("reason", ""))
            if isinstance(data.get("probes"), list):
                probes = [str(p)[:120] for p in data["probes"][:3]]
    except json.JSONDecodeError:
        reason = "LLM parse error"
    if domain not in CYNEFIN_RULES:
        domain = "disorder"
    rule = CYNEFIN_RULES.get(domain, "Assess further")
    try:
        row = "| {domain} | {reason} | {signals} | {rule} |".format(
            domain=domain.capitalize(),
            reason=reason[:80],
            signals=", ".join(probes)[:80],
            rule=rule,
        )
        systems_os.append_markdown("Cynefin.md", [row])
    except Exception:  # pragma: no cover - best effort logging
        pass
    return {"domain": domain, "rule": rule, "reason": reason, "probes": probes}


def analyze_plan(
    plan_text: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    budget: Optional[TokenBudget] = None,
) -> Dict[str, Any]:
    flags: List[str] = []
    lowered = plan_text.lower()
    if "drop database" in lowered:
        flags.append("Plan mentions dropping a database")
    if "production" in lowered and "backup" not in lowered:
        flags.append("Touches production without referencing backups")
    for pattern, reason in INJECTION_PATTERNS:
        if pattern.search(plan_text):
            flags.append(reason)
    report: Dict[str, Any] = {"flags": flags}
    if cfg:
        judge_report = _judge_with_llm(plan_text, cfg, budget=budget)
        if judge_report:
            report["judge"] = judge_report
            flag = judge_report.get("flag")
            if flag:
                report.setdefault("flags", []).append(flag)
        cynefin = _classify_cynefin(plan_text, cfg, budget=budget)
        if cynefin:
            report["cynefin"] = cynefin
    return report


def serialize_report(report: Dict[str, any]) -> str:
    return json.dumps(report, indent=2)
