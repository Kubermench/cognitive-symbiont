from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from symbiont.llm.client import LLMClient

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


def _judge_with_llm(plan_text: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
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
        raw = client.generate(prompt)
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


def analyze_plan(plan_text: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    flags: List[str] = []
    lowered = plan_text.lower()
    if "drop database" in lowered:
        flags.append("Plan mentions dropping a database")
    if "production" in lowered and "backup" not in lowered:
        flags.append("Touches production without referencing backups")
    report: Dict[str, Any] = {"flags": flags}
    if cfg:
        judge_report = _judge_with_llm(plan_text, cfg)
        if judge_report:
            report["judge"] = judge_report
            flag = judge_report.get("flag")
            if flag:
                report.setdefault("flags", []).append(flag)
    return report


def serialize_report(report: Dict[str, any]) -> str:
    return json.dumps(report, indent=2)
