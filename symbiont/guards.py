from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

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


def analyze_plan(plan_text: str) -> Dict[str, any]:
    flags: List[str] = []
    lowered = plan_text.lower()
    if "drop database" in lowered:
        flags.append("Plan mentions dropping a database")
    if "production" in lowered and "backup" not in lowered:
        flags.append("Touches production without referencing backups")
    return {"flags": flags}


def serialize_report(report: Dict[str, any]) -> str:
    return json.dumps(report, indent=2)

