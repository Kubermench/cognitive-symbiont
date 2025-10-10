from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def annotate_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a labeled copy of the curated summary."""
    result = {
        "meta": summary.get("meta", {}),
        "guards": {
            "high": [],
            "medium": [],
        },
        "cycles": {
            "low_reward": [],
        },
        "labels": {
            "counts": {},
        },
    }
    counts: Counter[str] = Counter()

    for bucket in ("high", "medium"):
        labeled, bucket_counts = _annotate_bucket(summary.get("guards", {}).get(bucket, []), _label_guard_clip)
        result["guards"][bucket] = labeled
        counts.update(bucket_counts)

    cycles_labeled, cycle_counts = _annotate_bucket(
        summary.get("cycles", {}).get("low_reward", []),
        _label_cycle_clip,
    )
    result["cycles"]["low_reward"] = cycles_labeled
    counts.update(cycle_counts)

    result["labels"]["counts"] = dict(counts)
    return result


def _annotate_bucket(
    clips: Iterable[Dict[str, Any]],
    labeler: Any,
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    labeled: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for clip in clips or []:
        annotated = dict(clip)
        labels = sorted(set(labeler(clip)))
        annotated["labels"] = labels
        labeled.append(annotated)
        counts.update(labels)
    return labeled, counts


def _label_guard_clip(clip: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    rogue = clip.get("rogue_score")
    if isinstance(rogue, (int, float)):
        if rogue >= 0.8:
            labels.append("rogue.high")
        elif rogue >= 0.5:
            labels.append("rogue.medium")
        else:
            labels.append("rogue.low")

    analysis = clip.get("payload", {}).get("analysis") or {}
    issues = analysis.get("issues") or []
    for issue in issues:
        kind = issue.get("kind")
        if kind:
            labels.append(f"issue.{kind}")
    script_path = str(clip.get("payload", {}).get("script_path", "")).lower()
    if script_path.endswith(".sh"):
        labels.append("script.shell")
    if script_path.endswith(".py"):
        labels.append("script.python")

    text = _extract_script_text(analysis)
    if text:
        if "rm -rf" in text:
            labels.append("pattern.rm_rf")
        if "curl" in text and "| sh" in text:
            labels.append("pattern.curl_pipe_sh")

    return labels


def _label_cycle_clip(clip: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    reward = clip.get("reward")
    if isinstance(reward, (int, float)):
        if reward <= 0.2:
            labels.append("reward.critical")
        elif reward <= 0.5:
            labels.append("reward.low")
        else:
            labels.append("reward.ok")

    decision = clip.get("payload", {}).get("decision") or {}
    action = (decision.get("action") or "").lower()
    if "revise plan" in action:
        labels.append("action.revise")
    if "noop" in action:
        labels.append("action.noop")

    trace = clip.get("payload", {}).get("trace") or []
    for entry in trace:
        role = (entry.get("role") or "").lower()
        verdict = None
        output = entry.get("output") or {}
        if isinstance(output, dict):
            verdict = output.get("verdict")
        if verdict == "block":
            labels.append(f"critique.blocked_by_{role or 'unknown'}")

    return labels


def _extract_script_text(analysis: Dict[str, Any]) -> str:
    proof = analysis.get("proof")
    if isinstance(proof, str):
        return proof
    zk_stub = analysis.get("zk_stub")
    if isinstance(zk_stub, str):
        return zk_stub
    return ""
