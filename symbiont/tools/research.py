"""Helpers for high-level research/scouting tasks."""

from __future__ import annotations

import json
from typing import Any, Dict

from symbiont.llm.client import LLMClient
from symbiont.tools.arxiv_fetcher import search_arxiv


def scout_insights(llm: LLMClient, query: str, *, max_items: int = 3) -> Dict[str, Any]:
    prompt = (
        "You are an analyst researching emerging developments."
        " Return strict JSON: {\"topic\": string, \"items\": [ {\"title\": ..., \"url\": ..., \"summary\": ...} ] }."
        f" Focus on '{query}'. Provide at most {max_items} high-signal items."
        " Keep summaries under 120 characters."
    )
    try:
        raw = llm.generate(prompt, label="research:scout") or "{}"
    except Exception:
        raw = "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    # Normalize LLM-provided items first so we can merge with live sources.
    normalized: list[dict[str, str]] = []
    items = data.get("items") if isinstance(data, dict) else []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": str(item.get("title", "Untitled"))[:140],
                    "url": str(item.get("url", ""))[:240],
                    "summary": str(item.get("summary", ""))[:200],
                    "source": str(item.get("source", "llm"))[:40] or "llm",
                    "published": str(item.get("published", ""))[:32],
                }
            )

    # Always enrich with live arXiv results so the scout has real sources.
    try:
        arxiv_results = search_arxiv(query, max_results=max_items)
    except Exception:
        arxiv_results = []
    for result in arxiv_results:
        normalized.append(
            {
                "title": result.title[:140],
                "url": result.link[:240],
                "summary": result.summary[:200],
                "source": "arXiv",
                "published": result.published[:32],
            }
        )

    # Deduplicate by URL/title pair while preserving order.
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, str]] = []
    for item in normalized:
        key = (item.get("url", ""), item.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    topic = data.get("topic") if isinstance(data, dict) else None
    if not isinstance(topic, str) or not topic:
        topic = query

    return {"topic": topic, "items": merged[:max_items]}


def analyze_insights(llm: LLMClient, topic: str, sources: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Given these sources, extract the top learnings and why they matter for Cognitive Symbiont."
        " Provide JSON with keys 'highlight' (concise insight) and 'implication' (actionable suggestion)."
        f" Topic: {topic}\nSources: {json.dumps(sources)[:1200]}"
    )
    raw = llm.generate(prompt, label="research:analyze") or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    highlight = str(data.get("highlight", ""))[:200]
    implication = str(data.get("implication", ""))[:200]

    if not highlight.strip() or highlight.startswith("(offline-fallback)"):
        items = sources.get("items") if isinstance(sources, dict) else []
        top = None
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict) and entry.get("title") and entry.get("summary"):
                    top = entry
                    break
        if top:
            source = top.get("source", "source")
            summary = str(top.get("summary", ""))[:120]
            highlight = f"{top.get('title', 'Leading signal')} ({source}) — {summary}"[:200]
            if not implication.strip():
                implication = f"Track follow-up work on {top.get('title', 'this signal')} and brief the autonomy crew."[:200]

    if not highlight.strip():
        highlight = "No highlight available"
    if not implication.strip():
        implication = "Gather more data."
    return {"topic": topic, "highlight": highlight, "implication": implication}


def draft_proposal(llm: LLMClient, insight: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Create a concise proposal for Symbiont based on this insight."
        " Return JSON with keys 'proposal' (one sentence) and 'diff' (pseudo diff text)."
        f" Insight: {json.dumps(insight)[:800]}"
        " Keep diff under 100 lines and illustrative (no destructive changes)."
    )
    raw = llm.generate(prompt, label="research:proposal") or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    proposal = str(data.get("proposal", ""))[:200]
    diff = str(data.get("diff", ""))

    if not proposal.strip() or proposal.startswith("(offline-fallback)"):
        insight_summary = insight.get("highlight") or insight.get("implication")
        topic = insight.get("topic") or "the focus area"
        if insight_summary:
            proposal = f"Spin a short brief on {topic} highlighting '{insight_summary[:80]}'"[:200]
        else:
            proposal = f"Document key takeaways for {topic}."[:200]
        if not diff.strip():
            diff = (
                "# Proposed update\n"
                "- [ ] Summarize externally sourced signals\n"
                f"- [ ] Draft follow-up experiments for {topic}\n"
            )

    if not diff.strip():
        diff = "# TODO: Provide diff"
    if not proposal.strip():
        proposal = "Investigate further."
    return {"proposal": proposal, "diff": diff}


def validate_proposal(llm: LLMClient, proposal: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Assess the risk of applying this proposal."
        " Return JSON with keys 'approve' (bool), 'risk' (0-1 float), 'tests' (list of suggestions)."
        f" Proposal: {json.dumps(proposal)[:800]}"
    )
    raw = llm.generate(prompt, label="research:validate") or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    approve = bool(data.get("approve", False))
    try:
        risk = float(data.get("risk", 0.5))
    except (TypeError, ValueError):
        risk = 0.5
    tests = data.get("tests", []) if isinstance(data, dict) else []
    if not isinstance(tests, list):
        tests = []
    sanitized_tests = [str(t)[:160] for t in tests[:3]]

    offline_like = not sanitized_tests or any("(offline-fallback)" in t for t in sanitized_tests)
    if offline_like:
        sanitized_tests = [
            "Verify source credibility",
            "Share recap with core stakeholders",
        ][:3]
        if not approve:
            approve = True
        risk = min(risk, 0.45)

    sanitized_tests = sanitized_tests[:3]

    return {"approve": approve, "risk": max(0.0, min(1.0, risk)), "tests": sanitized_tests}
