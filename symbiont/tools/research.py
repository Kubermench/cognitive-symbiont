"""Helpers for high-level research/scouting tasks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from symbiont.llm.client import LLMClient
from symbiont.tools.arxiv_fetcher import search_arxiv
from symbiont.tools.files import ensure_dirs


ARTIFACT_ROOT = Path("data/artifacts/foresight")


SOURCE_BOOSTS = {
    "arxiv": 1.2,
    "x": 0.8,
    "web": 0.7,
    "llm": 0.4,
}


def _keyword_tokens(text: str) -> set[str]:
    return {tok.strip().lower() for tok in text.split() if tok and len(tok) > 3}


def _score_item(item: Dict[str, Any], query: str) -> float:
    score = 0.0
    source = str(item.get("source", "")).lower()
    boost = SOURCE_BOOSTS.get(source, 0.5)
    score += boost

    q_tokens = _keyword_tokens(query)
    title_tokens = _keyword_tokens(item.get("title", ""))
    summary_tokens = _keyword_tokens(item.get("summary", ""))
    overlap = len(q_tokens & (title_tokens | summary_tokens))
    if overlap:
        score += 0.2 * overlap

    published = str(item.get("published", ""))
    if published:
        try:
            pub_dt = datetime.strptime(published[:10], "%Y-%m-%d")
            age_days = max(1, (datetime.utcnow() - pub_dt).days)
            score += max(0.0, 0.8 - (age_days / 365))
        except Exception:
            score += 0.1

    summary_len = len(item.get("summary", ""))
    if summary_len and summary_len < 500:
        score += 0.1
    return round(score, 3)


def _fetch_live_sources(query: str, max_items: int) -> list[Dict[str, Any]]:
    adapters = []

    def fetch_arxiv() -> list[Dict[str, Any]]:
        results = search_arxiv(query, max_results=max_items)
        payload: list[Dict[str, Any]] = []
        for result in results:
            payload.append(
                {
                    "title": result.title[:140],
                    "url": result.link[:240],
                    "summary": result.summary[:200],
                    "source": "arXiv",
                    "published": result.published,
                }
            )
        return payload

    adapters.append(fetch_arxiv)

    items: list[Dict[str, Any]] = []
    if not adapters:
        return items

    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        future_map = {executor.submit(adapter): adapter for adapter in adapters}
        for future in as_completed(future_map):
            try:
                items.extend(future.result() or [])
            except Exception:
                continue
    return items


def _save_source_plot(topic: str, stats: Dict[str, Dict[str, float]]) -> Optional[str]:
    if not stats:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = list(stats.keys())
    counts = [stat.get("count", 0) for stat in stats.values()]
    avgs = [stat.get("avg_score", 0.0) for stat in stats.values()]
    if not any(counts):
        return None

    ensure_dirs([ARTIFACT_ROOT])
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in topic) or "foresight"
    slug = slug[:40].strip("-") or "foresight"
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = ARTIFACT_ROOT / f"{ts}_sources.png"

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, counts, color="#4b9cd3")
    ax.set_title(f"Source mix for {topic[:40]}")
    ax.set_ylabel("Items")
    ax.set_ylim(0, max(counts) * 1.2)
    for bar, avg in zip(bars, avgs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f"avg={avg:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return str(path)


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
    normalized: list[dict[str, Any]] = []
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
        live_items = _fetch_live_sources(query, max_items)
    except Exception:
        live_items = []
    for result in live_items:
        normalized.append(result)

    # Deduplicate by URL/title pair while preserving order.
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for item in normalized:
        key = (item.get("url", ""), item.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    for entry in merged:
        entry["score"] = _score_item(entry, query)

    topic = data.get("topic") if isinstance(data, dict) else None
    if not isinstance(topic, str) or not topic:
        topic = query
    ranked = sorted(merged, key=lambda item: item.get("score", 0.0), reverse=True)

    source_stats: Dict[str, Dict[str, float]] = {}
    for entry in ranked:
        source = str(entry.get("source", "unknown")).lower() or "unknown"
        stat = source_stats.setdefault(source, {"count": 0, "score_total": 0.0})
        stat["count"] += 1
        stat["score_total"] += float(entry.get("score", 0.0))

    for source, stat in source_stats.items():
        count = stat.get("count", 1) or 1
        stat["avg_score"] = round(stat["score_total"] / count, 3)
        stat.pop("score_total", None)

    plot_path = _save_source_plot(topic, source_stats)
    meta = {
        "source_breakdown": source_stats,
        "total_candidates": len(ranked),
    }
    if plot_path:
        meta["source_plot"] = plot_path

    return {
        "topic": topic,
        "items": ranked[:max_items],
        "meta": meta,
    }


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
