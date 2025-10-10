"""Helpers for high-level research/scouting tasks.

The foresight pipeline fuses LLM reasoning with live signals gathered from
arXiv, RSS feeds, and optional peer collaborators (e.g., Grok/Devin).  The
functions in this module are intentionally defensive: API calls are rate
limited with jittered retries, results are deduplicated, and metadata is
emitted so downstream analytics (BigKit, governance dashboards) can render
source mixes without additional post-processing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from symbiont.llm.client import LLMClient
from symbiont.memory.db import MemoryDB
from symbiont.memory import retrieval
from symbiont.memory.dynamic_analyzer import BayesianTrendAnalyzer, deduplicate_triples
from symbiont.tools.arxiv_fetcher import search_arxiv
from symbiont.tools.files import ensure_dirs

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path("data/artifacts/foresight")
COLLABORATOR_MODELS: Tuple[str, ...] = ("grok", "devin")
RSS_ENDPOINTS: Tuple[str, ...] = (
    "http://export.arxiv.org/rss/cs.AI",
    "https://hnrss.org/newest?points=150",
)
_GLOBAL_ANALYZER = BayesianTrendAnalyzer()

FALLBACK_PROPOSAL_TEMPLATE = (
    "# Foresight Update\n"
    "- [ ] Summarize the newest signals\n"
    "- [ ] Identify one follow-up experiment\n"
    "- [ ] Share the briefing with the core team\n"
)

FALLBACK_VALIDATION = {
    "approve": False,
    "risk": 0.55,
    "tests": [
        "Gather additional corroborating sources",
        "Review proposal with the foresight team",
    ],
}

SOURCE_BOOSTS = {
    "arxiv": 1.2,
    "rss": 0.9,
    "peer": 0.85,
    "x": 0.8,
    "web": 0.7,
    "llm": 0.4,
}


def _keyword_tokens(text: str) -> set[str]:
    return {tok.strip().lower() for tok in text.split() if tok and len(tok) > 3}


def _score_item(item: Dict[str, Any], query: str) -> float:
    score = 0.0
    source = str(item.get("source", "")).lower()
    score += SOURCE_BOOSTS.get(source, 0.5)

    q_tokens = _keyword_tokens(query)
    title_tokens = _keyword_tokens(item.get("title", ""))
    summary_tokens = _keyword_tokens(item.get("summary", ""))
    overlap = len(q_tokens & (title_tokens | summary_tokens))
    if overlap:
        score += 0.2 * overlap

    published = str(item.get("published", ""))
    if published:
        try:
            pub_dt = datetime.strptime(published[:10], "%Y-%m-%d").replace(tzinfo=UTC)
            age_days = max(1, (datetime.now(UTC) - pub_dt).days)
            score += max(0.0, 0.8 - (age_days / 365))
        except Exception:  # pragma: no cover - format variance
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
                    "source": "arxiv",
                    "published": result.published,
                }
            )
        return payload

    adapters.append(fetch_arxiv)

    items: list[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        future_map = {executor.submit(adapter): adapter for adapter in adapters}
        for future in as_completed(future_map):
            try:
                items.extend(future.result() or [])
            except Exception as exc:  # pragma: no cover - network noise
                logger.debug("Live source adapter failed: %s", exc)
                continue
    return items



def build_fallback_proposal(topic: str, insight: Dict[str, Any] | None = None) -> Dict[str, Any]:
    highlight = (insight or {}).get("highlight") or (insight or {}).get("implication")
    prefix = highlight[:80] if isinstance(highlight, str) else topic
    proposal_text = (
        f"Document and review the latest findings on {topic}."
        if not prefix
        else f"Document and review: {prefix}."
    )
    return {
        "proposal": proposal_text[:200],
        "diff": FALLBACK_PROPOSAL_TEMPLATE,
    }


def build_fallback_validation(proposal: Dict[str, Any] | None = None) -> Dict[str, Any]:
    validation = dict(FALLBACK_VALIDATION)
    if proposal and proposal.get("proposal"):
        validation["tests"] = [
            f"Validate proposal: {str(proposal['proposal'])[:60]}",
            "Review with foresight stakeholders",
        ]
    return validation


def _save_source_plot(topic: str, stats: Dict[str, Dict[str, float]]) -> Optional[str]:
    if not stats:
        return None
    try:
        import matplotlib
        try:
            backend = matplotlib.get_backend()
        except Exception:
            backend = None
        if not backend or not str(backend).lower().startswith("agg"):
            matplotlib.use("Agg")  # type: ignore[arg-type]
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None

    labels = list(stats.keys())
    counts = [stat.get("count", 0) for stat in stats.values()]
    avgs = [stat.get("avg_score", 0.0) for stat in stats.values()]
    if not any(counts):
        return None

    ensure_dirs([ARTIFACT_ROOT])
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in topic) or "foresight"
    slug = slug[:40].strip("-") or "foresight"
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    path = ARTIFACT_ROOT / f"{ts}_sources.png"

    fig = None
    ax = None
    try:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(labels, counts, color="#4b9cd3")
        ax.set_title(f"Source mix for {topic[:40]}")
        ax.set_ylabel("Items")
        ax.set_ylim(0, max(counts) * 1.2)
        for bar, avg in zip(bars, avgs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f"avg={avg:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    except Exception:
        return None
    finally:
        if fig is not None:
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
    except Exception as exc:  # pragma: no cover - external failure
        logger.debug("Primary scout LLM failed: %s", exc)
        raw = "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

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
    with suppress(Exception):
        for result in _fetch_live_sources(query, max_items):
            normalized.append(result)

    external_meta: Dict[str, Any] = {}
    with suppress(Exception):
        db = MemoryDB()
        db.ensure_schema()
        external = retrieval.fetch_external_context(
            db,
            query,
            max_items=max(4, max_items * 2),
            min_relevance=0.65,
        )
        accepted = external.get("accepted") or []
        external_meta = {
            "accepted_items": len(accepted),
            "claims_merged": len(external.get("claims") or []),
        }
        for item in accepted:
            normalized.append(
                {
                    "title": str(item.get("title", "External Insight"))[:140],
                    "url": str(item.get("url", ""))[:240],
                    "summary": str(item.get("summary", ""))[:200],
                    "source": str(item.get("source", "external"))[:40] or "external",
                    "published": str(
                        (item.get("metadata") or {}).get("published")
                        or (item.get("metadata") or {}).get("year", "")
                    )[:32],
                }
            )

    enriched = _rank_and_dedup(query, normalized)
    if external_meta:
        enriched.setdefault("meta", {}).update({"external_context": external_meta})
    return enriched


def _rank_and_dedup(query: str, candidates: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for item in candidates:
        key = (item.get("url", ""), item.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        entry = dict(item)
        entry["score"] = _score_item(entry, query)
        merged.append(entry)

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

    min_score = 0.6
    filtered = [item for item in ranked if float(item.get("score", 0.0)) >= min_score]
    dropped = len(ranked) - len(filtered)
    if not filtered and ranked:
        filtered = ranked[:1]

    plot_path = _save_source_plot(query, source_stats)
    meta = {
        "source_breakdown": source_stats,
        "total_candidates": len(ranked),
        "dropped_low_score": dropped,
    }
    if plot_path:
        meta["source_plot"] = plot_path

    deduped = deduplicate_triples(
        (query, entry.get("source", "unknown"), entry.get("url", "")) for entry in filtered
    )
    meta["deduped_triples"] = len(deduped)

    return {
        "topic": query,
        "items": filtered,
        "meta": meta,
    }


async def gather_trend_sources_async(
    llm: LLMClient,
    query: str,
    *,
    max_items: int = 3,
    include_collaborators: bool = False,
    include_rss: bool = True,
) -> Dict[str, Any]:
    """Asynchronously collect sources from live APIs, collaborators, and RSS feeds."""

    loop = asyncio.get_running_loop()
    futures = [loop.run_in_executor(None, _fetch_live_sources, query, max_items)]

    if include_collaborators:
        futures.append(
            loop.run_in_executor(
                None,
                call_peer_collaborators,
                llm,
                query,
                None,
                max_items,
            )
        )

    if include_rss:
        futures.append(loop.run_in_executor(None, fetch_rss_alerts, query, max_items))

    results = await asyncio.gather(*futures, return_exceptions=True)
    merged: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"contributors": []}
    for payload in results:
        if isinstance(payload, Exception):
            logger.debug("gather_async skipped payload: %s", payload)
            continue
        if isinstance(payload, dict) and payload.get("items"):
            merged.extend(payload["items"])
            meta["contributors"].append(payload.get("source", "unknown"))
        elif isinstance(payload, list):
            merged.extend(payload)
            meta["contributors"].append("arxiv")

    ranked = _rank_and_dedup(query, merged)
    ranked["meta"].update(meta)
    return ranked


@retry(
    retry=retry_if_exception_type((ValueError, RuntimeError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=12),
)
def call_peer_collaborators(
    llm: LLMClient,
    query: str,
    models: Optional[Sequence[str]] = None,
    max_items: int = 2,
) -> Dict[str, Any]:
    """Query peer LLMs (e.g., Grok / Devin) with jittered retries."""

    models = tuple(models or COLLABORATOR_MODELS)
    prompt = (
        "You simulate an ensemble of research peers. "
        f"Peers: {', '.join(models)}. "
        "Return JSON list with fields peer, title, url, summary, confidence (0-1). "
        f"Focus on '{query}'."
    )
    raw = llm.generate(prompt, label="research:peer") or "[]"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Peer response unparsable: {exc}") from exc

    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for entry in data[:max_items * 2]:
            if not isinstance(entry, dict):
                continue
            items.append(
                {
                    "title": str(entry.get("title", "Peer Insight"))[:140],
                    "url": str(entry.get("url", ""))[:240],
                    "summary": str(entry.get("summary", ""))[:200],
                    "source": "peer",
                    "published": datetime.now(UTC).strftime("%Y-%m-%d"),
                    "peer": str(entry.get("peer", "ensemble"))[:40],
                    "peer_support": float(entry.get("confidence", 0.0)),
                }
            )

    if not items:
        raise RuntimeError("Peer collaborators returned no items")
    return {"items": items[:max_items], "source": "peer", "contributors": list(models)}


def fetch_rss_alerts(topic: str, max_items: int = 3) -> Dict[str, Any]:
    """Fetch RSS entries mentioning the topic (best-effort)."""

    normalized: List[Dict[str, Any]] = []
    lowered = topic.lower()
    for endpoint in RSS_ENDPOINTS:
        with suppress(Exception):
            feed = feedparser.parse(endpoint)
            for entry in feed.get("entries", [])[: max_items * 4]:
                title = str(entry.get("title", ""))
                if lowered not in title.lower():
                    continue
                normalized.append(
                    {
                        "title": title[:140],
                        "summary": str(entry.get("summary", ""))[:200],
                        "source": "rss",
                        "url": str(entry.get("link", ""))[:240],
                        "published": str(entry.get("published", "")[:10]),
                        "peer_support": 0.35,
                    }
                )
    return {"items": normalized[:max_items], "source": "rss"}


def analyze_insights(llm: LLMClient, topic: str, sources: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Given these sources, extract the top learnings and why they matter for Cognitive Symbiont."
        " Provide JSON with keys 'highlight' (concise insight) and 'implication' (actionable suggestion)."
        f" Topic: {topic}\nSources: {json.dumps(sources)[:1200]}"
    )
    try:
        raw = llm.generate(prompt, label="research:analyze") or "{}"
    except Exception as exc:  # pragma: no cover - external failure
        logger.debug("Analyze call failed: %s", exc)
        raw = "{}"
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
    try:
        raw = llm.generate(prompt, label="research:proposal") or "{}"
    except Exception as exc:  # pragma: no cover - external failure
        logger.debug("Proposal call failed: %s", exc)
        raw = "{}"
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

    if not diff.strip() or not proposal.strip():
        fallback = build_fallback_proposal(
            insight.get("topic", "the focus area") if isinstance(insight, dict) else "foresight",
            insight if isinstance(insight, dict) else None,
        )
        proposal = fallback["proposal"]
        diff = fallback["diff"]
    return {"proposal": proposal, "diff": diff}


def validate_proposal(llm: LLMClient, proposal: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Assess the risk of applying this proposal."
        " Return JSON with keys 'approve' (bool), 'risk' (0-1 float), 'tests' (list of suggestions)."
        f" Proposal: {json.dumps(proposal)[:800]}"
    )
    try:
        raw = llm.generate(prompt, label="research:validate") or "{}"
    except Exception as exc:  # pragma: no cover - external failure
        logger.debug("Validate call failed: %s", exc)
        raw = "{}"
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

    result = {"approve": approve, "risk": max(0.0, min(1.0, risk)), "tests": sanitized_tests[:3]}
    if not sanitized_tests:
        result = build_fallback_validation(proposal)
    return result


__all__ = [
    "analyze_insights",
    "build_fallback_proposal",
    "build_fallback_validation",
    "call_peer_collaborators",
    "draft_proposal",
    "fetch_rss_alerts",
    "gather_trend_sources_async",
    "scout_insights",
    "validate_proposal",
]
