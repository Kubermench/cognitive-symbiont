"""Bayesian weighting and dedup helpers for foresight hunts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


OptionalJudge = Callable[[Dict[str, Any]], float]


@dataclass
class BayesianTrendAnalyzer:
    """Rank foresight sources using lightweight Bayesian updating.

    The implementation mirrors the foresight-arxiv design notes: each source
    begins with a prior (roughly reflecting historic quality), then receives
    multiplicative updates for freshness, topical overlap, and peer support.
    """

    priors: Dict[str, float] = field(
        default_factory=lambda: {
            "arxiv": 0.78,
            "peer": 0.68,
            "rss": 0.52,
            "web": 0.35,
            "x": 0.32,
            "llm": 0.25,
        }
    )
    freshness_half_life_days: float = 120.0
    noise_floor: float = 0.08
    hype_threshold: float = 0.6

    def rank_sources(self, topic: str, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        topic_tokens = {tok for tok in topic.lower().split() if len(tok) > 3}
        for item in items:
            source = str(item.get("source", "web")).lower()
            prior = self.priors.get(source, 0.3)

            published = str(item.get("published", ""))[:10]
            freshness_days = _infer_age_days(published)
            decay = self._freshness_decay(freshness_days)

            summary_tokens = {
                tok for tok in str(item.get("summary", "")).lower().split() if len(tok) > 3
            }
            overlap = min(1.0, len(topic_tokens & summary_tokens) / 4.0)
            support = min(1.0, float(item.get("peer_support", 0.0)))

            score = self._posterior(prior, decay, overlap, support)
            enriched = dict(item)
            enriched["score"] = round(score, 3)
            ranked.append(enriched)

        ranked.sort(key=lambda entry: entry.get("score", 0.0), reverse=True)
        return ranked

    def apply_hype_filter(
        self,
        items: Sequence[Dict[str, Any]],
        hype_judge: OptionalJudge | None = None,
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for item in items:
            hype_score = 0.0
            if hype_judge:
                try:
                    hype_score = float(hype_judge(item))
                except Exception:
                    hype_score = 0.0
            if hype_judge and hype_score < self.hype_threshold:
                continue
            filtered.append(item)
        return filtered

    def _posterior(
        self,
        prior: float,
        decay: float,
        overlap: float,
        support: float,
    ) -> float:
        odds = max(prior, 1e-4) / max(1.0 - prior, 1e-4)
        odds *= math.exp(0.8 * decay)
        odds *= math.exp(0.7 * overlap)
        odds *= math.exp(0.8 * support)
        posterior = odds / (1.0 + odds)
        return max(self.noise_floor, min(0.999, posterior))

    def _freshness_decay(self, freshness_days: float) -> float:
        if not math.isfinite(freshness_days) or freshness_days <= 0:
            return 1.0
        half_life = max(1.0, self.freshness_half_life_days)
        return math.exp(-freshness_days / half_life)


def deduplicate_triples(triples: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen: set[str] = set()
    unique: List[Tuple[str, str, str]] = []
    for subj, rel, obj in triples:
        key = hashlib.sha256(f"{subj}::{rel}::{obj}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append((subj, rel, obj))
    return unique


def _infer_age_days(published: str) -> float:
    if not published:
        return 365.0
    try:
        pub_dt = datetime.strptime(published, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return 365.0
    return max(1.0, (datetime.now(UTC) - pub_dt).days)


__all__ = ["BayesianTrendAnalyzer", "deduplicate_triples"]
