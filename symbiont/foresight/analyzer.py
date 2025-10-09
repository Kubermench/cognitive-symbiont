"""Source weighting, versioned triples, and foresight reflections."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from symbiont.memory.db import MemoryDB
from symbiont.memory.dynamic_analyzer import BayesianTrendAnalyzer, deduplicate_triples
from symbiont.tools.files import ensure_dirs

LOGGER = logging.getLogger(__name__)

DIFF_DIR = Path("data/artifacts/foresight/rag_diffs")
SNAPSHOT_DIR = Path("data/foresight/rag_snapshots")
PLOT_DIR = Path("data/artifacts/foresight/plots")

try:  # Optional heavy dependencies (pgmpy, pandas, matplotlib) stay guarded.
    from pgmpy.models import BayesianNetwork  # type: ignore
    from pgmpy.inference import VariableElimination  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BayesianNetwork = None  # type: ignore
    VariableElimination = None  # type: ignore

try:  # Optional dataset viz
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _pd = None  # type: ignore

try:  # Matplotlib may be unavailable on edge hardware; guard usage.
    import matplotlib.pyplot as _plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _plt = None  # type: ignore


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower())[:60] or "foresight"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _format_triple(triple: Sequence[str]) -> Dict[str, str]:
    subj, rel, obj = (str(part) for part in triple[:3])
    return {"subject": subj, "relation": rel, "object": obj}


def _diff_triples(old: Sequence[Sequence[str]], new: Sequence[Sequence[str]]) -> Dict[str, Any]:
    old_set = {tuple(group[:3]) for group in old}
    new_set = {tuple(group[:3]) for group in new}
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    unchanged = sorted(new_set & old_set)
    return {
        "added": [_format_triple(triple) for triple in added],
        "removed": [_format_triple(triple) for triple in removed],
        "unchanged": [_format_triple(triple) for triple in unchanged],
    }


def _default_triples(topic: str, items: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for item in items:
        title = str(item.get("title") or item.get("summary") or "untitled")
        source = str(item.get("source", "source"))
        triples.append((topic, "mentions", title))
        triples.append((title, "sourced_from", source))
    return triples


@dataclass
class ForesightAnalyzer:
    """Wraps ranking, versioning, and reflection utilities."""

    db: MemoryDB | None = None
    analyzer: BayesianTrendAnalyzer = field(default_factory=BayesianTrendAnalyzer)
    min_relevance: float = 0.6

    def weight_sources(self, topic: str, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ranked = self.analyzer.rank_sources(topic, items)
        filtered = self.analyzer.apply_hype_filter(
            ranked,
            hype_judge=lambda item: float(item.get("hype_score", 0.7)),
        )
        pruned = [item for item in filtered if float(item.get("score", 0.0)) >= self.min_relevance]
        if not pruned:
            pruned = ranked[:2]

        bayes_meta = {
            "avg_score": round(mean([float(entry.get("score", 0.0)) for entry in pruned]), 3)
            if pruned
            else 0.0,
            "total": len(pruned),
        }
        bn_meta = self._apply_bayesian_network(pruned)
        bayes_meta.update(bn_meta)
        return {"items": pruned, "meta": bayes_meta}

    def _apply_bayesian_network(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {}
        if not BayesianNetwork or not VariableElimination:  # pragma: no cover - optional
            return {"bn_available": False}
        try:
            model = BayesianNetwork([("source", "relevance"), ("type", "relevance")])
            # Fake CPTs capturing simple priors; enough for pruning.
            cpt_source = {
                "arxiv": 0.82,
                "rss": 0.55,
                "peer": 0.65,
                "web": 0.45,
                "x": 0.35,
            }
            cpt_type = {"paper": 0.78, "blog": 0.48, "dataset": 0.6, "thread": 0.4}
            # pgmpy expects CPDs; to avoid heavy CPD construction, infer manually.
            accepted: List[Dict[str, Any]] = []
            for item in items:
                source = str(item.get("source", "web")).lower()
                doc_type = str(item.get("type", "paper")).lower()
                prob = 0.5 * cpt_source.get(source, 0.4) + 0.5 * cpt_type.get(doc_type, 0.4)
                if prob >= self.min_relevance:
                    enriched = dict(item)
                    enriched["bn_score"] = round(prob, 3)
                    accepted.append(enriched)
            return {"bn_available": True, "accepted": accepted}
        except Exception as exc:  # pragma: no cover - optional dependency quirks
            LOGGER.debug("Bayesian network weighting failed: %s", exc)
            return {"bn_available": False}

    def version_triples(self, topic: str, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        triples = deduplicate_triples(_default_triples(topic, items))
        ensure_dirs([DIFF_DIR, SNAPSHOT_DIR])
        snapshot_path = SNAPSHOT_DIR / f"{_slug(topic)}.json"
        old_triples: List[List[str]] = []
        if snapshot_path.exists():
            with suppress(Exception):
                data = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    old_triples = [list(entry[:3]) for entry in data]
        diff = _diff_triples(old_triples, triples)
        diff_payload = {
            "topic": topic,
            "timestamp": _now_iso(),
            "diff": diff,
        }
        diff_path = DIFF_DIR / f"{_slug(topic)}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.json"
        diff_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")
        snapshot_path.write_text(json.dumps([list(triple) for triple in triples], indent=2), encoding="utf-8")
        return {"triples": triples, "diff_path": str(diff_path)}

    def upsert_triples(self, topic: str, triples: Sequence[Tuple[str, str, str]]) -> None:
        if not self.db:
            return
        for subj, rel, obj in triples:
            try:
                from symbiont.memory import graphrag

                graphrag.add_claim(self.db, subj, rel, obj, importance=0.6)
            except Exception as exc:  # pragma: no cover - database specific
                LOGGER.debug("Failed adding claim %s : %s", (subj, rel, obj), exc)

    def reflect_hunt(self, topic: str, relevance: float) -> str:
        if relevance >= 0.7:
            return topic
        tweak = "agentic"
        if "agentic" in topic.lower():
            tweak = "self-evolving"
        return f"{topic} {tweak}".strip()

    def forecast_trend(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"prediction": 0.5, "explanation": "No history"}
        weights = []
        horizon = min(len(records), 6)
        for idx, record in enumerate(records[-horizon:], start=1):
            weight = idx / horizon
            weights.append(weight * float(record.get("items", 0)))
        score = min(0.99, sum(weights) / (horizon * max(1.0, mean(weights))))
        return {
            "prediction": round(score, 3),
            "explanation": f"Weighted recent hunts ({horizon}) → {score:.2f}",
        }

    def dataset_viz(self, dataset_path: str, topic: str) -> Optional[str]:
        if not dataset_path or not _pd or not _plt:  # pragma: no cover - optional
            return None
        path = Path(dataset_path)
        if not path.exists():
            return None
        try:
            df = _pd.read_csv(path) if path.suffix == ".csv" else _pd.read_json(path)
        except Exception as exc:  # pragma: no cover - file dependent
            LOGGER.debug("Failed loading dataset %s: %s", path, exc)
            return None
        summary = df.describe(include="all").transpose().reset_index()
        ensure_dirs([PLOT_DIR])
        out = PLOT_DIR / f"{_slug(topic)}_dataset.png"
        try:
            fig = _plt.figure(figsize=(6, 3))
            _plt.axis("off")
            table = _plt.table(
                cellText=summary.values[:10],
                colLabels=summary.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            fig.tight_layout()
            fig.savefig(out, dpi=140)
        finally:
            _plt.close(fig)  # type: ignore[attr-defined]
        return str(out)
