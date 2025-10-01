from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from ..llm.client import LLMClient
from ..memory.db import MemoryDB
from ..memory import graphrag
from ..ports.ai_peer import AIPeerBridge
from ..tools.files import ensure_dirs
from ..guards import analyze_plan


@dataclass
class SwarmVariant:
    triple: Dict[str, str]
    score: float
    justification: str


class SwarmCoordinator:
    """Coordinates swarm evolution of belief triples using simulated peers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.evo_cfg = (self.config.get("evolution") or {})
        self.enabled = bool(self.evo_cfg.get("swarm_enabled", False))
        self.variants = int(self.evo_cfg.get("swarm_variants", 3))
        self.timeout_seconds = int(self.evo_cfg.get("swarm_timeout", 20))
        self.max_delta_ratio = float(self.evo_cfg.get("swarm_max_delta", 0.05))
        self.repo_root = Path(self.config.get("initiative", {}).get("repo_path", ".")).resolve()

        self.llm = LLMClient(self.config.get("llm", {}))
        self.peer = AIPeerBridge(self.config)
        self.db = MemoryDB(db_path=self.config.get("db_path", "./data/symbiont.db"))
        ensure_dirs([self.repo_root / "data" / "artifacts" / "swarm"])

    # ------------------------------------------------------------------
    def after_cycle(self, cycle_result: Dict[str, Any]) -> Optional[List[SwarmVariant]]:
        if not self.enabled:
            return None
        last_action = (cycle_result.get("decision") or {}).get("action", "")
        if not last_action:
            return None
        belief_hint = f"Action: {last_action}"
        return self.run(belief_hint, variants=self.variants, auto=True)

    # ------------------------------------------------------------------
    def run(self, belief_text: str, *, variants: int | None = None, auto: bool = False) -> List[SwarmVariant]:
        variants = variants or self.variants
        self.db.ensure_schema()
        seed = self._ensure_seed_triple(belief_text, auto)
        if not seed:
            return []

        forks = self._fork_variants(seed, variants)
        scored = self._score_variants(forks)
        winners = self._merge_variants(scored)
        if winners:
            self._apply_winners(winners, seed)
        return winners

    # ------------------------------------------------------------------
    def _ensure_seed_triple(self, belief_text: str, auto: bool) -> Optional[Dict[str, str]]:
        belief_text = (belief_text or "").strip()
        if belief_text.startswith("belief:"):
            belief_text = belief_text[len("belief:"):].strip()

        if belief_text.count("->") == 2:
            subject, relation, obj = [segment.strip() for segment in belief_text.split("->", 2)]
            return {"subject": subject, "relation": relation, "object": obj}

        if auto:
            with self.db._conn() as conn:
                row = conn.execute(
                    "SELECT e.name, r.name, c.object FROM claims c "
                    "JOIN entities e ON c.subject_id=e.id JOIN relations r ON c.relation_id=r.id "
                    "ORDER BY c.updated_at DESC LIMIT 1"
                ).fetchone()
            if row:
                return {"subject": row[0], "relation": row[1], "object": row[2]}

        prompt = (
            "Interpret the text into a belief triple as JSON {subject, relation, object}.\n" + belief_text
        )
        raw = self.llm.generate(prompt) or "{}"
        try:
            data = json.loads(raw)
            if {"subject", "relation", "object"} <= data.keys():
                return {
                    "subject": str(data["subject"]).strip(),
                    "relation": str(data["relation"]).strip(),
                    "object": str(data["object"]).strip(),
                }
        except Exception:
            pass
        return None

    def _fork_variants(self, seed: Dict[str, str], n: int) -> List[Dict[str, str]]:
        base = json.dumps(seed)
        prompt = (
            "You are coordinating a swarm of planner agents. Starting belief: "
            f"{base}. Produce {n} variant triples as JSON list.\n"
            "Each triple must be concise and developer-focused."
        )
        raw = self.llm.generate(prompt) or "[]"
        try:
            data = json.loads(raw)
            variants = [
                {
                    "subject": str(item.get("subject", seed["subject"])).strip(),
                    "relation": str(item.get("relation", seed["relation"])).strip(),
                    "object": str(item.get("object", seed["object"])).strip(),
                }
                for item in data[:n]
            ]
            return variants or [seed]
        except Exception:
            return [seed]

    def _score_variants(self, variants: Iterable[Dict[str, str]]) -> List[SwarmVariant]:
        scored: List[SwarmVariant] = []
        for variant in variants:
            message = (
                "Rate this dev belief triple from 0-1 (float). Respond as JSON {score, justification}.\n"
                f"Triple: {variant}"
            )
            transcript = self.peer.chat(message, simulate_only=False)
            try:
                payload = json.loads(transcript.response)
                score = float(payload.get("score", 0.0))
                justification = str(payload.get("justification", transcript.response))
            except Exception:
                score = 0.0
                justification = transcript.response.strip()[:200]
            scored.append(SwarmVariant(triple=variant, score=max(0.0, min(score, 1.0)), justification=justification))
        return scored

    def _merge_variants(self, scored: List[SwarmVariant]) -> List[SwarmVariant]:
        if not scored:
            return []

        graph = nx.Graph()
        for i, variant in enumerate(scored):
            node_id = f"v{i}"
            graph.add_node(node_id, triple=variant.triple, score=variant.score)

        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                vi, vj = scored[i], scored[j]
                weight = self._variant_similarity(vi.triple, vj.triple)
                if weight > 0:
                    graph.add_edge(f"v{i}", f"v{j}", weight=weight)

        components = list(nx.connected_components(graph))
        winners: List[SwarmVariant] = []
        for component in components:
            subgraph = graph.subgraph(component)
            sorted_nodes = sorted(
                subgraph.nodes(data=True), key=lambda node: node[1]["score"], reverse=True
            )
            if not sorted_nodes:
                continue
            top_node, attrs = sorted_nodes[0]
            if attrs["score"] < 0.6:
                continue
            triple = attrs["triple"]
            justification = next(
                (variant.justification for variant in scored if variant.triple == triple),
                "",
            )
            winners.append(SwarmVariant(triple=triple, score=attrs["score"], justification=justification))

        return winners

    def _variant_similarity(self, a: Dict[str, str], b: Dict[str, str]) -> float:
        score = 0.0
        for key in ("subject", "relation", "object"):
            if a.get(key) == b.get(key):
                score += 0.33
        return min(score, 1.0)

    def _apply_winners(self, winners: List[SwarmVariant], seed: Dict[str, str]) -> None:
        with self.db._conn() as conn:
            total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] or 1
        max_updates = max(1, int(total_claims * self.max_delta_ratio))

        applied = 0
        for variant in winners:
            if applied >= max_updates:
                break
            triple = variant.triple
            annotation = f"swarm merge {time.strftime('%Y-%m-%d %H:%M:%S')}"
            graphrag.add_claim(
                self.db,
                triple["subject"],
                triple["relation"],
                triple["object"],
                importance=min(0.9, max(0.4, variant.score)),
                source_url=annotation,
            )
            applied += 1

        self._write_artifact(seed, winners)

    def _write_artifact(self, seed: Dict[str, str], winners: List[SwarmVariant]) -> None:
        artifacts_dir = self.repo_root / "data" / "artifacts" / "swarm"
        ensure_dirs([artifacts_dir])
        path = artifacts_dir / f"swarm_{int(time.time())}.json"
        payload = {
            "seed": seed,
            "winners": [
                {"triple": w.triple, "score": w.score, "justification": w.justification}
                for w in winners
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # basic governance check on descriptions
        plan_text = json.dumps(payload)
        report = analyze_plan(plan_text)
        if report.get("flags"):
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n\n# guard_flags\n"
                + json.dumps(report["flags"], indent=2),
                encoding="utf-8",
            )
