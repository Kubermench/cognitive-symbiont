"""Proposal generation helpers with causal and RLHF safeguards."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

from symbiont.llm.client import LLMClient
from symbiont.tools import research
from symbiont.tools.rlhf_tuner import RLHFTuner
from symbiont.tools.zk_prover import prove_diff

LOGGER = logging.getLogger(__name__)

try:  # Optional heavy dependency; guard import.
    from dowhy import gcm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gcm = None  # type: ignore


def _basic_causal_score(proposal: Dict[str, Any], triples: Sequence[Sequence[str]]) -> float:
    if gcm is None or not triples:  # pragma: no cover - optional
        penalty = 0.0
        diff = str(proposal.get("diff", "")).lower()
        if "rogue" in diff or "danger" in diff:
            penalty += 0.2
        if "delete" in diff:
            penalty += 0.15
        return min(1.0, penalty)
    try:  # pragma: no cover - depends on optional dependency
        model = gcm.InvertibleStructuralCausalModel()
        model.add_node("mod")
        model.add_node("rogue")
        model.add_edge("mod", "rogue")
        # Synthetic observational data
        samples = [{"mod": "apply", "rogue": 0.3}, {"mod": "reject", "rogue": 0.15}]
        model.fit(samples)
        apply = model.predict_interventions({"mod": "apply"})
        reject = model.predict_interventions({"mod": "reject"})
        return max(0.0, float(apply["rogue"] - reject["rogue"]))
    except Exception as exc:  # pragma: no cover - dependency quirks
        LOGGER.debug("DoWhy causal evaluation failed: %s", exc)
        return 0.3


@dataclass
class ForesightSuggester:
    """Draft foresight proposals while enforcing causal & RLHF checks."""

    llm: LLMClient
    tuner: RLHFTuner = field(default_factory=RLHFTuner)
    causal_threshold: float = 0.2

    def draft(self, insight: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        context = context or {}
        proposal = research.draft_proposal(self.llm, insight)
        topic = str(insight.get("topic") or context.get("query") or "foresight")
        if not str(proposal.get("proposal", "")).strip():
            fallback = research.build_fallback_proposal(topic, insight)
            proposal["proposal"] = fallback["proposal"]
            proposal.setdefault("diff", fallback["diff"])
        elif not str(proposal.get("diff", "")).strip():
            fallback = research.build_fallback_proposal(topic, insight)
            proposal["diff"] = fallback["diff"]
        triples: Sequence[Sequence[str]] = context.get("triples") or []
        risk = _basic_causal_score(proposal, triples)
        proposal["causal_risk"] = round(risk, 3)
        if risk >= self.causal_threshold:
            proposal["flagged"] = True
            proposal.setdefault("notes", []).append("Causal risk exceeds threshold; require manual review.")

        diff = str(proposal.get("diff", ""))
        proposal["zk_proof"] = prove_diff(diff)

        reward_signal = float(context.get("relevance", 0.0))
        query = str(context.get("query") or insight.get("topic") or "foresight")
        self.tuner.record_outcome(query, reward_signal, proposal)
        proposal["query_suggestion"] = self.tuner.suggest_query(query)
        return proposal

    def validate(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        validation = research.validate_proposal(self.llm, proposal)
        risk = float(proposal.get("causal_risk", 0.0))
        if risk >= self.causal_threshold:
            validation["approve"] = False
            validation.setdefault("tests", []).append("Escalate to governance council (causal risk high).")
        return validation
