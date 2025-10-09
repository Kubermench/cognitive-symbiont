import asyncio
import json
from pathlib import Path

import pytest

from symbiont.foresight.hunter import HuntConfig, run_hunt_async
from symbiont.foresight.analyzer import ForesightAnalyzer
from symbiont.foresight.suggester import ForesightSuggester


class DummyLLM:
    """Minimal LLM stub returning deterministic JSON."""

    def __init__(self, proposal_diff: str = "# noop"):
        self.proposal_diff = proposal_diff

    def generate(self, prompt: str, **_: dict) -> str:  # type: ignore[override]
        prompt_lower = prompt.lower()
        if "Return JSON".lower() in prompt_lower and "items" in prompt_lower:
            return json.dumps(
                {
                    "topic": "dummy",
                    "items": [
                        {"title": "Edge case", "url": "https://example.com", "summary": "Agentic systems", "source": "arxiv"},
                    ],
                }
            )
        if "proposal" in prompt_lower:
            return json.dumps({"proposal": "Document findings", "diff": self.proposal_diff})
        if "assess the risk" in prompt_lower:
            return json.dumps({"approve": True, "risk": 0.3, "tests": ["Smoke test"]})
        return "{}"


@pytest.mark.asyncio
async def test_async_hunt_offline_returns_mock(monkeypatch, tmp_path):
    from symbiont.foresight import hunter

    mock_path = tmp_path / "offline.json"
    mock_payload = {
        "topic": "offline sample",
        "items": [{"title": "Cached insight", "url": "https://offline", "summary": "Fallback", "source": "cache"}],
        "meta": {"offline": True},
    }
    mock_path.write_text(json.dumps(mock_payload), encoding="utf-8")

    monkeypatch.setattr(hunter, "OFFLINE_MOCK", mock_path)
    monkeypatch.setattr(hunter, "HUNT_DIR", tmp_path / "hunts")
    monkeypatch.setattr(hunter, "METRIC_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(hunter, "STATE_PATH", tmp_path / "state.json")

    payload, artifact = await run_hunt_async(DummyLLM(), "Edge safety", config=HuntConfig(offline=True))
    assert payload["meta"]["offline"] is True
    assert artifact.exists()
    metrics = json.loads(Path(artifact).parent.parent.joinpath("metrics.json").read_text())
    assert metrics[-1]["topic"] == "Edge safety"


def test_analyzer_versions_triples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    analyzer = ForesightAnalyzer()
    items = [
        {"title": "Async hunts", "summary": "agentic foresight", "source": "arxiv"},
        {"title": "Edge viz", "summary": "pi5 safe", "source": "rss"},
    ]
    info = analyzer.version_triples("Agentic foresight", items)
    assert Path(info["diff_path"]).exists()
    follow_up = analyzer.version_triples("Agentic foresight", items[:1])
    diff = json.loads(Path(follow_up["diff_path"]).read_text())
    assert diff["diff"]["removed"], "Expected removed triples when items shrink"


def test_suggester_flags_causal_risk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    suggester = ForesightSuggester(DummyLLM(proposal_diff="remove guards delete rogue"))
    context = {"triples": [("topic", "mentions", "agentic foresight")], "relevance": 0.2, "query": "agentic"}
    proposal = suggester.draft({"topic": "agentic", "highlight": "risk"}, context=context)
    assert proposal["causal_risk"] >= suggester.causal_threshold
    assert proposal.get("flagged") is True
    validation = suggester.validate(proposal)
    assert validation["approve"] is False
