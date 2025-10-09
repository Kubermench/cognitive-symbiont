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


def test_async_hunt_offline_returns_mock(monkeypatch, tmp_path):
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

    payload, artifact = asyncio.run(run_hunt_async(DummyLLM(), "Edge safety", config=HuntConfig(offline=True)))
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


def test_run_hunt_async_persists_and_caches(monkeypatch, tmp_path):
    from symbiont.foresight import hunter

    monkeypatch.setattr(hunter, "HUNT_DIR", tmp_path / "hunts")
    monkeypatch.setattr(hunter, "METRIC_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(hunter, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(hunter.rate_limiter, "limit", lambda source: None)

    class StubRotator:
        def rotate_if_due(self):
            return "token"

    monkeypatch.setattr(hunter, "CredentialRotator", lambda *args, **kwargs: StubRotator())

    calls = {"count": 0}

    async def fake_async(llm, query, config):
        calls["count"] += 1
        return {
            "topic": query,
            "items": [{"title": "Async insight", "summary": "Edge safe", "source": "arxiv"}],
            "meta": {"contributors": ["arxiv"], "avg_score": 0.72},
        }

    monkeypatch.setattr(hunter, "_async_gather_sources", fake_async)

    payload, artifact = asyncio.run(
        run_hunt_async(DummyLLM(), "Hybrid checks", config=HuntConfig(offline=False, cache_ttl_minutes=60))
    )
    assert artifact.exists()
    assert payload["meta"]["contributors"] == ["arxiv"]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics[-1]["topic"] == "Hybrid checks"
    assert calls["count"] == 1

    cached_payload, cached_artifact = asyncio.run(
        run_hunt_async(DummyLLM(), "Hybrid checks", config=HuntConfig(offline=False, cache_ttl_minutes=60))
    )
    assert cached_artifact == artifact
    assert cached_payload["_artifact_path"] == str(artifact)
    assert calls["count"] == 1  # cache hit


def test_analyzer_weight_sources_with_bn(monkeypatch):
    from symbiont.foresight import analyzer as analyzer_mod

    class FakeBN:
        def __init__(self, edges):
            self.edges = edges

    class FakeVE:
        pass

    monkeypatch.setattr(analyzer_mod, "BayesianNetwork", FakeBN)
    monkeypatch.setattr(analyzer_mod, "VariableElimination", FakeVE)

    analyzer = ForesightAnalyzer()
    items = [
        {"title": "Trend", "summary": "agentic foresight rising", "source": "arxiv", "type": "paper"},
        {"title": "Hype", "summary": "viral thread", "source": "x", "type": "thread"},
    ]
    result = analyzer.weight_sources("agentic foresight", items)
    meta = result["meta"]
    assert meta.get("bn_available") is True
    assert isinstance(meta.get("accepted"), list)


def test_analyzer_reflect_and_forecast():
    analyzer = ForesightAnalyzer()
    assert analyzer.reflect_hunt("agentic foresight", 0.8) == "agentic foresight"
    assert "agentic" in analyzer.reflect_hunt("edge safety", 0.3)
    forecast = analyzer.forecast_trend([{"items": 3}, {"items": 6}])
    assert forecast["prediction"] > 0.5


def test_rlhf_tuner_roundtrip(tmp_path):
    from symbiont.tools.rlhf_tuner import RLHFTuner

    tuner = RLHFTuner(state_path=tmp_path / "rlhf.json")
    proposal = {"diff": "# change", "extra": "info"}
    tuner.record_outcome("agentic", 0.8, proposal)
    assert "agentic" in tuner.history
    assert tuner.suggest_query("agentic") == "agentic"
    tuner_reloaded = RLHFTuner(state_path=tmp_path / "rlhf.json")
    assert tuner_reloaded.history["agentic"]["count"] >= 1
    for _ in range(3):
        tuner.record_outcome("agentic", 0.0, proposal)
    assert "resilience" in tuner.suggest_query("agentic")


def test_zk_prover_verify(tmp_path, monkeypatch):
    from symbiont.tools import zk_prover

    monkeypatch.chdir(tmp_path)
    diff_text = "example diff"
    proof = zk_prover.prove_diff(diff_text)
    assert Path(proof["path"]).exists()
    assert zk_prover.verify_diff(proof, diff_text) is True
    assert zk_prover.verify_diff(proof, diff_text + "tamper") is False


def test_credential_rotator_uses_state(monkeypatch, tmp_path):
    from symbiont.foresight import hunter

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(hunter, "STATE_PATH", state_path)
    calls = {"count": 0}

    def fake_load_secret(spec, fallback_env=None):
        calls["count"] += 1
        return "secret-token"

    monkeypatch.setattr(hunter.secrets, "load_secret", fake_load_secret)

    rotator = hunter.CredentialRotator({"method": "env", "env": "FOO"}, hours=24)
    first = rotator.rotate_if_due()
    second = rotator.rotate_if_due()
    assert first == second == "secret-token"
    assert calls["count"] == 1
    assert json.loads(state_path.read_text())["credential"] == "secret-token"


def test_gather_with_backoff_falls_back(monkeypatch):
    from symbiont.foresight import hunter

    monkeypatch.setattr(hunter.rate_limiter, "limit", lambda source: None)

    async def empty_async(*args, **kwargs):
        return {"topic": "q", "items": [], "meta": {}}

    monkeypatch.setattr(hunter, "_async_gather_sources", empty_async)
    monkeypatch.setattr(hunter, "_load_offline_sources", lambda: {"topic": "offline", "meta": {"offline": True}})

    payload = asyncio.run(hunter._gather_with_backoff(DummyLLM(), "query", HuntConfig()))
    assert payload["meta"]["offline"] is True


def test_async_gather_sources_merges(monkeypatch):
    from symbiont.foresight import hunter
    from symbiont.tools import research

    async def fake_trend(llm, query, *, max_items, include_collaborators, include_rss):
        return {
            "topic": query,
            "items": [{"title": "Async", "summary": "signal", "source": "arxiv"}],
            "meta": {"contributors": ["async"]},
        }

    def fake_scout(llm, query, max_items):
        return {
            "topic": query,
            "items": [{"title": "Scout", "summary": "insight", "source": "rss"}],
            "meta": {"contributors": ["scout"], "source_breakdown": {"rss": {"count": 1}}},
        }

    def fake_rss(query, max_items):
        return {"items": [{"title": "RSS", "summary": "feed", "source": "rss"}], "meta": {"contributors": ["rss"]}}

    monkeypatch.setattr(research, "gather_trend_sources_async", fake_trend)
    monkeypatch.setattr(research, "scout_insights", fake_scout)
    monkeypatch.setattr(research, "fetch_rss_alerts", fake_rss)

    cfg = HuntConfig()
    result = asyncio.run(hunter._async_gather_sources(DummyLLM(), "hybrid", cfg))
    assert result["items"]
    assert set(result["meta"]["contributors"]) == {"async", "scout", "rss"}


def test_persist_hunt_appends_metrics(monkeypatch, tmp_path):
    from symbiont.foresight import hunter

    monkeypatch.setattr(hunter, "HUNT_DIR", tmp_path / "hunts")
    monkeypatch.setattr(hunter, "METRIC_PATH", tmp_path / "metrics.json")
    cfg = HuntConfig()
    payload = {
        "topic": "hybrid",
        "items": [{"title": "A", "summary": "B"}],
        "meta": {"contributors": ["arxiv"]},
    }
    first_artifact = hunter._persist_hunt("hybrid", payload, cfg)
    assert first_artifact.exists()
    payload2 = {
        "topic": "hybrid",
        "items": [],
        "meta": {"contributors": []},
    }
    hunter._persist_hunt("hybrid", payload2, cfg)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert len(metrics) == 2
