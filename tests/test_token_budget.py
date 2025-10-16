import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.llm.budget import TokenBudget
from symbiont.llm.client import LLMClient


def test_token_budget_records_attempt(monkeypatch):
    cfg = {"provider": "ollama", "model": "phi3:mini", "mode": "local"}
    client = LLMClient(cfg)

    def fake_dispatch(*args, **kwargs):
        return "synthetic response"

    monkeypatch.setattr(LLMClient, "_dispatch", fake_dispatch, raising=True)

    budget = TokenBudget(limit=200, label="unit")
    result = client.generate("Plan the sprint milestones", budget=budget, label="test")

    assert result == "synthetic response"
    assert budget.used > 0
    assert budget.events
    assert budget.events[-1]["outcome"] == "ok"


def test_token_budget_denies_when_limit_reached(monkeypatch):
    cfg = {"provider": "ollama", "model": "phi3:mini", "mode": "local"}
    client = LLMClient(cfg)
    called = False

    def fake_dispatch(*args, **kwargs):
        nonlocal called
        called = True
        return "should not run"

    monkeypatch.setattr(LLMClient, "_dispatch", fake_dispatch, raising=True)

    budget = TokenBudget(limit=5, label="tight")
    output = client.generate("x" * 200, budget=budget, label="denied")

    assert output == ""
    assert called is False
    assert budget.events
    assert budget.events[-1]["outcome"] == "denied"
    assert budget.used == 0


def test_token_budget_history(tmp_path):
    sink = tmp_path / "budget.json"
    history = tmp_path / "history.jsonl"
    budget = TokenBudget(limit=100, label="hist", sink_path=sink, history_path=history)
    budget.log_attempt(
        prompt_tokens=10,
        response_tokens=5,
        provider="ollama",
        model="phi3:mini",
        label="hist",
        source="test",
        outcome="ok",
        latency=0.05,
    )

    assert sink.exists()
    assert history.exists()
    lines = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines and lines[0]["label"] == "hist"


def test_token_budget_trims_event_buffer():
    budget = TokenBudget(limit=0, label="trim")
    for idx in range(250):
        budget.log_attempt(
            prompt_tokens=2,
            response_tokens=1,
            provider="ollama",
            model="phi3:mini",
            label="trim",
            source=f"unit-{idx}",
            outcome="ok",
            latency=0.01,
        )

    assert len(budget.events) == 200
    # Oldest events fall off the buffer
    assert budget.events[0]["source"] == "unit-50"
    assert budget.used == 750


def test_token_budget_snapshot_throttle(tmp_path, monkeypatch):
    sink = tmp_path / "snap.json"
    budget = TokenBudget(limit=0, label="snap", sink_path=sink)

    write_calls = {"count": 0}
    original_write = Path.write_text

    def fake_write(self, *args, **kwargs):
        if self == sink:
            write_calls["count"] += 1
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fake_write, raising=False)
    monkeypatch.setattr("symbiont.llm.budget.time.monotonic", lambda: 0.0)

    for idx in range(5):
        budget.log_attempt(
            prompt_tokens=1,
            response_tokens=1,
            provider="ollama",
            model="phi3:mini",
            label="snap",
            source=f"throttle-{idx}",
            outcome="ok",
            latency=0.01,
        )

    assert write_calls["count"] == 1
