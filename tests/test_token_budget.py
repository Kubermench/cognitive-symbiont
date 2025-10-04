import sys
from pathlib import Path

import pytest

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
