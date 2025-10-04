import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont import guards


class DummyLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps({"risk": 0.7, "verdict": "high", "reasons": ["danger"]})


def test_analyze_plan_with_judge(monkeypatch):
    monkeypatch.setattr(guards, "LLMClient", lambda cfg: DummyLLM(cfg))
    cfg = {
        "llm": {"provider": "ollama", "model": "phi3:mini"},
        "guard": {"judge": {"enabled": True, "risk_threshold": 0.5}},
    }
    report = guards.analyze_plan("deploy to production", cfg)
    assert report["judge"]["risk"] == 0.7
    assert any("LLM judge" in flag for flag in report["flags"])


def test_analyze_plan_handles_bad_json(monkeypatch):
    class Noisy:
        def __init__(self, cfg):
            pass

        def generate(self, prompt: str) -> str:
            return "not-json"

    monkeypatch.setattr(guards, "LLMClient", lambda cfg: Noisy(cfg))
    cfg = {"guard": {"judge": {"enabled": True, "risk_threshold": 0.5}}}
    report = guards.analyze_plan("safe plan", cfg)
    assert "judge" in report
    assert report["judge"]["risk"] is None
