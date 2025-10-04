import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.initiative import daemon


def test_retry_succeeds_after_failures():
    calls = []

    def op():
        calls.append("call")
        if len(calls) < 3:
            raise RuntimeError("boom")
        return "ok"

    sleeps = []
    result = daemon._retry(op, attempts=3, base_delay=0.1, backoff=2, sleep_fn=sleeps.append)

    assert result == "ok"
    assert calls == ["call", "call", "call"]
    assert sleeps == [0.1, 0.2]


def test_retry_propagates_final_exception():
    sleeps = []

    def op():
        raise ValueError("still broken")

    with pytest.raises(ValueError):
        daemon._retry(op, attempts=2, base_delay=0.05, backoff=2, sleep_fn=sleeps.append)

    assert sleeps == [0.05]


def test_propose_once_uses_retry(monkeypatch):
    saved_state = {}

    monkeypatch.setattr(daemon, "_load_state", lambda: {"targets": {}})

    def fake_save(state):
        saved_state.update(state)

    monkeypatch.setattr(daemon, "_save_state", fake_save)
    monkeypatch.setattr(daemon, "_now", lambda: 123)

    class DummyOrchestrator:
        def __init__(self, cfg):
            self.cfg = cfg
            self.calls = 0

        def cycle(self, goal: str):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("temporary glitch")
            return {"decision": {"action": "ok"}, "trace": []}

    dummy = DummyOrchestrator({})

    def orchestrator_factory(cfg):
        dummy.cfg = cfg
        return dummy

    monkeypatch.setattr(daemon, "Orchestrator", orchestrator_factory)

    retry_calls = []
    original_retry = daemon._retry

    def fake_retry(operation, **kwargs):
        retry_calls.append(kwargs)
        return original_retry(operation, sleep_fn=lambda *_: None, **kwargs)

    monkeypatch.setattr(daemon, "_retry", fake_retry)

    cfg = {
        "db_path": "./data/symbiont.db",
        "initiative": {
            "goal_template": "Test goal",
            "retry": {"attempts": 3, "base_delay": 0.0, "backoff": 2.0},
        },
    }

    res = daemon.propose_once(cfg, reason="test")

    assert res["decision"]["action"] == "ok"
    assert dummy.calls == 2
    assert retry_calls and retry_calls[0]["attempts"] == 3
    assert saved_state["last_proposal_ts"] == 123


def test_propose_once_publishes_pubsub(tmp_path, monkeypatch):
    log_path = tmp_path / "events.jsonl"
    state = {"targets": {}}

    monkeypatch.setattr(daemon, "_load_state", lambda: state)
    monkeypatch.setattr(daemon, "_save_state", lambda s: state.update(s))
    monkeypatch.setattr(daemon, "_now", lambda: 456)

    class DummyOrchestrator:
        def __init__(self, _cfg):
            pass

        def cycle(self, goal: str):
            return {"decision": {"action": "ok"}, "goal": goal}

    monkeypatch.setattr(daemon, "Orchestrator", lambda cfg: DummyOrchestrator(cfg))
    monkeypatch.setattr(daemon, "_retry", lambda op, **kwargs: op())

    cfg = {
        "db_path": "./data/symbiont.db",
        "initiative": {
            "goal_template": "Goal",
            "pubsub": {
                "enabled": True,
                "backend": "memory",
                "log_path": str(log_path),
            },
        },
    }

    daemon.propose_once(cfg, reason="test")

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[0]["type"] == "initiative.proposal"
