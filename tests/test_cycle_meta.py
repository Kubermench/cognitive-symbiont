from __future__ import annotations

from typing import Dict, Any

import pytest

from symbiont.agents import reflector as reflector_module
from symbiont.agents.reflector import CycleReflector


@pytest.fixture(autouse=True)
def isolated_reflector_state(tmp_path, monkeypatch):
    tmp_state = tmp_path / "state.json"
    monkeypatch.setattr(reflector_module, "STATE_FILE", tmp_state)
    yield


def _run_cycle(reflector: CycleReflector, *, action: str, bullets: list[str], reward: float, episode: int) -> None:
    trace = [{"role": "architect", "output": {"bullets": bullets}}]
    result: Dict[str, Any] = {
        "episode_id": episode,
        "decision": {"action": action},
        "trace": trace,
        "reward": reward,
    }
    reflector.observe_cycle(result)


def test_meta_learner_adjusts_thresholds_on_repeated_low_reward_cycles() -> None:
    config = {"evolution": {"repeat_threshold": 3, "empty_bullet_threshold": 2}}
    reflector = CycleReflector(config)

    for idx in range(4):
        _run_cycle(
            reflector,
            action="Refine plan",
            bullets=[],
            reward=0.3,
            episode=idx + 1,
        )

    adjustments = reflector._state.get("meta_adjustments", {})
    assert adjustments.get("repeat_threshold") == 2
    assert adjustments.get("empty_bullet_threshold") == 1


def test_meta_learner_relaxes_thresholds_after_high_reward_diverse_cycles() -> None:
    config = {"evolution": {"repeat_threshold": 3, "empty_bullet_threshold": 2}}
    reflector = CycleReflector(config)

    for idx in range(4):
        _run_cycle(
            reflector,
            action="Refine plan",
            bullets=[],
            reward=0.3,
            episode=idx + 1,
        )

    for idx in range(12):
        _run_cycle(
            reflector,
            action=f"Action {idx}",
            bullets=[f"step {idx}"],
            reward=0.85,
            episode=idx + 10,
        )

    adjustments = reflector._state.get("meta_adjustments", {})
    assert "repeat_threshold" not in adjustments
    assert "empty_bullet_threshold" not in adjustments
