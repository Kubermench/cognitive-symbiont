import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.agents.swarm import SwarmCoordinator
from symbiont.ports.ai_peer import PeerTranscript


def test_swarm_score_variants_assigns_agent_ids(tmp_path, monkeypatch):
    cfg = {
        "db_path": str(tmp_path / "sym.db"),
        "initiative": {"repo_path": str(tmp_path)},
        "evolution": {"swarm_enabled": True},
        "ports": {"ai_peer": {"stub_mode": True}},
    }
    coordinator = SwarmCoordinator(cfg)

    transcript_dir = tmp_path / "data" / "artifacts" / "ai_peer"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    captured = []

    def fake_chat(
        prompt: str,
        *,
        simulate_only: bool = False,
        agent_id: str | None = None,
        budget=None,
    ):
        captured.append(agent_id)
        payload = {"score": 0.9, "justification": "ok"}
        path = transcript_dir / f"peer_{agent_id}.json"
        path.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "response": json.dumps(payload),
                    "simulated": True,
                    "timestamp": 0,
                    "agent_id": agent_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return PeerTranscript(
            prompt=prompt,
            response=json.dumps(payload),
            simulated=True,
            path=str(path),
            agent_id=agent_id,
        )

    monkeypatch.setattr(coordinator.peer, "chat", fake_chat)

    variants = coordinator._score_variants([
        {"subject": "a", "relation": "b", "object": "c"},
    ])

    assert len(variants) == 1
    assert variants[0].agent_id
    assert captured == [variants[0].agent_id]
