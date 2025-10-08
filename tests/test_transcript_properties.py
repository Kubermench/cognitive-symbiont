import json
import sys
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.agents.swarm import SwarmCoordinator


def _swarm_cfg(tmp_path: Path) -> dict:
    return {
        "db_path": str(tmp_path / "sym.db"),
        "initiative": {
            "repo_path": str(tmp_path),
            "state": {"backend": "sqlite", "path": str(tmp_path / "state.db")},
        },
        "evolution": {"swarm_enabled": True},
        "ports": {"ai_peer": {"stub_mode": True}},
    }


def _valid_transcript(prompt: str, score: float, justification: str, agent_id: str, simulated: bool) -> dict:
    triple_prompt = (
        "You are agent. Triple: {"
        f"'subject': '{prompt[:10]}', "
        "'relation': 'rel', "
        "'object': 'obj'"
        "}"
    )
    return {
        "prompt": triple_prompt,
        "response": json.dumps({"score": score, "justification": justification}),
        "simulated": simulated,
        "timestamp": 0,
        "agent_id": agent_id,
    }


valid_transcript_strategy = st.builds(
    _valid_transcript,
    prompt=st.text(min_size=3, max_size=30),
    score=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    justification=st.text(max_size=120),
    agent_id=st.text(min_size=1, max_size=12),
    simulated=st.booleans(),
)


invalid_transcript_strategy = st.one_of(
    st.text(min_size=1, max_size=200),
    st.binary(min_size=1, max_size=128).map(lambda b: b.decode("utf-8", errors="ignore")),
)


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.one_of(valid_transcript_strategy, invalid_transcript_strategy), min_size=1, max_size=5))
def test_merge_from_transcripts_handles_fuzz(tmp_path, transcripts):
    cfg = _swarm_cfg(tmp_path)
    coordinator = SwarmCoordinator(cfg)
    transcripts_dir = Path(cfg["initiative"]["repo_path"]) / "data" / "artifacts" / "ai_peer"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for idx, payload in enumerate(transcripts):
        path = transcripts_dir / f"peer_{idx}.json"
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(payload, encoding="utf-8")

    winners = coordinator.merge_from_transcripts()

    processed_dir = transcripts_dir / "processed"
    assert not any(p.is_file() for p in transcripts_dir.glob("peer_*.json"))
    assert processed_dir.exists()
    assert all(p.is_file() for p in processed_dir.glob("peer_*.json"))

    for winner in winners:
        assert isinstance(winner.score, float)
