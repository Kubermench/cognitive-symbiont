import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.agents.swarm import SwarmCoordinator, SwarmVariant


@pytest.fixture
def swarm_cfg(tmp_path):
    return {
        "db_path": str(tmp_path / "sym.db"),
        "initiative": {"repo_path": str(tmp_path)},
        "evolution": {"swarm_enabled": True},
        "ports": {"ai_peer": {"stub_mode": True}},
    }


def test_swarm_run_merges_best_variants(monkeypatch, swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)

    seed = {"subject": "repo", "relation": "needs", "object": "linting"}
    monkeypatch.setattr(coordinator, "_ensure_seed_triple", lambda text, auto: seed)
    v1 = seed
    v2 = {"subject": "repo", "relation": "needs", "object": "type hints"}
    v3 = {"subject": "docs", "relation": "require", "object": "refresh"}
    v4 = {"subject": "ci", "relation": "adds", "object": "coverage"}

    monkeypatch.setattr(coordinator, "_fork_variants", lambda seed, n: [v1, v2, v3, v4])

    scored = [
        SwarmVariant(triple=v1, score=0.92, justification="top", agent_id="a1"),
        SwarmVariant(triple=v2, score=0.74, justification="similar", agent_id="a2"),
        SwarmVariant(triple=v3, score=0.58, justification="below threshold", agent_id="a3"),
        SwarmVariant(triple=v4, score=0.81, justification="second winner", agent_id="a4"),
    ]
    monkeypatch.setattr(coordinator, "_score_variants", lambda variants: scored)

    applied = []

    def capture_apply(winners, _seed):
        applied.extend(winners)

    monkeypatch.setattr(coordinator, "_apply_winners", capture_apply)

    winners = coordinator.run("belief: repo -> needs -> linting", apply=True)

    assert winners == applied
    assert len(winners) == 2
    assert winners[0].triple == v1 and winners[0].agent_id == "a1"
    assert winners[1].triple == v4 and winners[1].agent_id == "a4"


def test_swarm_run_handles_missing_seed(monkeypatch, swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    monkeypatch.setattr(coordinator, "_ensure_seed_triple", lambda text, auto: None)
    winners = coordinator.run("belief: missing -> triple -> data", apply=False)
    assert winners == []


def test_merge_from_transcripts_promotes_winners(monkeypatch, swarm_cfg, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg = dict(swarm_cfg)
    cfg["initiative"] = {"repo_path": str(repo_root)}
    coordinator = SwarmCoordinator(cfg)
    coordinator.db.ensure_schema()

    transcripts_dir = Path(cfg["initiative"]["repo_path"]) / "data" / "artifacts" / "ai_peer"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "prompt": "Judge triple. Triple: {'subject': 'repo', 'relation': 'adds', 'object': 'tests'}",
        "response": json.dumps({"score": 0.82, "justification": "ship"}),
        "simulated": True,
        "timestamp": 123,
        "agent_id": "agent-a",
    }
    (transcripts_dir / "peer_a.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload_b = {
        "prompt": "Judge triple. Triple: {'subject': 'docs', 'relation': 'needs', 'object': 'refresh'}",
        "response": json.dumps({"score": 0.91, "justification": "docs"}),
        "simulated": True,
        "timestamp": 456,
        "agent_id": "agent-b",
    }
    (transcripts_dir / "peer_b.json").write_text(json.dumps(payload_b, indent=2), encoding="utf-8")

    monkeypatch.setattr("symbiont.agents.swarm.graphrag.add_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "symbiont.agents.swarm.analyze_plan", lambda _: {"flags": {"risk": "high"}}
    )

    winners = coordinator.merge_from_transcripts()

    assert winners
    assert {w.agent_id for w in winners} == {"agent-a", "agent-b"}

    processed = transcripts_dir / "processed"
    assert sorted(p.name for p in processed.iterdir()) == ["peer_a.json", "peer_b.json"]


def test_after_cycle_respects_disabled_flag(swarm_cfg):
    cfg = dict(swarm_cfg)
    cfg["evolution"]["swarm_enabled"] = False
    coordinator = SwarmCoordinator(cfg)
    assert coordinator.after_cycle({}) is None


def test_after_cycle_invokes_run(monkeypatch, swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    winners = [
        SwarmVariant(
            triple={"subject": "repo", "relation": "adds", "object": "tests"},
            score=0.8,
            justification="ok",
            agent_id="after-1",
        )
    ]
    called = {}
    monkeypatch.setattr(coordinator, "run", lambda *args, **kwargs: called.setdefault("run", winners))
    monkeypatch.setattr(coordinator, "merge_from_transcripts", lambda: called.setdefault("merge", True))

    result = coordinator.after_cycle({"decision": {"action": "Tighten lint"}})
    assert result == winners
    assert called["run"]
    assert called["merge"] is True


def test_after_cycle_ignores_empty_action(swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    assert coordinator.after_cycle({"decision": {"action": ""}}) is None


def test_ensure_seed_triple_parses_variants(swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    triple = coordinator._ensure_seed_triple("belief: repo -> needs -> linting", auto=False)
    assert triple == {"subject": "repo", "relation": "needs", "object": "linting"}

    # JSON fallback
    prompt = '{"subject": "docs", "relation": "updates", "object": "tutorial"}'
    coordinator.llm.generate = lambda _: prompt
    triple = coordinator._ensure_seed_triple("Describe", auto=False)
    assert triple == {"subject": "docs", "relation": "updates", "object": "tutorial"}


def test_ensure_seed_triple_auto_fetches_latest(swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    coordinator.db.ensure_schema()
    with coordinator.db._conn() as conn:
        conn.execute("INSERT INTO entities (name) VALUES (?)", ("repo",))
        conn.execute("INSERT INTO relations (name) VALUES (?)", ("needs",))
        subj_id = conn.execute("SELECT id FROM entities WHERE name='repo'").fetchone()[0]
        rel_id = conn.execute("SELECT id FROM relations WHERE name='needs'").fetchone()[0]
        conn.execute(
            "INSERT INTO claims (subject_id, relation_id, object, importance, source_url) VALUES (?,?,?,?,?)",
            (subj_id, rel_id, "linting", 0.5, "seed"),
        )

    triple = coordinator._ensure_seed_triple("", auto=True)
    assert triple == {"subject": "repo", "relation": "needs", "object": "linting"}


def test_fork_variants_handles_invalid_json(monkeypatch, swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    coordinator.llm.generate = lambda _: "not-json"
    seed = {"subject": "repo", "relation": "needs", "object": "lint"}
    variants = coordinator._fork_variants(seed, 3)
    assert variants == [seed]


def test_fork_variants_parses_json(monkeypatch, swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    payload = json.dumps([
        {"subject": "repo", "relation": "needs", "object": "lint"},
        {"subject": "docs", "relation": "needs", "object": "refresh"},
    ])
    coordinator.llm.generate = lambda _: payload
    seed = {"subject": "repo", "relation": "needs", "object": "lint"}
    result = coordinator._fork_variants(seed, 2)
    assert result[1]["subject"] == "docs"


def test_score_variants_handles_bad_payload(monkeypatch, swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)

    def fake_chat(prompt, simulate_only=False, agent_id=None):
        class Dummy:
            response = "not-json"
            path = str(Path(coordinator.repo_root) / "data" / "artifacts" / "ai_peer" / "peer_test.json")
            simulated = True

            def __init__(self):
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.path).write_text(json.dumps({"prompt": prompt, "response": "not-json", "simulated": True, "timestamp": 0}), encoding="utf-8")

        return Dummy()

    monkeypatch.setattr(coordinator.peer, "chat", fake_chat)
    variants = coordinator._score_variants([{"subject": "a", "relation": "b", "object": "c"}])
    assert variants[0].score == 0.0


def test_apply_winners_writes_artifact(monkeypatch, swarm_cfg, tmp_path):
    cfg = dict(swarm_cfg)
    cfg["initiative"] = {"repo_path": str(tmp_path)}
    coordinator = SwarmCoordinator(cfg)
    coordinator.db.ensure_schema()
    monkeypatch.setattr("symbiont.agents.swarm.graphrag.add_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "symbiont.agents.swarm.analyze_plan", lambda _: {"flags": {"risk": "high"}}
    )

    winners = [
        SwarmVariant(
            triple={"subject": "repo", "relation": "adds", "object": "tests"},
            score=0.9,
            justification="good",
            agent_id="agent-1",
        )
    ]
    coordinator._apply_winners(winners, winners[0].triple)

    artifacts = Path(cfg["initiative"]["repo_path"]) / "data" / "artifacts" / "swarm"
    files = list(artifacts.glob("swarm_*.json"))
    assert files
    raw = files[0].read_text()
    head, *_tail = raw.split("\n\n# guard_flags", 1)
    content = json.loads(head)
    assert content["winners"][0]["agent_id"] == "agent-1"
    assert "guard_flags" in raw


def test_merge_from_transcripts_archives_invalid(monkeypatch, swarm_cfg, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cfg = dict(swarm_cfg)
    cfg["initiative"] = {"repo_path": str(repo_root)}
    coordinator = SwarmCoordinator(cfg)

    transcripts_dir = Path(cfg["initiative"]["repo_path"]) / "data" / "artifacts" / "ai_peer"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    (transcripts_dir / "nested").mkdir()
    (transcripts_dir / "peer_bad.json").write_text("not json", encoding="utf-8")
    (transcripts_dir / "peer_missing.json").write_text(
        json.dumps({"prompt": "no triple", "response": "{}", "simulated": True, "timestamp": 1}),
        encoding="utf-8",
    )
    (transcripts_dir / "peer_bad_response.json").write_text(
        json.dumps(
            {
                "prompt": "Triple: {'subject': 'repo', 'relation': 'adds', 'object': 'tests'}",
                "response": "not-json",
                "simulated": True,
                "timestamp": 2,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("symbiont.agents.swarm.graphrag.add_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr("symbiont.agents.swarm.analyze_plan", lambda _: {})

    winners = coordinator.merge_from_transcripts()
    assert winners == []
    processed = transcripts_dir / "processed"
    assert {p.name for p in processed.iterdir()} == {
        "peer_bad.json",
        "peer_missing.json",
        "peer_bad_response.json",
    }


def test_merge_variants_handles_empty_list(swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    assert coordinator._merge_variants([]) == []


def test_parse_triple_from_prompt_handles_invalid(swarm_cfg):
    coordinator = SwarmCoordinator(swarm_cfg)
    assert coordinator._parse_triple_from_prompt("") is None
    assert coordinator._parse_triple_from_prompt("Triple: notdict") is None
