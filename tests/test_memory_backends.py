from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json

import pytest

from symbiont.memory.backends import GraphMemoryBackend, Mem0Backend, LettaBackend
from symbiont.memory.db import MemoryDB


class _Mem0Recorder:
    def __init__(self, options: Dict[str, Any]):
        self.saved = []
        self.options = options

    def is_ready(self) -> bool:
        return True

    def upsert(self, *, kind: str, key: str, payload: Dict[str, Any], user_id: str | None) -> None:
        self.saved.append((kind, key, payload, user_id))

    def load(self, *, kind: str, key: str, user_id: str | None):
        return {"kind": kind, "key": key}


class _Mem0Failing:
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def is_ready(self) -> bool:
        return True

    def upsert(self, *, kind: str, key: str, payload: Dict[str, Any], user_id: str | None) -> None:
        raise RuntimeError("boom")

    def load(self, *, kind: str, key: str, user_id: str | None):
        raise RuntimeError("boom")


class _LettaRecorder:
    def __init__(self, options: Dict[str, Any]):
        self.sessions = []
        self.prefs = []

    def is_ready(self) -> bool:
        return True

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        self.sessions.append((session_id, state))

    def load_session_state(self, session_id: str):
        return {"session": session_id}

    def save_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self.prefs.append((user_id, preferences))

    def load_preferences(self, user_id: str):
        return {"user": user_id}


class _LettaFailing:
    def __init__(self, options: Dict[str, Any]):
        self.options = options

    def is_ready(self) -> bool:
        return True

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        raise RuntimeError("nope")

    def load_session_state(self, session_id: str):
        raise RuntimeError("nope")

    def save_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        raise RuntimeError("nope")

    def load_preferences(self, user_id: str):
        raise RuntimeError("nope")


@pytest.fixture
def temp_db(tmp_path: Path) -> MemoryDB:
    db_path = tmp_path / "sym.db"
    return MemoryDB(db_path=str(db_path))


def test_mem0_backend_remote_and_local(monkeypatch, tmp_path: Path, temp_db: MemoryDB):
    recorder = _Mem0Recorder({})
    monkeypatch.setattr("symbiont.memory.backends._Mem0Adapter", lambda options: recorder)
    backend = Mem0Backend(temp_db, options={})

    backend.remember_preferences("alice", {"theme": "dark"})
    backend.save_session_state("sess-1", {"step": 1})

    assert recorder.saved[0][0] == "preferences"
    assert recorder.saved[1][0] == "session_state"

    # Remote load returns adapter payload
    assert backend.recall_preferences("alice") == {"kind": "preferences", "key": "alice"}
    assert backend.load_session_state("sess-1") == {"kind": "session_state", "key": "sess-1"}

    # Local fallback should also persist to disk
    pref_path = tmp_path / "memory_layers" / "preferences" / "alice.json"
    assert pref_path.exists()
    data = json.loads(pref_path.read_text())
    assert data["theme"] == "dark"


def test_mem0_backend_falls_back_on_failure(monkeypatch, tmp_path: Path, temp_db: MemoryDB):
    monkeypatch.setattr("symbiont.memory.backends._Mem0Adapter", lambda options: _Mem0Failing(options))
    backend = Mem0Backend(temp_db, options={})
    backend.remember_preferences("bob", {"mode": "safe"})

    pref_path = tmp_path / "memory_layers" / "preferences" / "bob.json"
    assert pref_path.exists()
    stored = json.loads(pref_path.read_text())
    assert stored["mode"] == "safe"
    # Remote load fails so local copy should be returned
    assert backend.recall_preferences("bob") == {"mode": "safe"}


def test_letta_backend_remote_and_local(monkeypatch, tmp_path: Path, temp_db: MemoryDB):
    recorder = _LettaRecorder({})
    monkeypatch.setattr("symbiont.memory.backends._LettaAdapter", lambda options: recorder)
    backend = LettaBackend(temp_db, options={})

    backend.save_session_state("cycle-1", {"progress": 0.5})
    backend.remember_preferences("carol", {"gpu": "mps"})

    assert recorder.sessions == [("cycle-1", {"progress": 0.5})]
    assert recorder.prefs == [("carol", {"gpu": "mps"})]
    assert backend.load_session_state("cycle-1") == {"session": "cycle-1"}
    assert backend.recall_preferences("carol") == {"user": "carol"}

    session_path = tmp_path / "memory_layers" / "session_state" / "cycle-1.json"
    assert session_path.exists()
    assert json.loads(session_path.read_text())["progress"] == 0.5


def test_letta_backend_fallback(monkeypatch, tmp_path: Path, temp_db: MemoryDB):
    monkeypatch.setattr("symbiont.memory.backends._LettaAdapter", lambda options: _LettaFailing(options))
    backend = LettaBackend(temp_db, options={})

    backend.save_session_state("cycle-2", {"progress": 0.9})
    assert backend.load_session_state("cycle-2") == {"progress": 0.9}


def test_graph_backend_incremental_indexing(tmp_path: Path):
    db_path = tmp_path / "sym.db"
    db = MemoryDB(db_path=str(db_path))
    db.ensure_schema()
    backend = GraphMemoryBackend(db)

    db.add_message("user", "hello")
    db.add_message("assistant", "world")
    art_id = db.add_artifact(task_id=None, kind="note", path="notes.md", summary="initial summary")

    first_count = backend.build_indices()
    with db._conn() as conn:
        vector_rows = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert first_count == vector_rows == 3

    # Drop existing message vectors to simulate backlog larger than the cap.
    with db._conn() as conn:
        conn.execute("DELETE FROM vectors WHERE kind='message'")
    limited_backfill = backend.build_indices(limit_if_new=1)
    assert limited_backfill == 1
    remaining_backfill = backend.build_indices(limit_if_new=1)
    assert remaining_backfill == 1
    assert backend.build_indices(limit_if_new=1) == 0
    with db._conn() as conn:
        vector_rows = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    assert vector_rows == 3

    db.add_message("user", "new input")
    second_count = backend.build_indices()
    assert second_count == 1

    # Simulate a missing artifact embedding and ensure the capped rebuild only
    # processes one record per invocation.
    with db._conn() as conn:
        conn.execute(
            "DELETE FROM vectors WHERE kind='artifact' AND ref_table='artifacts' AND ref_id=?",
            (art_id,),
        )
    backfill_count = backend.build_indices(limit_if_new=1)
    assert backfill_count == 1

    # Add multiple new messages and cap indexing to force backlog handling.
    db.add_message("assistant", "later one")
    db.add_message("user", "later two")
    limited_count = backend.build_indices(limit_if_new=1)
    assert limited_count == 1

    final_count = backend.build_indices()
    assert final_count == 1
    with db._conn() as conn:
        total_vectors = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    # All five messages plus one artifact should now be embedded.
    assert total_vectors == 6
