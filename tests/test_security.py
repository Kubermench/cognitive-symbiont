import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.memory.db import MemoryDB
from symbiont.tools.security import rotate_env_secret, scrub_payload, scrub_text
from symbiont.ports.ai_peer import AIPeerBridge


def test_scrub_text_redacts_pii():
    text = "Contact user@example.com or call +1 (555) 123-4567"
    sanitized = scrub_text(text)
    assert "example.com" not in sanitized
    assert "555" not in sanitized


def test_scrub_payload_recurses():
    payload = {
        "email": "user@example.com",
        "nested": {"note": "Token=secret-value"},
        "items": ["call +44 1234 567890"],
    }
    sanitized = scrub_payload(payload)
    assert sanitized["email"] == "[redacted-email]"
    assert sanitized["nested"]["note"].endswith("[redacted]")
    assert sanitized["items"][0] == "call [redacted-phone]"


def test_rotate_env_secret_logs(tmp_path):
    db_path = tmp_path / "sym.db"
    MemoryDB(str(db_path)).ensure_schema()
    rotate_env_secret("TEST_SECRET", "rotated", db_path=str(db_path), actor="tester")
    assert os.environ["TEST_SECRET"] == "rotated"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT actor, action FROM audit_logs ORDER BY id DESC LIMIT 1").fetchone()
    assert row == ("tester", "rotate:TEST_SECRET")


def test_rotate_env_secret_updates_dotenv(tmp_path):
    db_path = tmp_path / "sym.db"
    MemoryDB(str(db_path)).ensure_schema()
    dotenv = tmp_path / ".env"
    rotate_env_secret("ANOTHER_SECRET", "value123", db_path=str(db_path), persist_path=str(dotenv))
    content = dotenv.read_text(encoding="utf-8")
    assert "ANOTHER_SECRET=value123" in content


def test_peer_store_sanitizes(tmp_path):
    cfg = {
        "data_root": str(tmp_path),
        "ports": {"ai_peer": {"stub_mode": True}},
        "llm": {},
    }
    bridge = AIPeerBridge(cfg)
    path = bridge._store("Email me at user@example.com", "Reach me at admin@example.com", False, "agent-1")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "example.com" not in data["prompt"]
    assert "example.com" not in data["response"]
