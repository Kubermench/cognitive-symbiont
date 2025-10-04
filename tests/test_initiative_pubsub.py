import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.initiative.pubsub import PubSubClient, get_client


def test_pubsub_memory_backend(tmp_path):
    log_path = tmp_path / "events.jsonl"
    client = PubSubClient(
        {
            "enabled": True,
            "backend": "memory",
            "log_path": str(log_path),
        }
    )

    assert client.enabled is True
    client.publish({"hello": "world"})

    data = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(data[0]) == {"hello": "world"}


def test_pubsub_disabled_when_not_enabled():
    client = PubSubClient({"enabled": False})
    assert client.enabled is False


def test_get_client_reads_initiative_section():
    cfg = {
        "initiative": {
            "pubsub": {"enabled": True, "backend": "memory", "log_path": "events.jsonl"}
        }
    }
    client = get_client(cfg)
    assert client.enabled is True
