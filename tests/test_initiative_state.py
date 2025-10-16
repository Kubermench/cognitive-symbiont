import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbiont.initiative import state


def _cfg(tmp_path: Path) -> dict:
    return {
        "initiative": {
            "state": {
                "backend": "sqlite",
                "path": str(tmp_path / "state.db"),
                "swarm_ttl_seconds": 60,
            }
        }
    }


def test_state_store_records_daemon(tmp_path):
    cfg = _cfg(tmp_path)
    store = state.get_state_store(cfg)
    node_id = "test-node"
    store.record_daemon(
        node_id=node_id,
        pid=1234,
        status="running",
        poll_seconds=30,
        last_check_ts=111,
        last_proposal_ts=99,
        details={"foo": "bar"},
    )

    record = store.load_daemon(node_id)
    assert record is not None
    assert record["status"] == "running"
    assert record["details"]["foo"] == "bar"

    store.mark_daemon_stopped(node_id)
    stopped = store.load_daemon(node_id)
    assert stopped and stopped["status"] == "stopped"


def test_state_store_swarm_tracking(tmp_path):
    cfg = _cfg(tmp_path)
    store = state.get_state_store(cfg)
    worker_id = "worker-1"
    store.record_swarm_worker(
        worker_id=worker_id,
        node_id="node-a",
        status="running",
        goal="ship feature",
        variants=3,
        details={"apply": True},
    )

    workers = store.list_swarm_workers()
    assert any(w["worker_id"] == worker_id for w in workers)

    store.clear_swarm_worker(worker_id)
    assert not any(w["worker_id"] == worker_id for w in store.list_swarm_workers())


def test_resolve_node_id_prefers_cfg(monkeypatch):
    monkeypatch.setenv("SYMBIONT_NODE_ID", "env-node")
    cfg = {"initiative": {"node_id": "cfg-node"}}
    assert state.resolve_node_id(cfg) == "cfg-node"
    assert state.resolve_node_id({}) == "env-node"

