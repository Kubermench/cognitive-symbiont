from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..tools.files import ensure_dirs

logger = logging.getLogger(__name__)


class SQLiteStateBackend:
    """SQLite-backed persistence for initiative + swarm heartbeat."""

    def __init__(self, path: Optional[str] = None, *, ttl_seconds: int = 900):
        default_path = Path("./data/initiative/state.db")
        self.path = Path(path).expanduser().resolve() if path else default_path.resolve()
        ensure_dirs([self.path.parent])
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS daemon_state (
                    node_id TEXT PRIMARY KEY,
                    pid INTEGER,
                    status TEXT,
                    poll_seconds INTEGER,
                    last_check_ts INTEGER,
                    last_proposal_ts INTEGER,
                    details TEXT,
                    updated_at INTEGER
                );

                CREATE TABLE IF NOT EXISTS swarm_workers (
                    worker_id TEXT PRIMARY KEY,
                    node_id TEXT,
                    status TEXT,
                    active_goal TEXT,
                    variants INTEGER,
                    details TEXT,
                    updated_at INTEGER
                );
                """
            )

    def record_daemon(
        self,
        *,
        node_id: str,
        pid: int,
        status: str,
        poll_seconds: int,
        last_check_ts: int,
        last_proposal_ts: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = json.dumps(details or {}, separators=(",", ":"))
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daemon_state (node_id, pid, status, poll_seconds, last_check_ts, last_proposal_ts, details, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(node_id) DO UPDATE SET
                    pid=excluded.pid,
                    status=excluded.status,
                    poll_seconds=excluded.poll_seconds,
                    last_check_ts=excluded.last_check_ts,
                    last_proposal_ts=excluded.last_proposal_ts,
                    details=excluded.details,
                    updated_at=excluded.updated_at;
                """,
                (
                    node_id,
                    int(pid),
                    status,
                    int(poll_seconds),
                    int(last_check_ts),
                    int(last_proposal_ts),
                    payload,
                ),
            )

    def mark_daemon_stopped(self, node_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE daemon_state
                SET status='stopped', updated_at=strftime('%s','now')
                WHERE node_id=?;
                """,
                (node_id,),
            )

    def load_daemon(self, node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT node_id, pid, status, poll_seconds, last_check_ts, last_proposal_ts, details, updated_at FROM daemon_state"
        params: Iterable[Any]
        if node_id:
            query += " WHERE node_id=?"
            params = (node_id,)
        else:
            query += " ORDER BY updated_at DESC"
            params = ()

        with self._conn() as conn:
            row = conn.execute(query + " LIMIT 1", params).fetchone()
        if not row:
            return None
        try:
            details = json.loads(row[6] or "{}")
        except Exception:
            details = {}
        return {
            "node_id": row[0],
            "pid": row[1],
            "status": row[2],
            "poll_seconds": row[3],
            "last_check_ts": row[4],
            "last_proposal_ts": row[5],
            "details": details,
            "updated_at": row[7],
        }

    def record_swarm_worker(
        self,
        *,
        worker_id: str,
        node_id: str,
        status: str,
        goal: Optional[str] = None,
        variants: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = json.dumps(details or {}, separators=(",", ":"))
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO swarm_workers (worker_id, node_id, status, active_goal, variants, details, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(worker_id) DO UPDATE SET
                    node_id=excluded.node_id,
                    status=excluded.status,
                    active_goal=excluded.active_goal,
                    variants=excluded.variants,
                    details=excluded.details,
                    updated_at=excluded.updated_at;
                """,
                (
                    worker_id,
                    node_id,
                    status,
                    goal,
                    variants if variants is not None else None,
                    payload,
                ),
            )

    def clear_swarm_worker(self, worker_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM swarm_workers WHERE worker_id=?", (worker_id,))

    def list_swarm_workers(self) -> list[Dict[str, Any]]:
        cutoff = int(time.time()) - self.ttl_seconds
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT worker_id, node_id, status, active_goal, variants, details, updated_at FROM swarm_workers"
            ).fetchall()
            conn.execute("DELETE FROM swarm_workers WHERE updated_at < ?", (cutoff,))

        workers: list[Dict[str, Any]] = []
        for row in rows:
            try:
                details = json.loads(row[5] or "{}")
            except Exception:
                details = {}
            if row[6] < cutoff:
                continue
            workers.append(
                {
                    "worker_id": row[0],
                    "node_id": row[1],
                    "status": row[2],
                    "active_goal": row[3],
                    "variants": row[4],
                    "details": details,
                    "updated_at": row[6],
                }
            )
        return workers


class RedisStateBackend:
    """Optional Redis backend when a shared cache is available."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - redis optional
            raise RuntimeError("redis package not installed") from exc

        url = cfg.get("url")
        if url:
            self.client = redis.from_url(url)
        else:
            host = cfg.get("host", "localhost")
            port = int(cfg.get("port", 6379))
            db = int(cfg.get("db", 0))
            self.client = redis.Redis(host=host, port=port, db=db)
        self.prefix = cfg.get("prefix", "symbiont:state")
        self.ttl_seconds = int(cfg.get("ttl_seconds", 900))

    def record_daemon(
        self,
        *,
        node_id: str,
        pid: int,
        status: str,
        poll_seconds: int,
        last_check_ts: int,
        last_proposal_ts: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "node_id": node_id,
            "pid": int(pid),
            "status": status,
            "poll_seconds": int(poll_seconds),
            "last_check_ts": int(last_check_ts),
            "last_proposal_ts": int(last_proposal_ts),
            "details": details or {},
            "updated_at": int(time.time()),
        }
        key = f"{self.prefix}:daemon:{node_id}"
        self.client.set(key, json.dumps(payload), ex=self.ttl_seconds)

    def mark_daemon_stopped(self, node_id: str) -> None:
        key = f"{self.prefix}:daemon:{node_id}"
        data = self.client.get(key)
        if not data:
            return
        payload = json.loads(data)
        payload["status"] = "stopped"
        payload["updated_at"] = int(time.time())
        self.client.set(key, json.dumps(payload), ex=self.ttl_seconds)

    def load_daemon(self, node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if node_id:
            data = self.client.get(f"{self.prefix}:daemon:{node_id}")
            if not data:
                return None
            return json.loads(data)
        keys = sorted(self.client.scan_iter(f"{self.prefix}:daemon:*"))
        if not keys:
            return None
        return json.loads(self.client.get(keys[-1]) or b"{}")

    def record_swarm_worker(
        self,
        *,
        worker_id: str,
        node_id: str,
        status: str,
        goal: Optional[str] = None,
        variants: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "worker_id": worker_id,
            "node_id": node_id,
            "status": status,
            "active_goal": goal,
            "variants": variants,
            "details": details or {},
            "updated_at": int(time.time()),
        }
        key = f"{self.prefix}:swarm:{worker_id}"
        self.client.set(key, json.dumps(payload), ex=self.ttl_seconds)

    def clear_swarm_worker(self, worker_id: str) -> None:
        self.client.delete(f"{self.prefix}:swarm:{worker_id}")

    def list_swarm_workers(self) -> list[Dict[str, Any]]:
        workers = []
        for key in self.client.scan_iter(f"{self.prefix}:swarm:*"):
            data = self.client.get(key)
            if data:
                workers.append(json.loads(data))
        return workers


class InitiativeStateStore:
    """Facade wrapping the configured backend for initiative/swarm state."""

    def __init__(self, backend: str, cfg: Dict[str, Any]):
        backend = (backend or "sqlite").lower()
        ttl = int(cfg.get("swarm_ttl_seconds", 900) or 900)
        if backend == "redis":
            try:
                self._backend = RedisStateBackend(cfg.get("redis", {}))
                self.kind = "redis"
                return
            except Exception as exc:  # pragma: no cover - requires redis env
                logger.warning("Redis state backend unavailable (%s); falling back to SQLite", exc)
        path = cfg.get("path")
        self._backend = SQLiteStateBackend(path, ttl_seconds=ttl)
        self.kind = "sqlite"

    def record_daemon(self, **kwargs: Any) -> None:
        self._backend.record_daemon(**kwargs)

    def mark_daemon_stopped(self, node_id: str) -> None:
        self._backend.mark_daemon_stopped(node_id)

    def load_daemon(self, node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self._backend.load_daemon(node_id)

    def record_swarm_worker(self, **kwargs: Any) -> None:
        self._backend.record_swarm_worker(**kwargs)

    def clear_swarm_worker(self, worker_id: str) -> None:
        self._backend.clear_swarm_worker(worker_id)

    def list_swarm_workers(self) -> list[Dict[str, Any]]:
        return self._backend.list_swarm_workers()


_STORE_CACHE: Dict[tuple, InitiativeStateStore] = {}
_STORE_LOCK = threading.Lock()


def _store_cache_key(cfg: Dict[str, Any]) -> tuple:
    ini = cfg.get("initiative", {}) if isinstance(cfg, dict) else {}
    state_cfg = ini.get("state", {}) or {}
    backend = (state_cfg.get("backend") or "sqlite").lower()
    if backend == "redis":
        redis_cfg = state_cfg.get("redis", {}) or {}
        url = redis_cfg.get("url")
        if url:
            return ("redis", url)
        return (
            "redis",
            redis_cfg.get("host", "localhost"),
            int(redis_cfg.get("port", 6379)),
            int(redis_cfg.get("db", 0)),
        )
    path = state_cfg.get("path") or "./data/initiative/state.db"
    return ("sqlite", str(Path(path).expanduser().resolve()))


def get_state_store(cfg: Optional[Dict[str, Any]] = None) -> InitiativeStateStore:
    cfg = cfg or {}
    key = _store_cache_key(cfg)
    with _STORE_LOCK:
        store = _STORE_CACHE.get(key)
        if store is None:
            ini = cfg.get("initiative", {}) if isinstance(cfg, dict) else {}
            state_cfg = ini.get("state", {}) or {}
            backend = state_cfg.get("backend", "sqlite")
            store = InitiativeStateStore(backend, state_cfg)
            _STORE_CACHE[key] = store
        return store


def resolve_node_id(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or {}
    ini = cfg.get("initiative", {}) if isinstance(cfg, dict) else {}
    explicit = ini.get("node_id") or os.getenv("SYMBIONT_NODE_ID")
    if explicit:
        return str(explicit)
    return socket.gethostname()


def generate_worker_id(prefix: str = "swarm") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"

