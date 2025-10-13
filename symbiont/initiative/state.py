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
                    updated_at INTEGER,
                    heartbeat_ts INTEGER DEFAULT 0,
                    version TEXT DEFAULT '1.0',
                    capabilities TEXT DEFAULT '{}',
                    load_avg REAL DEFAULT 0.0,
                    memory_usage REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS swarm_workers (
                    worker_id TEXT PRIMARY KEY,
                    node_id TEXT,
                    status TEXT,
                    active_goal TEXT,
                    variants INTEGER,
                    details TEXT,
                    updated_at INTEGER,
                    heartbeat_ts INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 0,
                    assigned_at INTEGER DEFAULT 0,
                    completed_at INTEGER DEFAULT 0,
                    progress REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS coordination_locks (
                    lock_name TEXT PRIMARY KEY,
                    holder_node_id TEXT,
                    acquired_at INTEGER,
                    expires_at INTEGER,
                    purpose TEXT,
                    details TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS node_discovery (
                    node_id TEXT PRIMARY KEY,
                    hostname TEXT,
                    ip_address TEXT,
                    port INTEGER,
                    capabilities TEXT DEFAULT '{}',
                    last_seen INTEGER,
                    status TEXT DEFAULT 'active',
                    version TEXT DEFAULT '1.0'
                );

                CREATE INDEX IF NOT EXISTS idx_daemon_heartbeat ON daemon_state(heartbeat_ts);
                CREATE INDEX IF NOT EXISTS idx_swarm_heartbeat ON swarm_workers(heartbeat_ts);
                CREATE INDEX IF NOT EXISTS idx_coordination_expires ON coordination_locks(expires_at);
                CREATE INDEX IF NOT EXISTS idx_node_last_seen ON node_discovery(last_seen);
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
        capabilities: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
        load_avg: float = 0.0,
        memory_usage: float = 0.0,
    ) -> None:
        payload = json.dumps(details or {}, separators=(",", ":"))
        capabilities_json = json.dumps(capabilities or {}, separators=(",", ":"))
        heartbeat_ts = int(time.time())
        
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daemon_state (
                    node_id, pid, status, poll_seconds, last_check_ts, last_proposal_ts, 
                    details, updated_at, heartbeat_ts, version, capabilities, load_avg, memory_usage
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%s','now'), ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    pid=excluded.pid,
                    status=excluded.status,
                    poll_seconds=excluded.poll_seconds,
                    last_check_ts=excluded.last_check_ts,
                    last_proposal_ts=excluded.last_proposal_ts,
                    details=excluded.details,
                    updated_at=excluded.updated_at,
                    heartbeat_ts=excluded.heartbeat_ts,
                    version=excluded.version,
                    capabilities=excluded.capabilities,
                    load_avg=excluded.load_avg,
                    memory_usage=excluded.memory_usage;
                """,
                (
                    node_id,
                    int(pid),
                    status,
                    int(poll_seconds),
                    int(last_check_ts),
                    int(last_proposal_ts),
                    payload,
                    heartbeat_ts,
                    version,
                    capabilities_json,
                    float(load_avg),
                    float(memory_usage),
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

    def acquire_coordination_lock(
        self,
        lock_name: str,
        holder_node_id: str,
        purpose: str,
        ttl_seconds: int = 300,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Acquire a distributed coordination lock."""
        now = int(time.time())
        expires_at = now + ttl_seconds
        details_json = json.dumps(details or {}, separators=(",", ":"))
        
        with self._conn() as conn:
            # Clean up expired locks first
            conn.execute("DELETE FROM coordination_locks WHERE expires_at < ?", (now,))
            
            # Try to acquire the lock
            try:
                conn.execute(
                    """
                    INSERT INTO coordination_locks (lock_name, holder_node_id, acquired_at, expires_at, purpose, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (lock_name, holder_node_id, now, expires_at, purpose, details_json),
                )
                return True
            except sqlite3.IntegrityError:
                # Lock already exists and not expired
                return False

    def release_coordination_lock(self, lock_name: str, holder_node_id: str) -> bool:
        """Release a coordination lock if held by the specified node."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM coordination_locks WHERE lock_name = ? AND holder_node_id = ?",
                (lock_name, holder_node_id),
            )
            return cursor.rowcount > 0

    def list_coordination_locks(self) -> list[Dict[str, Any]]:
        """List all active coordination locks."""
        now = int(time.time())
        with self._conn() as conn:
            # Clean up expired locks
            conn.execute("DELETE FROM coordination_locks WHERE expires_at < ?", (now,))
            
            rows = conn.execute(
                "SELECT lock_name, holder_node_id, acquired_at, expires_at, purpose, details FROM coordination_locks"
            ).fetchall()
        
        locks = []
        for row in rows:
            try:
                details = json.loads(row[5] or "{}")
            except Exception:
                details = {}
            
            locks.append({
                "lock_name": row[0],
                "holder_node_id": row[1],
                "acquired_at": row[2],
                "expires_at": row[3],
                "purpose": row[4],
                "details": details,
                "ttl_remaining": max(0, row[3] - now),
            })
        return locks

    def register_node(
        self,
        node_id: str,
        hostname: str,
        ip_address: str,
        port: int = 0,
        capabilities: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ) -> None:
        """Register a node for discovery."""
        capabilities_json = json.dumps(capabilities or {}, separators=(",", ":"))
        now = int(time.time())
        
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO node_discovery (node_id, hostname, ip_address, port, capabilities, last_seen, status, version)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    hostname=excluded.hostname,
                    ip_address=excluded.ip_address,
                    port=excluded.port,
                    capabilities=excluded.capabilities,
                    last_seen=excluded.last_seen,
                    status=excluded.status,
                    version=excluded.version;
                """,
                (node_id, hostname, ip_address, port, capabilities_json, now, version),
            )

    def discover_nodes(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        """Discover active nodes in the cluster."""
        cutoff = int(time.time()) - max_age_seconds
        
        with self._conn() as conn:
            # Mark stale nodes as inactive
            conn.execute(
                "UPDATE node_discovery SET status = 'inactive' WHERE last_seen < ? AND status = 'active'",
                (cutoff,)
            )
            
            rows = conn.execute(
                "SELECT node_id, hostname, ip_address, port, capabilities, last_seen, status, version FROM node_discovery WHERE status = 'active'"
            ).fetchall()
        
        nodes = []
        for row in rows:
            try:
                capabilities = json.loads(row[4] or "{}")
            except Exception:
                capabilities = {}
            
            nodes.append({
                "node_id": row[0],
                "hostname": row[1],
                "ip_address": row[2],
                "port": row[3],
                "capabilities": capabilities,
                "last_seen": row[5],
                "status": row[6],
                "version": row[7],
            })
        return nodes

    def heartbeat_daemon(self, node_id: str) -> None:
        """Send a heartbeat for a daemon node."""
        now = int(time.time())
        with self._conn() as conn:
            conn.execute(
                "UPDATE daemon_state SET heartbeat_ts = ? WHERE node_id = ?",
                (now, node_id),
            )

    def heartbeat_worker(self, worker_id: str) -> None:
        """Send a heartbeat for a swarm worker."""
        now = int(time.time())
        with self._conn() as conn:
            conn.execute(
                "UPDATE swarm_workers SET heartbeat_ts = ? WHERE worker_id = ?",
                (now, worker_id),
            )

    def get_stale_daemons(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        """Get daemons that haven't sent heartbeats recently."""
        cutoff = int(time.time()) - max_age_seconds
        
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT node_id, pid, status, heartbeat_ts, updated_at FROM daemon_state WHERE heartbeat_ts < ? AND status != 'stopped'",
                (cutoff,)
            ).fetchall()
        
        return [
            {
                "node_id": row[0],
                "pid": row[1],
                "status": row[2],
                "heartbeat_ts": row[3],
                "updated_at": row[4],
                "stale_seconds": int(time.time()) - row[3],
            }
            for row in rows
        ]

    def get_stale_workers(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        """Get swarm workers that haven't sent heartbeats recently."""
        cutoff = int(time.time()) - max_age_seconds
        
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT worker_id, node_id, status, heartbeat_ts, updated_at FROM swarm_workers WHERE heartbeat_ts < ?",
                (cutoff,)
            ).fetchall()
        
        return [
            {
                "worker_id": row[0],
                "node_id": row[1],
                "status": row[2],
                "heartbeat_ts": row[3],
                "updated_at": row[4],
                "stale_seconds": int(time.time()) - row[3],
            }
            for row in rows
        ]


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

    def acquire_coordination_lock(
        self,
        lock_name: str,
        holder_node_id: str,
        purpose: str,
        ttl_seconds: int = 300,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Acquire a distributed coordination lock using Redis."""
        key = f"{self.prefix}:lock:{lock_name}"
        payload = {
            "holder_node_id": holder_node_id,
            "acquired_at": int(time.time()),
            "purpose": purpose,
            "details": details or {},
        }
        # Use SET with NX (only if not exists) and EX (expiry)
        return bool(self.client.set(key, json.dumps(payload), nx=True, ex=ttl_seconds))

    def release_coordination_lock(self, lock_name: str, holder_node_id: str) -> bool:
        """Release a coordination lock if held by the specified node."""
        key = f"{self.prefix}:lock:{lock_name}"
        # Lua script to atomically check holder and delete
        lua_script = """
        local key = KEYS[1]
        local holder = ARGV[1]
        local current = redis.call('GET', key)
        if current then
            local data = cjson.decode(current)
            if data.holder_node_id == holder then
                redis.call('DEL', key)
                return 1
            end
        end
        return 0
        """
        result = self.client.eval(lua_script, 1, key, holder_node_id)
        return bool(result)

    def list_coordination_locks(self) -> list[Dict[str, Any]]:
        """List all active coordination locks."""
        pattern = f"{self.prefix}:lock:*"
        locks = []
        for key in self.client.scan_iter(pattern):
            data = self.client.get(key)
            if data:
                try:
                    lock_data = json.loads(data)
                    lock_name = key.decode().split(":")[-1]
                    ttl = self.client.ttl(key)
                    locks.append({
                        "lock_name": lock_name,
                        "holder_node_id": lock_data.get("holder_node_id"),
                        "acquired_at": lock_data.get("acquired_at"),
                        "purpose": lock_data.get("purpose"),
                        "details": lock_data.get("details", {}),
                        "ttl_remaining": max(0, ttl) if ttl > 0 else 0,
                    })
                except Exception:
                    continue
        return locks

    def register_node(
        self,
        node_id: str,
        hostname: str,
        ip_address: str,
        port: int = 0,
        capabilities: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ) -> None:
        """Register a node for discovery."""
        key = f"{self.prefix}:node:{node_id}"
        payload = {
            "node_id": node_id,
            "hostname": hostname,
            "ip_address": ip_address,
            "port": port,
            "capabilities": capabilities or {},
            "last_seen": int(time.time()),
            "status": "active",
            "version": version,
        }
        self.client.set(key, json.dumps(payload), ex=self.ttl_seconds)

    def discover_nodes(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        """Discover active nodes in the cluster."""
        pattern = f"{self.prefix}:node:*"
        nodes = []
        cutoff = int(time.time()) - max_age_seconds
        
        for key in self.client.scan_iter(pattern):
            data = self.client.get(key)
            if data:
                try:
                    node_data = json.loads(data)
                    if node_data.get("last_seen", 0) >= cutoff:
                        nodes.append(node_data)
                except Exception:
                    continue
        return nodes

    def heartbeat_daemon(self, node_id: str) -> None:
        """Send a heartbeat for a daemon node."""
        key = f"{self.prefix}:daemon:{node_id}"
        data = self.client.get(key)
        if data:
            payload = json.loads(data)
            payload["heartbeat_ts"] = int(time.time())
            self.client.set(key, json.dumps(payload), ex=self.ttl_seconds)

    def heartbeat_worker(self, worker_id: str) -> None:
        """Send a heartbeat for a swarm worker."""
        key = f"{self.prefix}:swarm:{worker_id}"
        data = self.client.get(key)
        if data:
            payload = json.loads(data)
            payload["heartbeat_ts"] = int(time.time())
            self.client.set(key, json.dumps(payload), ex=self.ttl_seconds)

    def get_stale_daemons(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        """Get daemons that haven't sent heartbeats recently."""
        cutoff = int(time.time()) - max_age_seconds
        stale = []
        
        for key in self.client.scan_iter(f"{self.prefix}:daemon:*"):
            data = self.client.get(key)
            if data:
                try:
                    daemon_data = json.loads(data)
                    heartbeat_ts = daemon_data.get("heartbeat_ts", 0)
                    if heartbeat_ts < cutoff and daemon_data.get("status") != "stopped":
                        daemon_data["stale_seconds"] = int(time.time()) - heartbeat_ts
                        stale.append(daemon_data)
                except Exception:
                    continue
        return stale

    def get_stale_workers(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        """Get swarm workers that haven't sent heartbeats recently."""
        cutoff = int(time.time()) - max_age_seconds
        stale = []
        
        for key in self.client.scan_iter(f"{self.prefix}:swarm:*"):
            data = self.client.get(key)
            if data:
                try:
                    worker_data = json.loads(data)
                    heartbeat_ts = worker_data.get("heartbeat_ts", 0)
                    if heartbeat_ts < cutoff:
                        worker_data["stale_seconds"] = int(time.time()) - heartbeat_ts
                        stale.append(worker_data)
                except Exception:
                    continue
        return stale


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

    def acquire_coordination_lock(
        self,
        lock_name: str,
        holder_node_id: str,
        purpose: str,
        ttl_seconds: int = 300,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self._backend.acquire_coordination_lock(
            lock_name, holder_node_id, purpose, ttl_seconds, details
        )

    def release_coordination_lock(self, lock_name: str, holder_node_id: str) -> bool:
        return self._backend.release_coordination_lock(lock_name, holder_node_id)

    def list_coordination_locks(self) -> list[Dict[str, Any]]:
        return self._backend.list_coordination_locks()

    def register_node(
        self,
        node_id: str,
        hostname: str,
        ip_address: str,
        port: int = 0,
        capabilities: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ) -> None:
        return self._backend.register_node(
            node_id, hostname, ip_address, port, capabilities, version
        )

    def discover_nodes(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        return self._backend.discover_nodes(max_age_seconds)

    def heartbeat_daemon(self, node_id: str) -> None:
        return self._backend.heartbeat_daemon(node_id)

    def heartbeat_worker(self, worker_id: str) -> None:
        return self._backend.heartbeat_worker(worker_id)

    def get_stale_daemons(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        return self._backend.get_stale_daemons(max_age_seconds)

    def get_stale_workers(self, max_age_seconds: int = 300) -> list[Dict[str, Any]]:
        return self._backend.get_stale_workers(max_age_seconds)


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

