from __future__ import annotations
import os, sqlite3, time, json
from typing import List, Dict, Any, Optional

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")
SCHEMA_VERSION = 1

def _dedupe_vectors(conn: sqlite3.Connection) -> None:
    """Drop duplicate vector rows so unique constraints can succeed."""
    try:
        rows = conn.execute(
            """
            SELECT MIN(id) AS keep_id, kind, ref_table, ref_id
            FROM vectors
            GROUP BY kind, ref_table, ref_id
            HAVING COUNT(*) > 1
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return
    for keep_id, kind, ref_table, ref_id in rows or []:
        conn.execute(
            "DELETE FROM vectors WHERE kind=? AND ref_table=? AND ref_id=? AND id<>?",
            (kind, ref_table, ref_id, keep_id),
        )

def _ensure_vector_indexes(conn: sqlite3.Connection) -> None:
    """Ensure vector lookup indexes exist for fast hybrid search."""
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vectors_unique ON vectors(kind, ref_table, ref_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_ref_lookup ON vectors(ref_table, ref_id)")
    except sqlite3.OperationalError:
        pass

def _current_version(conn: sqlite3.Connection) -> Optional[int]:
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None

def _write_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(version),),
    )

class MemoryDB:
    def __init__(self, db_path: str = "./data/symbiont.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def ensure_schema(self):
        with open(SCHEMA_PATH,"r",encoding="utf-8") as f: ddl=f.read()
        with self._conn() as c:
            c.executescript(ddl)
            _dedupe_vectors(c)
            _ensure_vector_indexes(c)
            version = _current_version(c)
            if version is None:
                _write_version(c, SCHEMA_VERSION)
            elif version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Detected schema version {version} newer than supported {SCHEMA_VERSION}. "
                    "Upgrade the runtime before opening this database."
                )

    def start_episode(self, title: str) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO episodes (title, started_at, status) VALUES (?, ?, ?)", (title, int(time.time()), "open"))
            return cur.lastrowid

    def add_message(self, role: str, content: str, tags: str = ""):
        with self._conn() as c:
            c.execute("INSERT INTO messages (role, content, created_at, tags) VALUES (?, ?, ?, ?)", (role, content, int(time.time()), tags))

    def add_artifact(
        self,
        *,
        task_id: int | None,
        kind: str,
        path: str,
        summary: str | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO artifacts (task_id, type, path, summary, created_at) VALUES (?, ?, ?, ?, ?)",
                (task_id, kind, path, summary, int(time.time())),
            )
            return int(cur.lastrowid)

    def last_messages(self, limit: int = 10):
        with self._conn() as c:
            rows = c.execute("SELECT id, role, content, created_at FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id":r[0],"role":r[1],"content":r[2],"ts":r[3]} for r in rows]
    
    def add_task(
        self,
        episode_id: int | None,
        description: str,
        status: str,
        assignee_role: str,
        *,
        result: str | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO tasks (episode_id, description, status, assignee_role, result, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
                (episode_id, description, status, assignee_role, result),
            )
            return int(cur.lastrowid)

    def update_task_status(
        self,
        task_id: int,
        *,
        status: str,
        result: str | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET status = ?, result = ?, updated_at = strftime('%s','now') WHERE id = ?",
                (status, result, task_id),
            )

    def add_intent(self, episode_id: int, summary: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO intents (episode_id, summary, created_at, updated_at) VALUES (?, ?, strftime('%s','now'), strftime('%s','now'))",
                (episode_id, summary),
            )

    def latest_intent(self):
        with self._conn() as c:
            row = c.execute("SELECT episode_id, summary, updated_at FROM intents ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"episode_id": row[0], "summary": row[1], "updated_at": row[2]}

    def schema_version(self) -> Optional[int]:
        with self._conn() as c:
            return _current_version(c)

    def add_sd_run(
        self,
        *,
        goal: str,
        label: str,
        horizon: int,
        timestep: float,
        stats: Dict[str, Any],
        plot_path: str | None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO sd_runs (goal, label, horizon, timestep, stats_json, plot_path) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    goal,
                    label,
                    int(horizon),
                    float(timestep),
                    json.dumps(stats),
                    plot_path,
                ),
            )
