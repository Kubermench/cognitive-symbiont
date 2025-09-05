from __future__ import annotations
import os, sqlite3, time
from typing import List, Dict, Any

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

class MemoryDB:
    def __init__(self, db_path: str = "./data/symbiont.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def ensure_schema(self):
        with open(SCHEMA_PATH,"r",encoding="utf-8") as f: ddl=f.read()
        with self._conn() as c: c.executescript(ddl)

    def start_episode(self, title: str) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO episodes (title, started_at, status) VALUES (?, ?, ?)", (title, int(time.time()), "open"))
            return cur.lastrowid

    def add_message(self, role: str, content: str, tags: str = ""):
        with self._conn() as c:
            c.execute("INSERT INTO messages (role, content, created_at, tags) VALUES (?, ?, ?, ?)", (role, content, int(time.time()), tags))

    def last_messages(self, limit: int = 10):
        with self._conn() as c:
            rows = c.execute("SELECT id, role, content, created_at FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id":r[0],"role":r[1],"content":r[2],"ts":r[3]} for r in rows]
    
    def add_task(self, episode_id: int, description: str, status: str, assignee_role: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO tasks (episode_id, description, status, assignee_role, created_at) "
                "VALUES (?, ?, ?, ?, strftime('%s','now'))",
            (episode_id, description, status, assignee_role)
        )

