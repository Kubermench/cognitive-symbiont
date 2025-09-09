from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import sqlite3

from .db import MemoryDB


def _get_id(c: sqlite3.Connection, table: str, name: str) -> int:
    row = c.execute(f"SELECT id FROM {table} WHERE name=?", (name,)).fetchone()
    if row:
        return int(row[0])
    cur = c.execute(f"INSERT INTO {table} (name) VALUES (?)", (name,))
    return cur.lastrowid


def add_claim(db: MemoryDB, subject: str, relation: str, obj: str, importance: float = 0.5, source_url: Optional[str] = None) -> int:
    with db._conn() as c:
        sid = _get_id(c, 'entities', subject.strip())
        rid = _get_id(c, 'relations', relation.strip())
        cur = c.execute(
            "INSERT INTO claims (subject_id, relation_id, object, importance, source_url) VALUES (?,?,?,?,?)",
            (sid, rid, obj.strip(), float(importance), source_url or None),
        )
        return cur.lastrowid


def list_claims(db: MemoryDB, limit: int = 50) -> List[Dict[str, Any]]:
    with db._conn() as c:
        rows = c.execute(
            "SELECT cl.id, e.name, r.name, cl.object, cl.importance, cl.source_url, cl.updated_at "
            "FROM claims cl JOIN entities e ON cl.subject_id=e.id JOIN relations r ON cl.relation_id=r.id "
            "ORDER BY cl.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            'id': r[0],
            'subject': r[1],
            'relation': r[2],
            'object': r[3],
            'importance': r[4],
            'source_url': r[5],
            'updated_at': r[6],
        }
        for r in rows
    ]


def query_claims(db: MemoryDB, term: str, limit: int = 5) -> List[Dict[str, Any]]:
    t = f"%{term}%"
    with db._conn() as c:
        rows = c.execute(
            "SELECT cl.id, e.name, r.name, cl.object, cl.importance, cl.source_url, cl.updated_at "
            "FROM claims cl JOIN entities e ON cl.subject_id=e.id JOIN relations r ON cl.relation_id=r.id "
            "WHERE e.name LIKE ? OR r.name LIKE ? OR cl.object LIKE ? "
            "ORDER BY cl.importance DESC, cl.updated_at DESC LIMIT ?",
            (t, t, t, limit),
        ).fetchall()
    return [
        {
            'id': r[0],
            'subject': r[1],
            'relation': r[2],
            'object': r[3],
            'importance': r[4],
            'source_url': r[5],
            'updated_at': r[6],
        }
        for r in rows
    ]

