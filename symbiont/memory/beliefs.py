from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

from .db import MemoryDB


def list_beliefs(db: MemoryDB) -> List[Dict[str, Any]]:
    with db._conn() as c:
        rows = c.execute(
            "SELECT id, statement, confidence, evidence_json, created_at, updated_at FROM beliefs ORDER BY id DESC"
        ).fetchall()
    return [
        {
            "id": r[0],
            "statement": r[1],
            "confidence": r[2],
            "evidence_json": r[3] or "",
            "created_at": r[4],
            "updated_at": r[5],
        }
        for r in rows
    ]


def add_belief(db: MemoryDB, statement: str, confidence: float = 0.5, evidence_json: str = "") -> int:
    now = int(time.time())
    with db._conn() as c:
        cur = c.execute(
            "INSERT INTO beliefs (statement, confidence, evidence_json, created_at, updated_at) VALUES (?,?,?,?,?)",
            (statement, confidence, evidence_json, now, now),
        )
        return cur.lastrowid


def update_belief(db: MemoryDB, belief_id: int, statement: Optional[str] = None, confidence: Optional[float] = None, evidence_json: Optional[str] = None):
    fields = []
    vals = []
    if statement is not None:
        fields.append("statement = ?")
        vals.append(statement)
    if confidence is not None:
        fields.append("confidence = ?")
        vals.append(confidence)
    if evidence_json is not None:
        fields.append("evidence_json = ?")
        vals.append(evidence_json)
    fields.append("updated_at = ?")
    vals.append(int(time.time()))
    vals.append(belief_id)
    if not fields:
        return
    with db._conn() as c:
        c.execute(f"UPDATE beliefs SET {', '.join(fields)} WHERE id = ?", tuple(vals))


def delete_belief(db: MemoryDB, belief_id: int):
    with db._conn() as c:
        c.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))

