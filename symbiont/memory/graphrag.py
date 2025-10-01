from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import sqlite3, time
from collections import defaultdict

from .db import MemoryDB


def _get_id(c: sqlite3.Connection, table: str, name: str) -> int:
    row = c.execute(f"SELECT id FROM {table} WHERE name=?", (name,)).fetchone()
    if row:
        return int(row[0])
    cur = c.execute(f"INSERT INTO {table} (name) VALUES (?)", (name,))
    return cur.lastrowid

def _merge_sources(*sources: Optional[str]) -> Optional[str]:
    values = []
    for src in sources:
        if not src:
            continue
        for item in str(src).split("|"):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return " | ".join(values) if values else None


def add_claim(db: MemoryDB, subject: str, relation: str, obj: str, importance: float = 0.5, source_url: Optional[str] = None) -> Tuple[int, str]:
    obj_clean = obj.strip()
    now = int(time.time())
    with db._conn() as c:
        sid = _get_id(c, 'entities', subject.strip())
        rid = _get_id(c, 'relations', relation.strip())
        rows = c.execute(
            "SELECT id, object, importance, source_url FROM claims WHERE subject_id=? AND relation_id=?",
            (sid, rid),
        ).fetchall()

        if not rows:
            cur = c.execute(
                "INSERT INTO claims (subject_id, relation_id, object, importance, source_url, updated_at) VALUES (?,?,?,?,?,?)",
                (sid, rid, obj_clean, float(importance), source_url or None, now),
            )
            return cur.lastrowid, "inserted"

        totals = defaultdict(float)
        existing_map = {}
        sources_map = defaultdict(list)
        for claim_id, claim_obj, claim_imp, claim_src in rows:
            totals[claim_obj] += float(claim_imp or 0.0)
            existing_map[claim_obj] = (claim_id, float(claim_imp or 0.0), claim_src)
            if claim_src:
                sources_map[claim_obj].append(claim_src)

        totals[obj_clean] += float(importance)
        if source_url:
            sources_map[obj_clean].append(source_url)

        winner_obj = max(totals.items(), key=lambda kv: kv[1])[0]
        outcome = "merged" if winner_obj == obj_clean else "voted"

        # Strengthen winner
        if winner_obj in existing_map:
            win_id, _, win_src = existing_map[winner_obj]
            merged_source = _merge_sources(win_src, *(sources_map[winner_obj] or []))
            c.execute(
                "UPDATE claims SET importance=?, source_url=?, updated_at=? WHERE id=?",
                (round(totals[winner_obj], 6), merged_source, now, win_id),
            )
            winner_id = win_id
        else:
            cur = c.execute(
                "INSERT INTO claims (subject_id, relation_id, object, importance, source_url, updated_at) VALUES (?,?,?,?,?,?)",
                (sid, rid, winner_obj, round(totals[winner_obj], 6), _merge_sources(*sources_map[winner_obj]), now),
            )
            winner_id = cur.lastrowid

        # Update / insert challenger record if distinct
        if winner_obj != obj_clean:
            challenger = existing_map.get(obj_clean)
            if challenger:
                c.execute(
                    "UPDATE claims SET importance=?, source_url=?, updated_at=? WHERE id=?",
                    (round(max(challenger[1] * 0.7, 0.05), 6), _merge_sources(challenger[2], source_url), now, challenger[0]),
                )
            else:
                c.execute(
                    "INSERT INTO claims (subject_id, relation_id, object, importance, source_url, updated_at) VALUES (?,?,?,?,?,?)",
                    (sid, rid, obj_clean, round(max(float(importance) * 0.7, 0.05), 6), source_url or None, now),
                )

        # Nudge non-winning claims downward to reflect new vote
        for claim_id, claim_obj, claim_imp, _ in rows:
            if claim_obj == winner_obj:
                continue
            new_imp = round(max(claim_imp * 0.9, 0.05), 6)
            c.execute("UPDATE claims SET importance=?, updated_at=? WHERE id=?", (new_imp, now, claim_id))

        return winner_id, outcome


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


def adjust_claim_importance(db: MemoryDB, claim_id: int, delta: float) -> Optional[float]:
    now = int(time.time())
    with db._conn() as c:
        row = c.execute("SELECT importance FROM claims WHERE id=?", (claim_id,)).fetchone()
        if not row:
            return None
        new_imp = max(0.0, min(1.0, float(row[0] or 0.0) + float(delta)))
        c.execute("UPDATE claims SET importance=?, updated_at=? WHERE id=?", (round(new_imp, 6), now, claim_id))
        return new_imp


def delete_claim(db: MemoryDB, claim_id: int) -> None:
    with db._conn() as c:
        c.execute("DELETE FROM claims WHERE id=?", (claim_id,))
