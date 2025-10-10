from __future__ import annotations
import json
from typing import Iterable, Set, Optional, Dict, Any
from .embedder import cheap_embed, cosine
from .db import MemoryDB
from .external_sources import (
    ExternalSourceFetcher,
    fetch_and_store_external_context,
)

def build_indices(db, limit_if_new=None) -> int:
    """Embed only new messages/artifacts and upsert their vectors."""
    with db._conn() as conn:
        msgs = conn.execute("SELECT id, role, content FROM messages ORDER BY id ASC").fetchall()
        arts = conn.execute("SELECT id, type, path, summary FROM artifacts ORDER BY id ASC").fetchall()
        existing_msg_ids = _existing_ids(conn, "message", "messages")
        existing_art_ids = _existing_ids(conn, "artifact", "artifacts")

    new_msgs = [(mid, role, content) for mid, role, content in msgs if mid not in existing_msg_ids]
    new_arts = [(aid, typ, path, summary) for aid, typ, path, summary in arts if aid not in existing_art_ids]

    if limit_if_new:
        new_msgs = new_msgs[-int(limit_if_new):]
        new_arts = new_arts[-int(limit_if_new):]

    count = 0
    if new_msgs or new_arts:
        with db._conn() as conn:
            for mid, role, content in new_msgs:
                vec = cheap_embed([f"{role}: {content}"])[0]
                _upsert(conn, "message", "messages", mid, vec)
                count += 1
            for aid, typ, path, summary in new_arts:
                text = (summary or path or "").strip()
                if not text:
                    continue
                vec = cheap_embed([text])[0]
                _upsert(conn, "artifact", "artifacts", aid, vec)
                count += 1
    return count

def _existing_ids(conn, kind: str, table: str) -> Set[int]:
    try:
        rows = conn.execute("SELECT ref_id FROM vectors WHERE kind=? AND ref_table=?", (kind, table)).fetchall()
    except Exception:
        return set()
    return {int(r[0]) for r in rows}

def _upsert(conn, kind: str, table: str, ref_id: int, vec: Iterable[float]) -> None:
    payload = json.dumps(list(vec))
    conn.execute(
        """
        INSERT INTO vectors (kind, ref_table, ref_id, embedding)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(kind, ref_table, ref_id) DO UPDATE SET embedding=excluded.embedding
        """,
        (kind, table, ref_id, payload),
    )

def search(db, query: str, k: int = 5):
    qv = cheap_embed([query])[0]
    with db._conn() as c:
        rows = c.execute("SELECT id, kind, ref_table, ref_id, embedding FROM vectors").fetchall()
    scored=[]
    for _id, kind, table, ref_id, emb in rows:
        try: ev = json.loads(emb)
        except Exception: continue
        sc = cosine(qv, ev)
        prev = _preview(db, table, ref_id)
        scored.append({"id":_id,"kind":kind,"ref_table":table,"ref_id":ref_id,"score":sc,"preview":prev})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

def _preview(db, table, _id):
    with db._conn() as c:
        if table=="messages":
            row=c.execute("SELECT role, content FROM messages WHERE id=?",(_id,)).fetchone()
            if row: return f"{row[0]}→{row[1][:140]}"
        if table=="artifacts":
            row=c.execute("SELECT summary, path FROM artifacts WHERE id=?",(_id,)).fetchone()
            if row: return (row[0] or row[1] or "")[:140]
    return ""


def fetch_external_context(
    db: MemoryDB,
    query: str,
    *,
    max_items: int = 8,
    min_relevance: float = 0.7,
    fetcher: Optional[ExternalSourceFetcher] = None,
) -> Dict[str, Any]:
    """Public helper that fetches and merges external research into GraphRAG."""

    return fetch_and_store_external_context(
        db,
        query,
        max_items=max_items,
        min_relevance=min_relevance,
        fetcher=fetcher,
    )
