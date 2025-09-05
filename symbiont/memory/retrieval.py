from __future__ import annotations
import json
from .embedder import cheap_embed, cosine

def build_indices(db, limit_if_new=None) -> int:
    count=0
    with db._conn() as c:
        msgs = c.execute("SELECT id, role, content FROM messages ORDER BY id ASC").fetchall()
        arts = c.execute("SELECT id, type, path, summary FROM artifacts ORDER BY id ASC").fetchall()
    for mid, role, content in msgs[: (limit_if_new or len(msgs)) ]:
        vec = cheap_embed([f"{role}: {content}"])[0]
        _ins(db,"message","messages",mid,vec); count+=1
    for aid, typ, path, summary in arts[: (limit_if_new or len(arts)) ]:
        text = summary or path or ""
        if text.strip():
            vec = cheap_embed([text])[0]; _ins(db,"artifact","artifacts",aid,vec); count+=1
    return count

def _ins(db, kind, table, ref_id, vec):
    with db._conn() as c:
        c.execute("INSERT INTO vectors (kind, ref_table, ref_id, embedding) VALUES (?, ?, ?, ?)", (kind, table, ref_id, json.dumps(vec)))

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
