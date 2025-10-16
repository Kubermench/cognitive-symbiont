from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .db import MemoryDB
from .embedder import cheap_embed, cosine
from .external_sources import ExternalSourceFetcher, fetch_and_store_external_context

logger = logging.getLogger(__name__)


class MemoryBackendError(RuntimeError):
    """Raised when a memory backend cannot be initialised."""


class MemoryBackend(Protocol):
    """Common interface for memory backends used by Symbiont."""

    name: str

    def build_indices(self, *, limit_if_new: Optional[int] = None) -> int:
        """Embed new content and store indexes for retrieval."""

    def search(self, query: str, *, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the k most relevant memory records."""

    def fetch_external_context(
        self,
        query: str,
        *,
        max_items: int = 8,
        min_relevance: float = 0.7,
        fetcher: Optional[ExternalSourceFetcher] = None,
    ) -> Dict[str, Any]:
        """Fetch and persist external research context."""

    # Optional advanced capabilities -------------------------------------------------
    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """Persist agent state across sessions."""

    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        """Load previously persisted agent state."""

    def remember_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Persist user-level preferences."""

    def recall_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user-level preferences."""


class GraphMemoryBackend:
    """Default lightweight GraphRAG + SQLite backend."""

    name = "local"

    def __init__(self, db: MemoryDB):
        self.db = db

    # --------------------------------------------------------------------- retrieval
    def build_indices(self, *, limit_if_new: Optional[int] = None) -> int:
        with self.db._conn() as conn:
            last_msg_marker = _load_marker(conn, "vectors:last_message_id")
            last_art_marker = _load_marker(conn, "vectors:last_artifact_id")
            missing_msgs, fresh_msgs = _pending_rows(
                conn,
                table="messages",
                column_list="id, role, content",
                kind="message",
                last_marker=last_msg_marker,
            )
            missing_arts, fresh_arts = _pending_rows(
                conn,
                table="artifacts",
                column_list="id, type, path, summary",
                kind="artifact",
                last_marker=last_art_marker,
            )

        cap = 0
        if limit_if_new is not None:
            try:
                cap = max(0, int(limit_if_new))
            except (TypeError, ValueError):
                cap = 0
        if cap:
            fresh_msgs = fresh_msgs[-cap:]
            fresh_arts = fresh_arts[-cap:]
        pending_msgs = _merge_rows(missing_msgs, fresh_msgs)
        pending_arts = _merge_rows(missing_arts, fresh_arts)
        if cap:
            pending_msgs = pending_msgs[-cap:]
            pending_arts = pending_arts[-cap:]

        count = 0
        msg_marker = last_msg_marker
        art_marker = last_art_marker
        if pending_msgs or pending_arts:
            with self.db._conn() as conn:
                for mid, role, content in pending_msgs:
                    vec = cheap_embed([f"{role}: {content}"])[0]
                    _upsert(conn, "message", "messages", mid, vec)
                    count += 1
                    if mid > msg_marker:
                        msg_marker = mid
                for aid, typ, path, summary in pending_arts:
                    text = (summary or path or "").strip()
                    if not text:
                        continue
                    vec = cheap_embed([text])[0]
                    _upsert(conn, "artifact", "artifacts", aid, vec)
                    count += 1
                    if aid > art_marker:
                        art_marker = aid
                if msg_marker != last_msg_marker:
                    _store_marker(conn, "vectors:last_message_id", msg_marker)
                if art_marker != last_art_marker:
                    _store_marker(conn, "vectors:last_artifact_id", art_marker)
        return count

    def search(self, query: str, *, k: int = 5) -> List[Dict[str, Any]]:
        qv = cheap_embed([query])[0]
        with self.db._conn() as conn:
            rows = conn.execute(
                "SELECT id, kind, ref_table, ref_id, embedding FROM vectors"
            ).fetchall()
        scored = []
        for _id, kind, table, ref_id, emb in rows:
            try:
                ev = json.loads(emb)
            except Exception:
                continue
            sc = cosine(qv, ev)
            prev = _preview(self.db, table, ref_id)
            scored.append(
                {
                    "id": _id,
                    "kind": kind,
                    "ref_table": table,
                    "ref_id": ref_id,
                    "score": sc,
                    "preview": prev,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def fetch_external_context(
        self,
        query: str,
        *,
        max_items: int = 8,
        min_relevance: float = 0.7,
        fetcher: Optional[ExternalSourceFetcher] = None,
    ) -> Dict[str, Any]:
        return fetch_and_store_external_context(
            self.db,
            query,
            max_items=max_items,
            min_relevance=min_relevance,
            fetcher=fetcher,
        )

    # ------------------------------------------------------------------------ stubs
    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        _persist_locally(self.db, "session_state", session_id, state)

    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        return _load_local(self.db, "session_state", session_id)

    def remember_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        _persist_locally(self.db, "preferences", user_id, preferences)

    def recall_preferences(self, user_id: str) -> Dict[str, Any]:
        return _load_local(self.db, "preferences", user_id)


class Mem0Backend(GraphMemoryBackend):
    """Optional Mem0-backed personalization layer."""

    name = "mem0"

    def __init__(self, db: MemoryDB, options: Optional[Dict[str, Any]] = None):
        super().__init__(db)
        self._options = options or {}
        self._mem0 = _Mem0Adapter(self._options)

    def _pref_scope(self, user_id: str) -> Optional[str]:
        return _select_option(self._options, "preferences_user_id", [self._options.get("preferences_user_id_env"), "MEM0_USER_ID"]) or user_id

    def _session_scope(self, session_id: str) -> Optional[str]:
        return _select_option(self._options, "session_user_id", [self._options.get("session_user_id_env"), "MEM0_SESSION_USER"]) or session_id

    def remember_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        if self._mem0.is_ready():
            try:
                self._mem0.upsert(
                    kind="preferences",
                    key=user_id,
                    payload=preferences,
                    user_id=self._pref_scope(user_id),
                )
            except Exception as exc:
                logger.warning("Mem0 preference sync failed: %s", exc)
        super().remember_preferences(user_id, preferences)

    def recall_preferences(self, user_id: str) -> Dict[str, Any]:
        if self._mem0.is_ready():
            try:
                data = self._mem0.load(
                    kind="preferences",
                    key=user_id,
                    user_id=self._pref_scope(user_id),
                )
                if data is not None:
                    return data
            except Exception as exc:
                logger.warning("Mem0 preference load failed: %s", exc)
        return super().recall_preferences(user_id)

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        if self._mem0.is_ready():
            try:
                self._mem0.upsert(
                    kind="session_state",
                    key=session_id,
                    payload=state,
                    user_id=self._session_scope(session_id),
                )
            except Exception as exc:
                logger.warning("Mem0 session sync failed: %s", exc)
        super().save_session_state(session_id, state)

    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        if self._mem0.is_ready():
            try:
                data = self._mem0.load(
                    kind="session_state",
                    key=session_id,
                    user_id=self._session_scope(session_id),
                )
                if data is not None:
                    return data
            except Exception as exc:
                logger.warning("Mem0 session load failed: %s", exc)
        return super().load_session_state(session_id)


class LettaBackend(GraphMemoryBackend):
    """Optional Letta-backed hierarchical session memory."""

    name = "letta"

    def __init__(self, db: MemoryDB, options: Optional[Dict[str, Any]] = None):
        super().__init__(db)
        self._options = options or {}
        self._letta = _LettaAdapter(self._options)

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        if self._letta.is_ready():
            try:
                self._letta.save_session_state(session_id, state)
            except Exception as exc:
                logger.warning("Letta session sync failed: %s", exc)
        super().save_session_state(session_id, state)

    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        if self._letta.is_ready():
            try:
                data = self._letta.load_session_state(session_id)
                if data is not None:
                    return data
            except Exception as exc:
                logger.warning("Letta session load failed: %s", exc)
        return super().load_session_state(session_id)

    def remember_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        if self._letta.is_ready():
            try:
                self._letta.save_preferences(user_id, preferences)
            except Exception as exc:
                logger.warning("Letta preference sync failed: %s", exc)
        super().remember_preferences(user_id, preferences)

    def recall_preferences(self, user_id: str) -> Dict[str, Any]:
        if self._letta.is_ready():
            try:
                data = self._letta.load_preferences(user_id)
                if data is not None:
                    return data
            except Exception as exc:
                logger.warning("Letta preference load failed: %s", exc)
        return super().recall_preferences(user_id)


# ---------------------------------------------------------------------------- utils

def _pending_rows(
    conn,
    *,
    table: str,
    column_list: str,
    kind: str,
    last_marker: int,
):
    params: Tuple[Any, ...] = ()
    query = f"SELECT {column_list} FROM {table}"
    if last_marker:
        query += " WHERE id > ?"
        params = (last_marker,)
    query += " ORDER BY id ASC"
    fresh = conn.execute(query, params).fetchall()

    missing = []
    if last_marker:
        missing = conn.execute(
            f"""
            SELECT {column_list}
            FROM {table} src
            WHERE src.id <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM vectors v
                  WHERE v.kind=? AND v.ref_table=? AND v.ref_id = src.id
              )
            ORDER BY src.id ASC
            """,
            (last_marker, kind, table),
        ).fetchall()
    return missing, fresh


def _merge_rows(*groups: Iterable[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    combined: Dict[int, Tuple[Any, ...]] = {}
    for group in groups:
        for row in group or []:
            if not row:
                continue
            try:
                key = int(row[0])
            except (TypeError, ValueError):
                continue
            combined[key] = row
    return [combined[key] for key in sorted(combined)]


def _load_marker(conn, key: str) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key=?",
            (key,),
        ).fetchone()
    except Exception:
        return 0
    if not row or row[0] is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def _store_marker(conn, key: str, value: int) -> None:
    conn.execute(
        """
        INSERT INTO schema_meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, str(int(value))),
    )


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


def _preview(db: MemoryDB, table: str, ref_id: int) -> str:
    with db._conn() as conn:
        if table == "messages":
            row = conn.execute(
                "SELECT role, content FROM messages WHERE id=?",
                (ref_id,),
            ).fetchone()
            if row:
                return f"{row[0]}→{row[1][:140]}"
        if table == "artifacts":
            row = conn.execute(
                "SELECT summary, path FROM artifacts WHERE id=?",
                (ref_id,),
            ).fetchone()
            if row:
                return (row[0] or row[1] or "")[:140]
    return ""


def _persist_locally(db: MemoryDB, namespace: str, key: str, payload: Dict[str, Any]) -> None:
    root = Path(db.db_path).parent / "memory_layers" / namespace
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{key}.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _load_local(db: MemoryDB, namespace: str, key: str) -> Dict[str, Any]:
    root = Path(db.db_path).parent / "memory_layers" / namespace
    target = root / f"{key}.json"
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def _select_option(options: Optional[Dict[str, Any]], key: str, env_candidates: Iterable[Optional[str]]) -> Optional[str]:
    value = None
    if isinstance(options, dict):
        raw = options.get(key)
        if isinstance(raw, str):
            value = raw.strip() or None
        elif raw is not None:
            value = str(raw)
    if value:
        return value
    for env_key in env_candidates:
        if env_key and (env_val := os.getenv(env_key)):
            return env_val.strip() or None
    return None


def _as_dict(candidate: Any) -> Dict[str, Any]:
    if candidate is None:
        return {}
    if hasattr(candidate, "model_dump"):
        try:
            return candidate.model_dump()
        except Exception:
            pass
    if isinstance(candidate, dict):
        return candidate
    return {}


def _extract_metadata(entry: Any) -> Dict[str, Any]:
    data = _as_dict(entry)
    if not data:
        return {}
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    sub = data.get("memory")
    if isinstance(sub, dict):
        return _extract_metadata(sub)
    return {}


def _extract_text(entry: Any) -> Optional[str]:
    data = _as_dict(entry)
    if not data:
        return None
    for key in ("text", "content", "value"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            for key in ("content", "text", "value"):
                val = message.get(key)
                if isinstance(val, str) and val:
                    return val
    nested = data.get("memory") or data.get("result")
    if isinstance(nested, dict):
        return _extract_text(nested)
    return None


def _extract_id(entry: Any) -> Optional[str]:
    data = _as_dict(entry)
    if not data:
        return None
    for key in ("id", "memory_id", "uuid", "memoryId"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    nested = data.get("memory")
    if isinstance(nested, dict):
        return _extract_id(nested)
    return None


def _extract_timestamp(entry: Any) -> float:
    data = _as_dict(entry)
    if not data:
        return 0.0
    for key in ("updated_at", "created_at", "timestamp", "updatedAt", "createdAt"):
        val = data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and val:
            try:
                return float(val)
            except ValueError:
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
    return 0.0


def _backend_config(config: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    section = config.get(key)
    return section if isinstance(section, dict) else {}


@dataclass
class _Mem0Adapter:
    options: Dict[str, Any]
    _client: Any = None
    _last_error: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            from mem0.client.main import MemoryClient  # type: ignore
        except Exception as exc:
            self._last_error = str(exc)
            return

        api_key = _select_option(self.options, "api_key", [self.options.get("api_key_env"), "MEM0_API_KEY", "MEM0_TOKEN"])
        host = _select_option(self.options, "host", [self.options.get("host_env"), "MEM0_API_HOST"])
        org_id = _select_option(self.options, "org_id", [self.options.get("org_id_env"), "MEM0_ORG_ID"])
        project_id = _select_option(self.options, "project_id", [self.options.get("project_id_env"), "MEM0_PROJECT_ID"])

        if not api_key:
            self._last_error = "Mem0 API key missing"
            logger.debug("Mem0 adapter disabled: %s", self._last_error)
            return

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if host:
            kwargs["host"] = host
        if org_id:
            kwargs["org_id"] = org_id
        if project_id:
            kwargs["project_id"] = project_id

        try:
            self._client = MemoryClient(**kwargs)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Failed to initialise Mem0 client: %s", exc)
            self._client = None

    def is_ready(self) -> bool:
        return self._client is not None

    def upsert(self, *, kind: str, key: str, payload: Dict[str, Any], user_id: Optional[str]) -> None:
        if not self._client:
            raise RuntimeError("Mem0 client unavailable")
        metadata = {
            "symbiont_kind": kind,
            "symbiont_key": key,
        }
        if user_id:
            metadata["symbiont_user_id"] = user_id
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        record = self._find_latest(metadata, user_id=user_id)
        if record and (memory_id := _extract_id(record)):
            self._client.update(memory_id=memory_id, text=text, metadata=metadata)
        else:
            messages = [{"role": "system", "content": text}]
            params: Dict[str, Any] = {"metadata": metadata}
            if user_id:
                params["user_id"] = user_id
            self._client.add(messages, **params)

    def load(self, *, kind: str, key: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("Mem0 client unavailable")
        metadata = {
            "symbiont_kind": kind,
            "symbiont_key": key,
        }
        records = self._find_latest(metadata, user_id=user_id, multiple=True)
        if not records:
            return None
        latest = records[-1]
        raw = _extract_text(latest)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            logger.debug("Mem0 payload for %s/%s is not JSON; returning raw string.", kind, key)
            return {"raw": raw}

    def _find_latest(self, metadata: Dict[str, Any], *, user_id: Optional[str], multiple: bool = False) -> Optional[Any]:
        params: Dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        params_with_meta = dict(params)
        params_with_meta["metadata"] = metadata
        records: List[Any] = []
        try:
            records = self._client.get_all(version="v2", **params_with_meta)
        except TypeError:
            # Older deployments may not accept metadata in query params; fall back to manual filtering.
            records = self._client.get_all(version="v2", **params)
        except Exception as exc:
            logger.warning("Mem0 fetch failed: %s", exc)
            return None
        if not isinstance(records, list):
            return None
        filtered = [
            record
            for record in records
            if all(_extract_metadata(record).get(k) == v for k, v in metadata.items())
        ]
        if not filtered:
            return None
        filtered.sort(key=_extract_timestamp)
        if multiple:
            return filtered
        return filtered[-1]


@dataclass
class _LettaAdapter:
    options: Dict[str, Any]
    _client: Any = None
    _last_error: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            from letta_client import Letta  # type: ignore
        except Exception as exc:
            self._last_error = str(exc)
            logger.debug("Letta adapter disabled: %s", exc)
            return

        token = _select_option(self.options, "token", [self.options.get("token_env"), "LETTA_API_TOKEN", "LETTA_TOKEN"])
        project = _select_option(self.options, "project", [self.options.get("project_env"), "LETTA_PROJECT"])
        base_url = _select_option(self.options, "base_url", [self.options.get("base_url_env"), "LETTA_BASE_URL"])
        self._session_label = self.options.get("session_label") or "symbiont_session_state"
        self._preference_label = self.options.get("preference_label") or "symbiont_user_preferences"
        self._user_id = _select_option(self.options, "user_id", [self.options.get("user_id_env"), "LETTA_USER_ID"])

        if not token:
            self._last_error = "Letta token missing"
            logger.debug("Letta adapter disabled: %s", self._last_error)
            return

        kwargs: Dict[str, Any] = {"token": token}
        if project:
            kwargs["project"] = project
        if base_url:
            kwargs["base_url"] = base_url

        try:
            self._client = Letta(**kwargs)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Failed to initialise Letta client: %s", exc)
            self._client = None

    def is_ready(self) -> bool:
        return self._client is not None

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        self._save_block(self._session_label, session_id, state)

    def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._load_block(self._session_label, session_id)

    def save_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self._save_block(self._preference_label, user_id, preferences)

    def load_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._load_block(self._preference_label, user_id)

    def _save_block(self, label: str, key: str, payload: Dict[str, Any]) -> None:
        if not self._client:
            raise RuntimeError("Letta client unavailable")
        metadata = {
            "symbiont_label": label,
            "symbiont_key": key,
        }
        if self._user_id:
            metadata["symbiont_user_id"] = self._user_id
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        block = self._find_block(label, key)
        if block and getattr(block, "id", None):
            self._client.blocks.modify(
                block.id,
                value=serialized,
                metadata=metadata,
                label=label,
            )
        else:
            name = f"{label}-{key}"
            self._client.blocks.create(
                value=serialized,
                label=label,
                metadata=metadata,
                name=name[:255],
                hidden=True,
            )

    def _load_block(self, label: str, key: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            raise RuntimeError("Letta client unavailable")
        block = self._find_block(label, key)
        if not block:
            return None
        value = getattr(block, "value", None)
        if not isinstance(value, str) or not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            logger.debug("Letta block %s/%s payload is not JSON; returning raw string.", label, key)
            return {"raw": value}

    def _find_block(self, label: str, key: str):
        if not self._client:
            raise RuntimeError("Letta client unavailable")
        try:
            blocks = self._client.blocks.list(label=label, limit=100)
        except Exception as exc:
            logger.warning("Letta block list failed: %s", exc)
            return None
        for block in blocks or []:
            meta = getattr(block, "metadata", None) or {}
            if isinstance(meta, dict) and meta.get("symbiont_key") == key:
                return block
        return None


def resolve_backend(name: Optional[str], db: MemoryDB, config: Optional[Dict[str, Any]] = None) -> MemoryBackend:
    """Return a configured backend based on name/environment."""

    normalized = (name or os.getenv("SYMBIONT_MEMORY_LAYER") or "local").lower()
    backend_config = config or {}
    if normalized in {"local", "graph", "graphrag", "sqlite"}:
        return GraphMemoryBackend(db)
    if normalized in {"mem0", "mem-0"}:
        return Mem0Backend(db, options=_backend_config(backend_config, "mem0"))
    if normalized == "letta":
        return LettaBackend(db, options=_backend_config(backend_config, "letta"))
    raise MemoryBackendError(f"Unknown memory backend '{name}'")


def available_backends() -> Tuple[str, ...]:
    """Return a tuple of backend identifiers that can be requested."""
    options = ["local", "mem0", "letta"]
    return tuple(options)


def coerce_backend_name(
    config: Optional[Dict[str, Any]] = None,
    override: Optional[str] = None,
) -> Optional[str]:
    """Extract a backend identifier from config/overrides."""

    if override:
        return str(override)
    cfg = config or {}
    if isinstance(cfg, dict):
        memory_section = cfg.get("memory")
        if isinstance(memory_section, dict):
            candidate = memory_section.get("layer") or memory_section.get("backend")
            if candidate:
                return str(candidate)
        legacy = cfg.get("memory_layer")
        if legacy:
            return str(legacy)
    return None
