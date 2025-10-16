from __future__ import annotations

from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed

from ..memory.db import MemoryDB


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def _ensure_schema(db_path: str) -> None:
    MemoryDB(db_path=db_path).ensure_schema()


def init_memory_db(db_path: str) -> Path:
    """
    Ensure the SQLite database exists with the expected schema.

    Retries a handful of times to tolerate transient locks that can happen on
    slower filesystems during first-run bootstrap.
    """
    target = Path(db_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_schema(str(target))
    return target
