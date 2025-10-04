"""Lightweight pub/sub helper for initiative events.

The implementation prefers Redis when configured but gracefully falls back
to a local JSONL log file so unit tests and edge devices without Redis can
still observe what the daemon is doing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..tools.files import ensure_dirs


logger = logging.getLogger(__name__)


@dataclass
class PubSubConfig:
    enabled: bool = False
    backend: str = "memory"  # "memory" | "redis"
    channel: str = "symbiont:initiative"
    log_path: Optional[str] = None
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0


class PubSubClient:
    """Publish initiative events via Redis or to a local log."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, *, data_root: Path | None = None):
        parsed = PubSubConfig(**((cfg or {})))
        self.enabled = bool(parsed.enabled)
        self.backend = (parsed.backend or "memory").lower()
        self.channel = parsed.channel or "symbiont:initiative"
        self._redis = None
        self._log_path: Optional[Path] = None

        if not self.enabled:
            return

        if self.backend == "redis":
            try:
                import redis  # type: ignore

                if parsed.url:
                    self._redis = redis.from_url(parsed.url)
                else:
                    self._redis = redis.Redis(host=parsed.host, port=parsed.port, db=parsed.db)
                # Validate the connection early to avoid repeating failures later.
                self._redis.ping()
            except Exception as exc:  # pragma: no cover - requires Redis at runtime
                logger.warning("Redis unavailable for initiative pubsub: %s", exc)
                self._redis = None
                self.backend = "memory"

        if self.backend != "redis":
            root = Path(parsed.log_path) if parsed.log_path else (data_root or Path("data"))
            if root.is_dir():
                log_path = root / "initiative" / "events.jsonl"
            else:
                log_path = root
            ensure_dirs([log_path.parent])
            self._log_path = log_path

    def publish(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        payload = json.dumps(event, separators=(",", ":"))

        if self.backend == "redis" and self._redis is not None:
            try:
                self._redis.publish(self.channel, payload)
            except Exception as exc:  # pragma: no cover - depends on Redis availability
                logger.warning("Failed to publish initiative event to Redis: %s", exc)
        elif self._log_path is not None:
            try:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(payload + "\n")
            except Exception as exc:  # pragma: no cover - disk errors are environment specific
                logger.warning("Failed to append initiative event log: %s", exc)


def get_client(cfg: Dict[str, Any], *, data_root: Path | None = None) -> PubSubClient:
    initiative_cfg = (cfg.get("initiative") or {})
    pub_cfg = initiative_cfg.get("pubsub") or {}
    return PubSubClient(pub_cfg, data_root=data_root)

