"""Lightweight pub/sub helper for initiative events.

The implementation prefers Redis when configured but gracefully falls back
to a local JSONL log file so unit tests and edge devices without Redis can
still observe what the daemon is doing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
    retry_attempts: int = 3
    retry_initial_delay: float = 0.5
    retry_max_delay: float = 10.0
    retry_multiplier: float = 2.0


class PubSubClient:
    """Publish initiative events via Redis or to a local log."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, *, data_root: Path | None = None):
        parsed = PubSubConfig(**((cfg or {})))
        self.enabled = bool(parsed.enabled)
        self.backend = (parsed.backend or "memory").lower()
        self.channel = parsed.channel or "symbiont:initiative"
        self._redis = None
        self._log_path: Optional[Path] = None
        
        # Retry configuration
        self.retry_attempts = max(1, int(parsed.retry_attempts))
        self.retry_initial_delay = max(0.1, float(parsed.retry_initial_delay))
        self.retry_max_delay = max(self.retry_initial_delay, float(parsed.retry_max_delay))
        self.retry_multiplier = max(1.0, float(parsed.retry_multiplier))

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
            self._publish_with_retry(self._publish_redis, payload)
        elif self._log_path is not None:
            self._publish_with_retry(self._publish_log, payload)

    def _publish_redis(self, payload: str) -> None:
        """Publish to Redis without retry logic."""
        self._redis.publish(self.channel, payload)

    def _publish_log(self, payload: str) -> None:
        """Publish to log file without retry logic."""
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    def _publish_with_retry(self, publish_func, payload: str) -> None:
        """Execute publish function with exponential backoff retry."""
        if self.retry_attempts <= 1:
            try:
                publish_func(payload)
            except Exception as exc:
                logger.warning("Failed to publish initiative event: %s", exc)
            return

        def _log_retry_warning(retry_state: RetryCallState) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc:
                logger.warning(
                    "PubSub retry %s/%s failed: %s",
                    retry_state.attempt_number,
                    self.retry_attempts,
                    exc,
                )

        retryer = Retrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(
                multiplier=self.retry_initial_delay,
                exp_base=self.retry_multiplier,
                min=self.retry_initial_delay,
                max=self.retry_max_delay,
            ),
            reraise=False,  # Don't re-raise, just log the final failure
            before_sleep=_log_retry_warning,
            sleep=time.sleep,
        )
        
        try:
            retryer(publish_func, payload)
        except Exception as exc:
            logger.warning("Failed to publish initiative event after %s attempts: %s", self.retry_attempts, exc)


def get_client(cfg: Dict[str, Any], *, data_root: Path | None = None) -> PubSubClient:
    initiative_cfg = (cfg.get("initiative") or {})
    pub_cfg = initiative_cfg.get("pubsub") or {}
    return PubSubClient(pub_cfg, data_root=data_root)

