"""Async foresight hunt orchestration with defensive fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from symbiont.llm.client import LLMClient
from symbiont.tools import rate_limiter, research, secrets
from symbiont.tools.files import ensure_dirs

LOGGER = logging.getLogger(__name__)

OFFLINE_MOCK = Path("data/foresight/offline_sources.json")
HUNT_DIR = Path("data/artifacts/foresight/hunts")
METRIC_PATH = Path("data/artifacts/metrics/foresight_metrics.json")
STATE_PATH = Path("data/foresight/state.json")
UTC = timezone.utc


@dataclass
class HuntConfig:
    """Runtime knobs for foresight hunts."""

    offline: bool = False
    max_items: int = 6
    include_collaborators: bool = True
    include_rss: bool = True
    credential_spec: Dict[str, Any] | None = None
    credential_rotation_hours: int = 24
    jot_metrics: bool = True
    sources: Sequence[str] = ("arxiv", "rss", "peer", "web")
    cache_ttl_minutes: int = 45

    def artifact_dir(self) -> Path:
        ensure_dirs([HUNT_DIR])
        return HUNT_DIR


class CredentialRotator:
    """Persist simple rotation timestamps so API keys are refreshed regularly."""

    def __init__(self, spec: Dict[str, Any] | None, hours: int):
        self._spec = spec
        self._ttl = timedelta(hours=max(1, hours))
        self._state_path = STATE_PATH

    def rotate_if_due(self) -> Optional[str]:
        if not self._spec:
            return None
        now = datetime.now(UTC)
        state = self._load_state()
        last_iso = state.get("last_rotation")
        if last_iso:
            with suppress(Exception):
                last = datetime.fromisoformat(last_iso)
                if (now - last) < self._ttl:
                    return state.get("credential")
        try:
            credential = secrets.load_secret(self._spec, fallback_env="ARXIV_APP_ID")
        except Exception as exc:  # pragma: no cover - host dependent
            LOGGER.debug("Credential rotation failed: %s", exc)
            return state.get("credential")
        state.update({"last_rotation": now.isoformat(), "credential": credential})
        ensure_dirs([self._state_path.parent])
        self._state_path.write_text(json.dumps(state), encoding="utf-8")
        return credential

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {}
        with suppress(Exception):
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        return {}


def _load_offline_sources() -> Dict[str, Any]:
    if OFFLINE_MOCK.exists():
        with suppress(Exception):
            data = json.loads(OFFLINE_MOCK.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    return {
        "topic": "offline foresight seed",
        "items": [
            {
                "title": "Edge-safe agent orchestration",
                "summary": "Fallback insight cached for offline foresight hunts.",
                "source": "cache",
                "url": "https://example.com/offline",
                "published": datetime.now(UTC).strftime("%Y-%m-%d"),
            }
        ],
        "meta": {"offline": True, "contributors": ["cache"]},
    }


def _merge_items(*payloads: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"contributors": []}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        source = payload.get("source") or payload.get("topic") or "unknown"
        if isinstance(payload.get("items"), Iterable):
            for item in payload["items"]:  # type: ignore[index]
                if isinstance(item, dict):
                    items.append(dict(item))
        if isinstance(payload.get("meta"), dict):
            for key, value in payload["meta"].items():  # type: ignore[attr-defined]
                if key == "contributors" and isinstance(value, Iterable):
                    meta.setdefault("contributors", [])
                    for entry in value:
                        if entry not in meta["contributors"]:
                            meta["contributors"].append(entry)
                else:
                    meta[key] = value
        else:
            meta.setdefault("contributors", []).append(str(source))
    return items, meta


async def _async_gather_sources(llm: LLMClient, query: str, cfg: HuntConfig) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    tasks = [
        research.gather_trend_sources_async(
            llm,
            query,
            max_items=cfg.max_items,
            include_collaborators=cfg.include_collaborators,
            include_rss=cfg.include_rss,
        ),
        loop.run_in_executor(None, research.scout_insights, llm, query, cfg.max_items),
    ]

    if "rss" in cfg.sources and not cfg.offline:
        tasks.append(loop.run_in_executor(None, research.fetch_rss_alerts, query, cfg.max_items))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    merged: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"contributors": []}
    for result in results:
        if isinstance(result, Exception):
            LOGGER.debug("Hunt adaptor failed: %s", result)
            continue
        payload_items, payload_meta = _merge_items(result if isinstance(result, dict) else {})
        merged.extend(payload_items)
        for key, value in payload_meta.items():
            if key == "contributors":
                for contributor in value:
                    if contributor not in meta["contributors"]:
                        meta["contributors"].append(contributor)
            else:
                meta[key] = value
    return {"topic": query, "items": merged, "meta": meta}


class RateLimitError(RuntimeError):
    """Wrapper to trigger tenacity retries when rate limits fire."""


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential_jitter(initial=1, max=10),
    stop=stop_after_attempt(3),
)
async def _gather_with_backoff(llm: LLMClient, query: str, cfg: HuntConfig) -> Dict[str, Any]:
    if cfg.offline:
        return _load_offline_sources()
    try:
        rate_limiter.limit("arxiv")
        payload = await _async_gather_sources(llm, query, cfg)
    except Exception as exc:
        raise RateLimitError(str(exc)) from exc
    if not payload.get("items"):
        return _load_offline_sources()
    return payload


def _persist_hunt(topic: str, payload: Dict[str, Any], cfg: HuntConfig) -> Path:
    ensure_dirs([cfg.artifact_dir(), METRIC_PATH.parent])
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    out_path = cfg.artifact_dir() / f"{timestamp}_{topic.replace(' ', '_')[:40]}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if cfg.jot_metrics:
        metrics = {
            "timestamp": timestamp,
            "topic": topic,
            "items": len(payload.get("items", [])),
            "contributors": payload.get("meta", {}).get("contributors", []),
        }
        existing = []
        if METRIC_PATH.exists():
            with suppress(Exception):
                existing = json.loads(METRIC_PATH.read_text(encoding="utf-8"))
        existing = (existing if isinstance(existing, list) else [])[-50:]
        existing.append(metrics)
        METRIC_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return out_path


def _cache_topic_path(topic: str) -> Path:
    ensure_dirs([STATE_PATH.parent])
    slug = topic.replace(" ", "_")[:80]
    return STATE_PATH.parent / f"{slug}_cache.json"


def _load_cached(topic: str, ttl: timedelta) -> Dict[str, Any] | None:
    path = _cache_topic_path(topic)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ts = payload.get("_cached_at")
        if ts:
            cached_at = datetime.fromisoformat(ts)
            if datetime.now(UTC) - cached_at <= ttl:
                return payload
    except Exception:
        return None
    return None


def _store_cache(topic: str, payload: Dict[str, Any]) -> None:
    path = _cache_topic_path(topic)
    snapshot = dict(payload)
    snapshot["_cached_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(snapshot), encoding="utf-8")


async def run_hunt_async(
    llm: LLMClient,
    query: str,
    *,
    config: HuntConfig | None = None,
) -> Tuple[Dict[str, Any], Path]:
    cfg = config or HuntConfig()
    cached = _load_cached(query, timedelta(minutes=cfg.cache_ttl_minutes))
    if cached:
        return cached, Path(cached.get("_artifact_path", ""))

    rotator = CredentialRotator(cfg.credential_spec, cfg.credential_rotation_hours)
    rotator.rotate_if_due()
    payload = await _gather_with_backoff(llm, query, cfg)
    artifact = _persist_hunt(query, payload, cfg)
    payload["_artifact_path"] = str(artifact)
    _store_cache(query, payload)
    return payload, artifact
