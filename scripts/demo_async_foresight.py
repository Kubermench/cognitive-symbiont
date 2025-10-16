#!/usr/bin/env python3
"""Quick demo: run async foresight hunt with collaborator + RSS mix."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from symbiont.initiative.daemon import validate_foresight_config
from symbiont.llm.client import LLMClient
from symbiont.tools.research import gather_trend_sources_async


def _load_cfg() -> Dict[str, Any]:
    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _fallback_payload(topic: str, error: str) -> Dict[str, Any]:
    return {
        "topic": topic,
        "items": [],
        "meta": {"mode": "fallback", "error": error},
    }


def _load_offline_mock(topic: str) -> Dict[str, Any]:
    mock_path = Path("data/foresight/mocks.json")
    if not mock_path.exists():
        return _fallback_payload(topic, "offline dataset missing")
    try:
        data = json.loads(mock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fallback_payload(topic, f"offline dataset load error: {exc}")

    entries = data.get("entries") or []
    topic_lower = topic.lower()
    chosen: Dict[str, Any] | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if topic_lower in str(entry.get("topic", "")).lower():
            chosen = entry
            break
    if chosen is None and entries:
        chosen = entries[0]
    if chosen is None:
        return _fallback_payload(topic, "offline dataset empty")
    payload = {
        "topic": topic,
        "items": chosen.get("items", []),
        "meta": chosen.get("meta", {}),
    }
    payload.setdefault("meta", {}).update({"mode": "offline"})
    return payload


async def main(topic: str, *, offline: bool) -> None:
    if offline:
        payload = _load_offline_mock(topic)
        print(json.dumps(payload, indent=2))
        return

    cfg = _load_cfg()
    foresight_model = validate_foresight_config(cfg.get("foresight"))
    llm_cfg = cfg.get("llm", {})
    include_collab = foresight_model.collaboration.enabled
    include_rss = True

    try:
        payload = await gather_trend_sources_async(
            LLMClient(llm_cfg),
            topic,
            include_collaborators=include_collab,
            include_rss=include_rss,
        )
        payload.setdefault("meta", {}).update({"mode": "live"})
    except Exception as exc:
        payload = _load_offline_mock(topic)
        payload.setdefault("meta", {}).update({"mode": "offline", "error": str(exc)})
    print(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blended foresight hunt demo")
    parser.add_argument("topic", nargs="?", default="emergent agentic trends", help="Topic to research")
    parser.add_argument("--offline", action="store_true", help="Use cached offline responses")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    goal = args.topic
    offline_flag = args.offline or os.getenv("SYMBIONT_OFFLINE") == "1"
    asyncio.run(main(goal, offline=offline_flag))
