#!/usr/bin/env python3
"""Quick demo: run async foresight hunt with collaborator + RSS mix."""

from __future__ import annotations

import asyncio
import json
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
        "meta": {"error": error},
    }


async def main(topic: str) -> None:
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
    except Exception as exc:
        payload = _fallback_payload(topic, str(exc))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import sys

    goal = sys.argv[1] if len(sys.argv) > 1 else "emergent agentic trends"
    asyncio.run(main(goal))
