"""Run the full foresight engine (async hunts + analyzer + suggester)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from symbiont.foresight import ForesightAnalyzer, ForesightSuggester, HuntConfig, run_hunt_async
from symbiont.llm.client import LLMClient


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


async def _run(goal: str, cfg: dict, offline: bool = False) -> None:
    llm = LLMClient(cfg.get("llm", {}))
    hunt_cfg = HuntConfig(
        offline=offline,
        include_collaborators=cfg.get("foresight", {}).get("collaboration", {}).get("enabled", False),
        credential_spec=cfg.get("foresight", {}).get("credential"),
    )
    payload, artifact = await run_hunt_async(llm, goal, config=hunt_cfg)
    analyzer = ForesightAnalyzer()
    weighted = analyzer.weight_sources(goal, payload.get("items", []))
    version = analyzer.version_triples(goal, weighted["items"])
    suggester = ForesightSuggester(llm)
    proposal = suggester.draft(
        {"topic": goal, "highlight": weighted["meta"].get("avg_score", 0.0)},
        context={
            "triples": version.get("triples", []),
            "relevance": weighted["meta"].get("avg_score", 0.0),
            "query": goal,
        },
    )

    summary = {
        "goal": goal,
        "artifact": str(artifact),
        "avg_score": weighted["meta"].get("avg_score"),
        "triples_diff": version.get("diff_path"),
        "proposal": proposal,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo foresight engine workflow")
    parser.add_argument("goal", help="Topic to scout")
    parser.add_argument("--offline", action="store_true", help="Use cached sources instead of live APIs")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    asyncio.run(_run(args.goal, cfg, offline=args.offline))


if __name__ == "__main__":
    main()
