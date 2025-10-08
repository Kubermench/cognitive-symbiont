import json
from pathlib import Path

import yaml

from symbiont.agents.registry import AgentRegistry, CrewRunner
from symbiont.llm.client import LLMClient
from symbiont.memory.db import MemoryDB
from symbiont.tools import systems_os


def test_foresight_weaver_creates_log(monkeypatch, tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    (systems_dir / "Foresight.md").write_text(
        "## Foresight Research Log\n\n| Date | Topic | Highlight | Proposal | Approval |\n|------|-------|-----------|----------|----------|\n",
        encoding="utf-8",
    )
    original_root = systems_os.SYSTEMS_ROOT
    systems_os.SYSTEMS_ROOT = systems_dir  # type: ignore

    outputs = {
        "research:scout": json.dumps(
            {
                "topic": "Agentic self-evolution",
                "items": [
                    {
                        "title": "Scaling Agents via Continual Pre-training",
                        "url": "https://arxiv.org/abs/2509.13310",
                        "summary": "Continual pre-training teaches agent behaviours",
                    }
                ],
            }
        ),
        "research:analyze": json.dumps(
            {
                "highlight": "Continual pretraining boosts eternal mode",
                "implication": "Add offline skill loop to reflector",
            }
        ),
        "research:proposal": json.dumps(
            {
                "proposal": "Add continual_pretrain stub",
                "diff": "diff --git a/reflector.py b/reflector.py\n+def continual_pretrain(): pass\n",
            }
        ),
        "research:validate": json.dumps(
            {
                "approve": True,
                "risk": 0.3,
                "tests": ["pytest", "chaos 20%"],
            }
        ),
    }

    def fake_generate(self, prompt, label=None):  # type: ignore[override]
        return outputs.get(label or "", json.dumps({"summary": "ok"}))

    monkeypatch.setattr(LLMClient, "generate", fake_generate)

    cfg = {"db_path": str(tmp_path / "sym.db"), "llm": {}}
    crew_yaml = Path("configs/crews/foresight_weaver.yaml")
    registry = AgentRegistry.from_yaml(crew_yaml)
    db = MemoryDB(cfg["db_path"])
    runner = CrewRunner(registry, cfg, db)
    runner.run("foresight_weaver", "Self-research agenda")

    log = (systems_dir / "Foresight.md").read_text(encoding="utf-8")
    assert "Continual pretraining" in log

    systems_os.SYSTEMS_ROOT = original_root  # type: ignore
