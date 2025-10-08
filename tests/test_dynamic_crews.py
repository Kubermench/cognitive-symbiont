import json
from pathlib import Path

import yaml

from symbiont.orchestration.dynamic import generate_dynamic_crew_yaml
from symbiont.llm.client import LLMClient


def test_generate_dynamic_crew_yaml_success(monkeypatch, tmp_path: Path) -> None:
    def fake_generate(self, prompt, label=None):  # type: ignore[override]
        return json.dumps(
            {
                "crew_name": "analysis",
                "roles": [
                    {"name": "scout", "tools": ["repo_scan"]},
                    {"name": "architect", "tools": []},
                    {"name": "critic", "tools": []},
                ],
                "supervisor": {"routing": "sequential"},
            }
        )

    monkeypatch.setattr(LLMClient, "generate", fake_generate)
    cfg = {"llm": {}}
    crew_name, yaml_path = generate_dynamic_crew_yaml("Refactor module", cfg, output_dir=tmp_path)
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text())
    assert crew_name in data["crew"]
    sequence = data["crew"][crew_name]["sequence"]
    assert len(sequence) == 3


def test_generate_dynamic_crew_yaml_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(LLMClient, "generate", lambda *args, **kwargs: "{{invalid")
    cfg = {"llm": {}}
    crew_name, yaml_path = generate_dynamic_crew_yaml("Unknown goal", cfg, output_dir=tmp_path)
    data = yaml.safe_load(yaml_path.read_text())
    assert len(data["crew"][crew_name]["sequence"]) >= 3
    agents = data["agents"]
    assert any(agent_def["role"] == "critic" for agent_def in agents.values())
