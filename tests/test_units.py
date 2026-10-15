from __future__ import annotations

import os

import pytest
from conftest import CALC
from typer.testing import CliRunner

from symbiont import config, llm
from symbiont.cli import app
from symbiont.config import LLMConfig


def test_extract_code_takes_largest_block():
    reply = "Note:\n```\nx\n```\nFile:\n```python\nprint(1)\nprint(2)\n```\n"
    assert llm.extract_code(reply) == "print(1)\nprint(2)\n"
    assert llm.extract_code("no code here") is None


def test_config_defaults_and_overrides(tmp_path):
    assert config.load(tmp_path).llm.provider == "ollama"
    (tmp_path / ".symbiont.yml").write_text(
        "llm: {provider: anthropic, model: claude-sonnet-5-5}\n"
        "checks: [{name: unit, run: make test}]\n"
        "limits: {max_file_lines: 50}\n"
    )
    cfg = config.load(tmp_path)
    assert (cfg.llm.provider, cfg.llm.model) == ("anthropic", "claude-sonnet-5-5")
    assert cfg.checks[0].run == "make test"
    assert cfg.max_file_lines == 50


def test_template_is_valid_config(tmp_path):
    (tmp_path / ".symbiont.yml").write_text(config.TEMPLATE)
    assert config.load(tmp_path).llm.model == "phi3:mini"


def test_unknown_provider_and_missing_key():
    with pytest.raises(llm.LLMError, match="unknown llm provider"):
        llm.from_config(LLMConfig(provider="nope"))
    assert llm.from_config(LLMConfig(provider="none")) is None
    client = llm.from_config(LLMConfig(provider="openai", api_key_env="SYMBIONT_TEST_UNSET_KEY"))
    os.environ.pop("SYMBIONT_TEST_UNSET_KEY", None)
    with pytest.raises(llm.LLMError, match="SYMBIONT_TEST_UNSET_KEY"):
        client.complete("s", "p")


def test_cli_scan_propose_apply(repo, monkeypatch):
    monkeypatch.chdir(repo)
    run = CliRunner().invoke
    assert "No proposals yet" in run(app, ["list"]).output
    assert "F401" in run(app, ["scan"]).output

    result = run(app, ["propose", "--no-llm"])
    assert result.exit_code == 0, result.output
    assert "#1" in result.output

    shown = run(app, ["show", "1"]).output
    assert "-import os" in shown and "check 'tests': passes" in shown

    assert run(app, ["apply", "1"]).exit_code == 0
    assert "import os" not in (repo / "calc.py").read_text()
    assert run(app, ["apply", "1"]).exit_code == 1  # already applied
    assert run(app, ["undo", "1"]).exit_code == 0
    assert (repo / "calc.py").read_text() == CALC


def test_cli_outside_git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 2
    assert "not inside a git repository" in " ".join(result.output.split())


def test_invalid_config_is_a_clean_error(repo, monkeypatch):
    (repo / ".symbiont.yml").write_text('checks:\n  - run: "a"/b\n')
    with pytest.raises(config.ConfigError, match="is invalid"):
        config.load(repo)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 2 and "is invalid" in " ".join(result.output.split())
