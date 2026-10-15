"""Per-repo configuration, read from an optional `.symbiont.yml` at the repo root."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = ".symbiont.yml"

DEFAULT_SELECT = ["E", "F", "W", "B", "UP", "SIM", "I"]
DEFAULT_IGNORE = ["E501"]

TEMPLATE = """\
# Cognitive Symbiont settings. Every key is optional.
llm:
  provider: ollama        # ollama | openai | anthropic | none
  model: phi3:mini
  # host: http://localhost:11434          # ollama
  # base_url: https://api.openai.com/v1   # openai or any OpenAI-compatible server
  # api_key_env: OPENAI_API_KEY           # env var holding the key (openai/anthropic)
  timeout: 300

# Commands that must keep passing for a fix to be accepted.
# Leave unset to autodetect (pytest when the repo has tests).
# checks:
#   - name: tests
#     run: python -m pytest -q

lint:
  # Rules used when the repo has no ruff config of its own.
  select: [E, F, W, B, UP, SIM, I]
  ignore: [E501]

limits:
  max_file_lines: 400     # files larger than this are never sent to the LLM
  check_timeout: 900      # seconds per check command
"""


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "phi3:mini"
    host: str = "http://localhost:11434"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = None
    timeout: float = 300.0


@dataclass
class Check:
    name: str
    run: str


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    checks: list[Check] | None = None  # None means autodetect
    lint_select: list[str] = field(default_factory=lambda: list(DEFAULT_SELECT))
    lint_ignore: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    max_file_lines: int = 400
    check_timeout: int = 900


class ConfigError(ValueError):
    pass


def load(root: Path) -> Config:
    try:
        return _load(root)
    except ConfigError:
        raise
    except (yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError) as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise ConfigError(f"{CONFIG_NAME} is invalid: {detail}") from exc


def _load(root: Path) -> Config:
    path = root / CONFIG_NAME
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{CONFIG_NAME} must be a mapping")

    cfg = Config()
    llm = raw.get("llm") or {}
    for key in ("provider", "model", "host", "base_url", "api_key_env"):
        if llm.get(key) is not None:
            setattr(cfg.llm, key, str(llm[key]))
    if llm.get("timeout") is not None:
        cfg.llm.timeout = float(llm["timeout"])

    if raw.get("checks") is not None:
        cfg.checks = [
            Check(name=str(c.get("name") or c["run"]), run=str(c["run"])) for c in raw["checks"]
        ]

    lint = raw.get("lint") or {}
    if lint.get("select") is not None:
        cfg.lint_select = [str(s) for s in lint["select"]]
    if lint.get("ignore") is not None:
        cfg.lint_ignore = [str(s) for s in lint["ignore"]]

    limits = raw.get("limits") or {}
    if limits.get("max_file_lines") is not None:
        cfg.max_file_lines = int(limits["max_file_lines"])
    if limits.get("check_timeout") is not None:
        cfg.check_timeout = int(limits["check_timeout"])
    return cfg
