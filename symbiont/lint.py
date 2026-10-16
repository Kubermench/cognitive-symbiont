"""Ruff wrapper: finds lint problems and applies ruff's own safe fixes."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import Config

_RUFF_CONFIG_FILES = ("ruff.toml", ".ruff.toml")
_RULE_CODE = re.compile(r"^[A-Z]+[0-9]+$")


@dataclass(frozen=True)
class Finding:
    file: str  # path relative to the repo root
    code: str
    message: str
    line: int
    fixable: bool


def _has_own_ruff_config(root: Path) -> bool:
    if any((root / name).exists() for name in _RUFF_CONFIG_FILES):
        return True
    pyproject = root / "pyproject.toml"
    return pyproject.exists() and "[tool.ruff" in pyproject.read_text(
        encoding="utf-8", errors="ignore"
    )


def config_problem(root: Path) -> str | None:
    """Why the repo's own ruff config can't be used, or None if it's fine (or absent)."""
    if not _has_own_ruff_config(root):
        return None
    proc = _ruff(root, "check", "--no-cache", "--show-settings", ".")
    if proc.returncode == 0:
        return None
    lines = [ln.strip() for ln in proc.stderr.splitlines() if ln.strip()]
    return lines[-1] if lines else "ruff could not load its configuration"


def _base_args(root: Path) -> list[str]:
    # A broken repo config would make every ruff call fail; ignore it rather than give up.
    return ["--isolated"] if config_problem(root) else []


def _rule_args(root: Path, cfg: Config) -> list[str]:
    if _has_own_ruff_config(root) and not config_problem(root):
        return []  # respect the repo's own choices
    args = [*_base_args(root), "--select", ",".join(cfg.lint_select)]
    if cfg.lint_ignore:
        args += ["--ignore", ",".join(cfg.lint_ignore)]
    return args


def _ruff(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        cwd=root,
        capture_output=True,
        text=True,
    )


def available() -> bool:
    try:
        return _ruff(Path.cwd(), "--version").returncode == 0
    except OSError:
        return False


def findings(root: Path, cfg: Config, files: list[str] | None = None) -> list[Finding]:
    proc = _ruff(
        root,
        "check",
        "--no-cache",
        "--output-format",
        "json",
        "--exit-zero",
        *_rule_args(root, cfg),
        *(files or ["."]),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ruff failed: {proc.stderr.strip()}")
    out: list[Finding] = []
    for item in json.loads(proc.stdout or "[]"):
        code = item.get("code")
        if not code or not _RULE_CODE.match(code):  # syntax errors aren't rules; skip them
            continue
        fix = item.get("fix")
        out.append(
            Finding(
                file=str(Path(item["filename"]).resolve().relative_to(root.resolve())),
                code=code,
                message=item.get("message", ""),
                line=int((item.get("location") or {}).get("row") or 0),
                fixable=bool(fix) and fix.get("applicability") == "safe",
            )
        )
    return out


def counts(found: list[Finding]) -> Counter[tuple[str, str]]:
    return Counter((f.file, f.code) for f in found)


def autofix(root: Path, code: str, files: list[str] | None = None) -> None:
    """Apply ruff's safe fixes for one rule (other ruff settings still apply)."""
    _ruff(
        root,
        "check",
        "--no-cache",
        "--fix",
        "--exit-zero",
        *_base_args(root),
        "--select",
        code,
        *(files or ["."]),
    )
