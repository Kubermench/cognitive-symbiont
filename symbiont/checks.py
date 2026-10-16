"""The repo's own checks (usually its test suite), which must keep passing for a fix to count."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Check, Config


@dataclass
class Result:
    name: str
    passed: bool
    output: str  # tail of combined stdout/stderr


def _python(root: Path) -> str:
    """Prefer the repo's own virtualenv so its dependencies are importable."""
    for venv in (".venv", "venv"):
        for exe in ("bin/python", "Scripts/python.exe"):
            candidate = root / venv / exe
            if candidate.exists():
                return shlex.quote(str(candidate))
    return shlex.quote(sys.executable)


def detect(root: Path, cfg: Config) -> list[Check]:
    if cfg.checks is not None:
        return cfg.checks
    pyproject = root / "pyproject.toml"
    has_pytest = (
        (root / "pytest.ini").exists()
        or (root / "conftest.py").exists()
        or (root / "tests").is_dir()
        or (root / "test").is_dir()
        or (
            pyproject.exists()
            and "[tool.pytest" in pyproject.read_text(encoding="utf-8", errors="ignore")
        )
    )
    if has_pytest:
        return [Check(name="tests", run=f"{_python(root)} -m pytest -q -x -p no:cacheprovider")]
    return []


def run(root: Path, checks: list[Check], timeout: int, repo: Path | None = None) -> list[Result]:
    """Run checks in `root` (normally a sandbox). $SYMBIONT_ROOT points at the real repo."""
    env = {**os.environ, "SYMBIONT_ROOT": str(repo or root), "PYTHONDONTWRITEBYTECODE": "1"}
    results = []
    for check in checks:
        try:
            proc = subprocess.run(
                check.run,
                cwd=root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            output = (proc.stdout + proc.stderr)[-4000:]
            results.append(Result(check.name, proc.returncode == 0, output))
        except subprocess.TimeoutExpired:
            results.append(Result(check.name, False, f"timed out after {timeout}s"))
    return results


def regressions(baseline: list[Result], after: list[Result]) -> list[Result]:
    """Checks that passed before the change and fail after it."""
    passed_before = {r.name for r in baseline if r.passed}
    return [r for r in after if r.name in passed_before and not r.passed]
