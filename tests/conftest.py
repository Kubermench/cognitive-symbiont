from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

CALC = """\
import os


def divide(a, b):
    try:
        return a / b
    except:
        return None
"""

TEST_CALC = """\
from calc import divide


def test_divide():
    assert divide(6, 3) == 2
    assert divide(1, 0) is None
"""

FIXED_CALC = CALC.replace("except:", "except ZeroDivisionError:")


class FakeLLM:
    """Returns canned replies in order and records the prompts it was sent."""

    name = "fake:model"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def fenced(code: str) -> str:
    return f"Here you go:\n```python\n{code}```\n"


def sh(cwd: Path, *args: str) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "test")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "test@example.com")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "calc.py").write_text(CALC)
    (root / "tests" / "test_calc.py").write_text(TEST_CALC)
    (root / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    sh(root, "git", "init", "-q", "-b", "main")
    sh(root, "git", "add", "-A")
    sh(root, "git", "commit", "-q", "-m", "init")
    return root
