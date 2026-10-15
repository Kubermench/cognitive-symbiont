"""Verified proposals, kept as plain patch files under `.symbiont/` in the repo."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .git import exclude_locally, git

DIR_NAME = ".symbiont"


@dataclass
class Proposal:
    id: int
    title: str
    kind: str  # lint-autofix | lint-llm | ask
    files: list[str]
    evidence: list[str] = field(default_factory=list)  # what was verified, human readable
    status: str = "ready"  # ready | applied | undone | dropped
    created_at: float = field(default_factory=time.time)


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.dir = root / DIR_NAME / "proposals"

    def _ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        exclude_locally(self.root, f"/{DIR_NAME}/")

    def _meta(self, pid: int) -> Path:
        return self.dir / f"{pid:04d}.json"

    def patch_path(self, pid: int) -> Path:
        return self.dir / f"{pid:04d}.patch"

    def add(
        self, title: str, kind: str, files: list[str], patch: str, evidence: list[str]
    ) -> Proposal:
        self._ensure()
        pid = max((p.id for p in self.all()), default=0) + 1
        prop = Proposal(id=pid, title=title, kind=kind, files=files, evidence=evidence)
        self.patch_path(pid).write_text(patch, encoding="utf-8")
        self._save(prop)
        return prop

    def _save(self, prop: Proposal) -> None:
        self._meta(prop.id).write_text(json.dumps(asdict(prop), indent=2), encoding="utf-8")

    def all(self) -> list[Proposal]:
        if not self.dir.exists():
            return []
        return [
            Proposal(**json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(self.dir.glob("*.json"))
        ]

    def get(self, pid: int) -> Proposal:
        path = self._meta(pid)
        if not path.exists():
            raise KeyError(f"no proposal #{pid}")
        return Proposal(**json.loads(path.read_text(encoding="utf-8")))

    def patch(self, pid: int) -> str:
        return self.patch_path(pid).read_text(encoding="utf-8")

    def apply(self, pid: int) -> Proposal:
        return self._transition(pid, reverse=False, expect="ready", to="applied")

    def undo(self, pid: int) -> Proposal:
        return self._transition(pid, reverse=True, expect="applied", to="undone")

    def drop(self, pid: int) -> Proposal:
        prop = self.get(pid)
        prop.status = "dropped"
        self._save(prop)
        return prop

    def _transition(self, pid: int, reverse: bool, expect: str, to: str) -> Proposal:
        prop = self.get(pid)
        if prop.status != expect and not (expect == "ready" and prop.status == "undone"):
            raise ValueError(f"proposal #{pid} is {prop.status}")
        args = ["apply"] + (["-R"] if reverse else [])
        if not _ok(self.root, args + ["--check", str(self.patch_path(pid))]):
            raise ValueError(
                f"proposal #{pid} no longer applies cleanly; the files changed since it was made"
            )
        git(self.root, *args, str(self.patch_path(pid)))
        prop.status = to
        self._save(prop)
        return prop


def _ok(root: Path, args: list[str]) -> bool:
    return subprocess.run(["git", *args], cwd=root, capture_output=True).returncode == 0
