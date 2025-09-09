from __future__ import annotations
import os, time, subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class WatchEvent:
    kind: str
    message: str
    payload: dict


class GitIdleWatcher:
    def __init__(self, repo_path: str, idle_minutes: int = 120):
        self.repo_path = os.path.abspath(repo_path)
        self.idle_minutes = idle_minutes
        self._last_commit_seen = 0

    def check(self) -> Optional[WatchEvent]:
        try:
            out = subprocess.check_output([
                "git", "-C", self.repo_path, "log", "-1", "--format=%ct"
            ], stderr=subprocess.DEVNULL, text=True).strip()
            last_commit = int(out or 0)
            if not last_commit:
                return None
            minutes = (time.time() - last_commit) / 60
            if minutes >= self.idle_minutes and last_commit != self._last_commit_seen:
                self._last_commit_seen = last_commit
                return WatchEvent(
                    kind="git_idle",
                    message=f"Git idle for ~{int(minutes)} minutes",
                    payload={"repo": self.repo_path, "idle_min": int(minutes)}
                )
        except Exception:
            return None
        return None


class FileIdleWatcher:
    def __init__(self, root: str, idle_minutes: int = 120):
        self.root = os.path.abspath(root)
        self.idle_minutes = idle_minutes
        self._last_mtime_seen = 0

    def check(self) -> Optional[WatchEvent]:
        latest = 0
        skip_dirs = {".git", "data", ".venv", "node_modules", "__pycache__"}
        for r, ds, fs in os.walk(self.root):
            ds[:] = [d for d in ds if d not in skip_dirs]
            for fn in fs:
                try:
                    m = int(os.path.getmtime(os.path.join(r, fn)))
                    if m > latest:
                        latest = m
                except Exception:
                    pass
        if not latest:
            return None
        minutes = (time.time() - latest) / 60
        if minutes >= self.idle_minutes and latest != self._last_mtime_seen:
            self._last_mtime_seen = latest
            return WatchEvent(
                kind="file_idle",
                message=f"Files idle for ~{int(minutes)} minutes",
                payload={"root": self.root, "idle_min": int(minutes)}
            )
        return None

