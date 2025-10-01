from __future__ import annotations
import os, time, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable, List

try:
    import yaml
except Exception:  # pragma: no cover - yaml should be available but keep optional
    yaml = None


@dataclass
class WatchEvent:
    kind: str
    message: str
    payload: dict


@dataclass(slots=True)
class RepoWatchConfig:
    """Normalized watcher configuration for a repository root."""

    path: Path
    watchers: tuple[str, ...] = ("file", "git", "timer")
    idle_minutes: int = 120
    git_idle_minutes: int = 120
    timer_minutes: int = 120
    trigger_mode: str = "idle_and_timer"  # or "any"
    verify_rollback: bool = False

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "watchers": list(self.watchers),
            "idle_minutes": self.idle_minutes,
            "git_idle_minutes": self.git_idle_minutes,
            "timer_minutes": self.timer_minutes,
            "trigger_mode": self.trigger_mode,
            "verify_rollback": self.verify_rollback,
        }


def _normalize_watchers(raw: Iterable[str]) -> tuple[str, ...]:
    seen = []
    for item in raw:
        item = (item or "").strip().lower()
        if item and item in {"file", "git", "timer"} and item not in seen:
            seen.append(item)
    return tuple(seen) or ("file", "git", "timer")


def _load_yaml_watch_targets(path: Path) -> List[RepoWatchConfig]:
    if not path.exists() or yaml is None:
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    items = []
    for entry in raw.get("repositories", []):
        repo_path = Path(entry.get("path", ".")).expanduser().resolve()
        items.append(
            RepoWatchConfig(
                path=repo_path,
                watchers=_normalize_watchers(entry.get("watchers", ("file", "git", "timer"))),
                idle_minutes=int(entry.get("idle_minutes", 120)),
                git_idle_minutes=int(entry.get("git_idle_minutes", entry.get("idle_minutes", 120))),
                timer_minutes=int(entry.get("timer_minutes", 120)),
                trigger_mode=str(entry.get("trigger_mode", "idle_and_timer") or "idle_and_timer"),
                verify_rollback=bool(entry.get("verify_rollback", False)),
            )
        )
    return items


def build_repo_watch_configs(cfg: dict) -> List[RepoWatchConfig]:
    """Return normalized repo watcher configs from main config + optional YAML file."""

    ini = cfg.get("initiative", {}) if isinstance(cfg, dict) else {}
    configs: List[RepoWatchConfig] = []

    # Inline watch_targets takes precedence if provided.
    for entry in ini.get("watch_targets", []) or []:
        repo_path = Path(entry.get("path", ini.get("repo_path", "."))).expanduser().resolve()
        configs.append(
            RepoWatchConfig(
                path=repo_path,
                watchers=_normalize_watchers(entry.get("watchers", ini.get("watchers", ["file", "git", "timer"]))),
                idle_minutes=int(entry.get("idle_minutes", ini.get("idle_minutes", 120))),
                git_idle_minutes=int(entry.get("git_idle_minutes", entry.get("idle_minutes", ini.get("git_idle_minutes", ini.get("idle_minutes", 120))))),
                timer_minutes=int(entry.get("timer_minutes", ini.get("timer_minutes", 120))),
                trigger_mode=str(entry.get("trigger_mode", ini.get("trigger_mode", "idle_and_timer")) or "idle_and_timer"),
                verify_rollback=bool(entry.get("verify_rollback", ini.get("verify_rollback", False))),
            )
        )

    if configs:
        return configs

    # Fall back to optional YAML file if configured.
    yaml_path = ini.get("watch_config_path")
    if yaml_path:
        yaml_configs = _load_yaml_watch_targets(Path(yaml_path).expanduser().resolve())
        if yaml_configs:
            return yaml_configs

    # Legacy single-repo settings.
    repo_path = Path(ini.get("repo_path", ".")).expanduser().resolve()
    configs.append(
        RepoWatchConfig(
            path=repo_path,
            watchers=_normalize_watchers(ini.get("watchers", ["file", "git", "timer"])),
            idle_minutes=int(ini.get("idle_minutes", 120)),
            git_idle_minutes=int(ini.get("git_idle_minutes", ini.get("idle_minutes", 120))),
            timer_minutes=int(ini.get("timer_minutes", 120)),
            trigger_mode=str(ini.get("trigger_mode", "idle_and_timer") or "idle_and_timer"),
            verify_rollback=bool(ini.get("verify_rollback", False)),
        )
    )
    return configs


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
