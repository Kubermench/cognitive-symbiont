"""Git helpers and the isolated worktree every fix is attempted in."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# Ignored directories that checks usually need (installed dependencies); linked, not copied.
ENV_DIRS = (".venv", "venv", "node_modules")

# Identity for the throwaway snapshot commit inside a sandbox; it never reaches the user's branches.
_IDENTITY = [
    "-c",
    "user.name=symbiont",
    "-c",
    "user.email=symbiont@localhost",
    "-c",
    "commit.gpgsign=false",
]


class GitError(RuntimeError):
    pass


def git(cwd: Path, *args: str, input: bytes | None = None, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, input=input, capture_output=True)
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode(errors="replace")


def repo_root(start: Path) -> Path:
    try:
        return Path(git(start, "rev-parse", "--show-toplevel").strip())
    except (GitError, FileNotFoundError) as exc:
        raise GitError(f"{start} is not inside a git repository") from exc


def changed_files(worktree: Path) -> list[str]:
    """Paths changed relative to HEAD, new files included."""
    git(worktree, "add", "-A")
    return [p for p in git(worktree, "diff", "--cached", "--name-only", "HEAD").splitlines() if p]


def exclude_locally(root: Path, pattern: str) -> None:
    """Hide a path from `git status` without touching the repo's .gitignore."""
    exclude = Path(git(root, "rev-parse", "--git-path", "info/exclude").strip())
    if not exclude.is_absolute():
        exclude = root / exclude
    exclude.parent.mkdir(parents=True, exist_ok=True)
    lines = exclude.read_text(encoding="utf-8").splitlines() if exclude.exists() else []
    if pattern not in lines:
        with exclude.open("a", encoding="utf-8") as fh:
            fh.write(("\n" if lines and lines[-1] else "") + pattern + "\n")


class Sandbox:
    """A detached worktree holding a copy of the user's current state, uncommitted work included.

    Inside it, the user's state is committed as a snapshot so `diff()` returns exactly the
    changes made in the sandbox, as a patch that applies cleanly to the user's working tree.
    """

    def __init__(self, root: Path):
        self.root = root
        self.path: Path | None = None
        self._tmp: str | None = None

    def __enter__(self) -> Sandbox:
        self._tmp = tempfile.mkdtemp(prefix="symbiont-")
        self.path = Path(self._tmp) / "wt"
        git(self.root, "worktree", "add", "--detach", "--quiet", str(self.path), "HEAD")
        pending = subprocess.run(
            ["git", "diff", "HEAD", "--binary"], cwd=self.root, capture_output=True, check=True
        ).stdout
        if pending.strip():
            git(self.path, "apply", "--whitespace=nowarn", "-", input=pending)
        untracked = git(self.root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
        for rel in filter(None, untracked):
            src, dst = self.root / rel, self.path / rel
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for name in ENV_DIRS:
            src = self.root / name
            if src.is_dir() and _ignored(self.root, name) and not (self.path / name).exists():
                (self.path / name).symlink_to(src, target_is_directory=True)
        git(self.path, "add", "-A")
        git(
            self.path,
            *_IDENTITY,
            "commit",
            "--quiet",
            "--allow-empty",
            "--no-verify",
            "-m",
            "symbiont snapshot",
        )
        return self

    def diff(self) -> str:
        assert self.path is not None
        git(self.path, "add", "-A")
        return git(self.path, "diff", "--cached", "--binary", "HEAD")

    def changed_files(self) -> list[str]:
        assert self.path is not None
        return changed_files(self.path)

    def reset(self) -> None:
        assert self.path is not None
        git(self.path, "reset", "--quiet", "--hard", "HEAD")
        git(self.path, "clean", "-qfd")  # keeps ignored paths, including the linked env dirs

    def __exit__(self, *exc) -> None:
        if self.path is not None:
            git(self.root, "worktree", "remove", "--force", str(self.path), check=False)
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
        git(self.root, "worktree", "prune", check=False)


def _ignored(root: Path, rel: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", rel], cwd=root).returncode == 0
