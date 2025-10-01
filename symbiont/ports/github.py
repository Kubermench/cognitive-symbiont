from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from github import Github
except Exception:  # pragma: no cover
    Github = None


@dataclass
class PullRequestResult:
    dry_run: bool
    message: str
    url: Optional[str] = None
    number: Optional[int] = None


class GitHubGuard:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.github_cfg = (((self.config.get("ports", {}) or {}).get("github") or {}))
        self.repo_slug = self.github_cfg.get("repository")
        self.owner_allowlist = set(self.github_cfg.get("owner_allowlist", []))
        self.token_path = self.github_cfg.get("token_path")

    def create_pull_request(self, title: str, body: str, head: str = "symbiont/autopilot", base: Optional[str] = None, dry_run: bool = False) -> PullRequestResult:
        repo_slug = self.repo_slug or self._infer_repo_slug()
        if not repo_slug:
            return PullRequestResult(dry_run=True, message="Unable to determine repository slug")
        owner = repo_slug.split("/")[0]
        if self.owner_allowlist and owner not in self.owner_allowlist:
            return PullRequestResult(dry_run=True, message=f"Owner '{owner}' not allowlisted")

        token = self._load_token()
        if dry_run or Github is None or not token:
            return PullRequestResult(dry_run=True, message="Dry run or PyGithub/token unavailable")

        gh = Github(token)
        repo = gh.get_repo(repo_slug)
        pr = repo.create_pull(title=title, body=body, head=head, base=base or repo.default_branch)
        return PullRequestResult(dry_run=False, message="Pull request created", url=pr.html_url, number=pr.number)

    # ------------------------------------------------------------------
    def _load_token(self) -> Optional[str]:
        env_token = os.environ.get("SYMBIONT_GITHUB_TOKEN")
        if env_token:
            return env_token.strip()
        if not self.token_path:
            return None
        path = Path(self.token_path).expanduser()
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip()

    def _infer_repo_slug(self) -> Optional[str]:
        try:
            out = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, check=True)
            url = out.stdout.strip()
            if url.endswith(".git"):
                url = url[:-4]
            if url.startswith("git@"):
                _, rest = url.split(":", 1)
                return rest
            if url.startswith("https://"):
                return url.split("github.com/")[-1]
        except Exception:
            return None
        return None

