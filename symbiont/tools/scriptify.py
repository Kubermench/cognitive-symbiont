from __future__ import annotations
import os, re, time, sqlite3
from typing import List, Optional

def _strip_fences(cmd: str) -> str:
    s = cmd.strip()
    # Remove inline backticks or fenced code blocks if present
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`")
        # After removing fences, there might be a language tag on first line
        parts = s.splitlines()
        if parts:
            # drop possible language tag line
            s = "\n".join(parts[1:])
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1]
    return s


def _extract_cmds(bullets: List[str]) -> List[str]:
    cmds: List[str] = []
    # Prefer greedy, end-anchored match to avoid early stop on ')' inside content
    anchored = re.compile(r"\(cmd:\s*(.*)\s*\)\s*$", re.DOTALL)
    fallback = re.compile(r"\(cmd:\s*(.+?)\s*\)", re.DOTALL)
    for b in bullets or []:
        m = anchored.search(b)
        if m:
            cmds.append(_strip_fences(m.group(1)))
            continue
        for m in fallback.finditer(b):
            cmds.append(_strip_fences(m.group(1)))
    return cmds


def _recent_note_sources(db_path: str, within_seconds: int = 86400, limit: int = 3) -> List[str]:
    try:
        with sqlite3.connect(db_path) as c:
            now = int(time.time())
            rows = c.execute(
                "SELECT path FROM artifacts WHERE type='note' AND created_at>=? ORDER BY id DESC LIMIT ?",
                (now - within_seconds, limit),
            ).fetchall()
        out = []
        for (p,) in rows:
            try:
                first = open(p, 'r', encoding='utf-8').read().splitlines()[0]
            except Exception:
                first = os.path.basename(p)
            out.append(f"#   {first} — {p}")
        return out
    except Exception:
        return []


def write_script(bullets, base_dir: str, db_path: Optional[str] = None, episode_id: Optional[int] = None) -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"apply_{int(time.time())}.sh")
    cmds = _extract_cmds(bullets)
    body = ["# No explicit commands; fill in below", "# echo 'do-something'"] if not cmds else cmds
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Safety: consider stashing before applying changes:",
        "#   git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git stash push -u -k -m pre-symbiont-$(date +%s) || true",
        "",
    ]
    # annotate sources from recent notes
    if episode_id is not None and db_path:
        sources = []
        try:
            with sqlite3.connect(db_path) as c:
                rows = c.execute(
                    "SELECT a.path FROM episode_artifacts ea JOIN artifacts a ON ea.artifact_id=a.id WHERE ea.episode_id=? ORDER BY ea.linked_at DESC LIMIT 5",
                    (episode_id,),
                ).fetchall()
            for (p,) in rows:
                try:
                    first = open(p, 'r', encoding='utf-8').read().splitlines()[0]
                except Exception:
                    first = os.path.basename(p)
                sources.append(f"#   {first} — {p}")
        except Exception:
            sources = []
    else:
        db_fallback = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'symbiont.db')
        sources = _recent_note_sources(db_fallback)
    trailer = ["", "# Sources (recent notes):"] + (sources or ["#   (none)"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + body + trailer + ["\n"]))
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass
    return path


def write_rollback_script(apply_path: str) -> str:
    base_dir = os.path.dirname(apply_path)
    ts = int(time.time())
    path = os.path.join(base_dir, f"rollback_{ts}.sh")
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Attempt a safe rollback using git if available.",
        "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then",
        "  echo '[rollback] Using git to restore working tree...'",
        "  # Prefer popping last stash created by apply script; otherwise hard reset uncommitted changes.",
        "  git stash list | grep pre-symbiont- >/dev/null 2>&1 && git stash pop || true",
        "  git reset --hard HEAD || true",
        "  git clean -fd || true",
        "else",
        "  echo '[rollback] No git repo detected. Manual cleanup may be required.'",
        "fi",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body + ["\n"]))
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass
    return path
