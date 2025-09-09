from __future__ import annotations
import os, re, time, sqlite3
from typing import List

def _extract_cmds(bullets):
    cmds = []
    for b in bullets:
        m = re.search(r"\(cmd:\s*(.+?)\s*\)$", b)
        if m:
            cmds.append(m.group(1))
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


def write_script(bullets, base_dir: str) -> str:
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
    db_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'symbiont.db')
    sources = _recent_note_sources(db_path)
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
