#!/usr/bin/env bash
set -euo pipefail

# Attempt a safe rollback using git if available.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo '[rollback] Using git to restore working tree...'
  # Prefer popping last stash created by apply script; otherwise hard reset uncommitted changes.
  git stash list | grep pre-symbiont- >/dev/null 2>&1 && git stash pop || true
  git reset --hard HEAD || true
  git clean -fd || true
else
  echo '[rollback] No git repo detected. Manual cleanup may be required.'
fi


