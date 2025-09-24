#!/usr/bin/env bash
set -euo pipefail

# Symbiont Autopilot: propose → apply latest script → run sandbox CI (if present) → optional PR push

export PYTHONUNBUFFERED=1

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate || true
fi

echo "[autopilot] propose one 10-minute refactor"
python -m symbiont.cli propose-here || true

echo "[autopilot] find latest apply script"
latest_apply=$(ls -1t ./data/artifacts/scripts/apply_*.sh 2>/dev/null | head -n 1 || true)
if [ -z "${latest_apply:-}" ]; then
  echo "[autopilot] no apply script found; exiting"
  exit 0
fi

echo "[autopilot] run latest apply: $latest_apply"
python -m symbiont.cli run-script "$latest_apply" --yes || true

if [ -x sandbox/scripts/ci.sh ]; then
  echo "[autopilot] run sandbox CI"
  (cd sandbox && ./scripts/ci.sh) || true
fi

if command -v git >/dev/null 2>&1; then
  if [ -n "$(git status --porcelain || true)" ]; then
    echo "[autopilot] committing changes to branch symbiont/autopilot"
    git checkout -B symbiont/autopilot || true
    git add -A || true
    # Avoid committing local virtualenv if tracked
    git restore --staged .venv 2>/dev/null || true
    git rm -r --cached .venv 2>/dev/null || true
    git commit -m "symbiont: autopilot cycle $(date -u +%F)" || true
    if [ "${SYMBIONT_AUTOPILOT_PUSH:-0}" = "1" ]; then
      echo "[autopilot] pushing to origin (SYMBIONT_AUTOPILOT_PUSH=1)"
      git push -u origin symbiont/autopilot || true
    else
      echo "[autopilot] push disabled (set SYMBIONT_AUTOPILOT_PUSH=1 to enable)"
    fi
  else
    echo "[autopilot] no changes to commit"
  fi
fi

echo "[autopilot] done"
