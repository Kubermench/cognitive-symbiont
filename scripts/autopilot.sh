#!/usr/bin/env bash
set -euo pipefail

# Symbiont Autopilot: proposal → guard → sandbox → apply → evolve (repeatable)

export PYTHONUNBUFFERED=1

MODE="single"
CYCLES=1
AUTONOMY="guarded"
SWARM=false
ETERNAL_OVERRIDE=false
THRESHOLD_GUARDED=0.5
THRESHOLD_FULL=0.85

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode=*) MODE="${1#*=}" ;;
    --mode) MODE="$2"; shift ;;
    --cycles=*) CYCLES="${1#*=}" ;;
    --cycles) CYCLES="$2"; shift ;;
    --autonomy=*) AUTONOMY="${1#*=}" ;;
    --autonomy) AUTONOMY="$2"; shift ;;
    --swarm=*) SWARM="${1#*=}" ;;
    --swarm) SWARM="$2"; shift ;;
    --eternal-override) ETERNAL_OVERRIDE=true ;;
    *) echo "[autopilot] unknown flag $1"; exit 2 ;;
  esac
  shift
done

if [[ "$MODE" == "eternal" && "${CYCLES}" == "1" ]]; then
  CYCLES=10
fi

if [[ "$AUTONOMY" != "guarded" && "$AUTONOMY" != "full" ]]; then
  echo "[autopilot] invalid autonomy mode: $AUTONOMY"
  exit 2
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate || true
fi

threshold=$THRESHOLD_GUARDED
if [[ "$AUTONOMY" == "full" ]]; then
  threshold=$THRESHOLD_FULL
fi

for ((cycle=1; cycle<=CYCLES; cycle++)); do
  echo "[autopilot] cycle $cycle/$CYCLES (mode=$MODE autonomy=$AUTONOMY)"
  python -m symbiont.cli propose-here || true

  latest_apply=$(ls -1t ./data/artifacts/scripts/apply_*.sh 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_apply:-}" ]]; then
    echo "[autopilot] no apply script found; sleeping"
    sleep 5
    continue
  fi

  guard_json=$(python -m symbiont.cli guard --script "$latest_apply" --json)
  rogue_score=$(SYMBIONT_GUARD_JSON="$guard_json" python - <<'PY'
import json, os
report = json.loads(os.environ.get("SYMBIONT_GUARD_JSON", "{}"))
print(report.get("rogue_score", 0.0))
PY
  )

  echo "[autopilot] rogue score for $latest_apply: $rogue_score"
  python -m symbiont.cli guard --script "$latest_apply" || true

  if python - <<PY
threshold = float("$threshold")
score = float("$rogue_score")
exit(0 if score <= threshold else 1)
PY
  then
    echo "[autopilot] guard checks passed"
  else
    echo "[autopilot] guard triggered (score $rogue_score > $threshold)"
    if [[ "$AUTONOMY" == "full" ]]; then
      echo "[autopilot] full autonomy: logging warning and continuing"
    else
      echo "[autopilot] guarded autonomy: pausing loop"
      break
    fi
  fi

  python -m symbiont.cli rollback-test "$latest_apply" || {
    echo "[autopilot] rollback validation failed; reverting and pausing"
    git checkout -- . >/dev/null 2>&1 || true
    break
  }

  if ! python -m symbiont.cli run-script "$latest_apply" --yes; then
    echo "[autopilot] apply failed; attempting rollback"
    rollback_script="${latest_apply/apply_/rollback_}"
    if [[ -f "$rollback_script" ]]; then
      bash "$rollback_script" || true
    fi
    git checkout -- . >/dev/null 2>&1 || true
    break
  fi

  CI_ENV=()
  if [[ "$ETERNAL_OVERRIDE" == "true" ]]; then
    override_allowed=$(ROGUE_SCORE="$rogue_score" python - <<'PY'
import os
score = float(os.environ.get("ROGUE_SCORE", "0"))
print("ok" if score <= 0.5 else "skip")
PY
)
    if [[ "$override_allowed" == "ok" ]]; then
      echo "[autopilot] override active: lifting deploy cap"
      CI_ENV+=(SANDBOX_MAX_DEPLOYS_PER_DAY=999999)
    else
      echo "[autopilot] override requested but rogue score high ($rogue_score); keeping default cap"
    fi
  fi

  if [ -x sandbox/scripts/ci.sh ]; then
    echo "[autopilot] run sandbox CI"
    if (( ${#CI_ENV[@]} )); then
      CI_SUCCESS=0
      (cd sandbox && "${CI_ENV[@]}" ./scripts/ci.sh) || CI_SUCCESS=$?
    else
      CI_SUCCESS=0
      (cd sandbox && ./scripts/ci.sh) || CI_SUCCESS=$?
    fi
    if [[ $CI_SUCCESS -ne 0 ]]; then
      echo "[autopilot] sandbox CI failed; reverting"
      rollback_script="${latest_apply/apply_/rollback_}"
      if [[ -f "$rollback_script" ]]; then
        bash "$rollback_script" || true
      fi
      git checkout -- . >/dev/null 2>&1 || true
      break
    fi
  fi

  python -m symbiont.cli evolve_self --scope planner --strategy promote_diversity || true

  if [[ "$SWARM" == "true" || "$SWARM" == "1" ]]; then
    python -m symbiont.cli swarm_evolve --auto --variants 3 || true
    python -m symbiont.cli swarm_merge_transcripts || true
  fi

  if command -v git >/dev/null 2>&1; then
    if [ -n "$(git status --porcelain || true)" ]; then
      echo "[autopilot] committing changes to branch symbiont/autopilot"
      git checkout -B symbiont/autopilot || true
      git add -A || true
      git restore --staged .venv 2>/dev/null || true
      git rm -r --cached .venv 2>/dev/null || true
      git commit -m "symbiont: autopilot cycle $(date -u +%F)" || true
      if [ "${SYMBIONT_AUTOPILOT_PUSH:-0}" = "1" ]; then
        git push -u origin symbiont/autopilot || true
      fi
    else
      echo "[autopilot] no changes to commit"
    fi
  fi

  if [[ "$MODE" != "eternal" ]]; then
    break
  fi

done

echo "[autopilot] finished mode=$MODE autonomy=$AUTONOMY"
