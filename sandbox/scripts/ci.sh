#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CAP=${SANDBOX_MAX_DEPLOYS_PER_DAY:-5}
TODAY="sandbox/.state/deploys-$(date +%F)"
count=0; [[ -f "$TODAY" ]] && count=$(cat "$TODAY")
if (( count >= CAP )); then echo "[cap] daily deploy cap reached ($CAP)"; exit 2; fi
echo "[test] running pytest"
python -m pip install -r requirements.txt >/dev/null
python -m pytest -q
echo "[build] docker image"
docker build -t symbiont-sandbox:latest .
echo "[stage] up compose"
docker compose up -d
echo "[health] waiting for healthcheck"
for i in {1..30}; do
  if curl -sf http://localhost:8001/healthz | grep -q ok; then ok=1; break; fi
  sleep 1
done
if [[ "${ok:-0}" != "1" ]]; then echo "[health] failed"; docker compose logs --no-color; docker compose down; exit 1; fi
echo "[ok] staged"
echo $((count+1)) > "$TODAY"
