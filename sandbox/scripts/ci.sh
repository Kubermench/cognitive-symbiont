#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p sandbox/.state
CAP=${SANDBOX_MAX_DEPLOYS_PER_DAY:-5}
TODAY="sandbox/.state/deploys-$(date +%F)"
count=0; [[ -f "$TODAY" ]] && count=$(cat "$TODAY")
if (( count >= CAP )); then echo "[cap] daily deploy cap reached ($CAP)"; exit 2; fi

echo "[test] running pytest"
python -m pip install -r requirements.txt >/dev/null
python -m pip install -r sandbox/requirements.txt >/dev/null
pushd sandbox >/dev/null
python -m pytest -q
popd >/dev/null

echo "[build] docker image"
docker build -f sandbox/Dockerfile -t symbiont-sandbox:latest .

echo "[stage] up compose"
docker compose -f sandbox/docker-compose.yml up -d --build
echo "[health] waiting for healthcheck"
for i in {1..30}; do
  if curl -sf http://localhost:8001/healthz | grep -q ok; then ok=1; break; fi
  sleep 1
done
if [[ "${ok:-0}" != "1" ]]; then 
  echo "[health] failed"
  docker compose -f sandbox/docker-compose.yml logs --no-color
  docker compose -f sandbox/docker-compose.yml down
  exit 1
fi
echo "[ok] staged"
echo $((count+1)) > "$TODAY"
