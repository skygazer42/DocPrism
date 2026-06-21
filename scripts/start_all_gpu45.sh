#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
mkdir -p "$ROOT/logs"

nohup "$ROOT/scripts/start_vlm_router_gpu45.sh" > "$ROOT/logs/vlm-router.log" 2>&1 &
echo "vlm_router_pid=$!"

for _ in $(seq 1 300); do
  if curl -sf --noproxy '*' http://127.0.0.1:18100/health >/dev/null; then
    break
  fi
  sleep 1
done

nohup "$ROOT/scripts/start_orchestrator.sh" > "$ROOT/logs/orchestrator.log" 2>&1 &
echo "orchestrator_pid=$!"

for _ in $(seq 1 60); do
  if curl -sf --noproxy '*' http://127.0.0.1:18180/health >/dev/null; then
    break
  fi
  sleep 1
done

curl -sS --noproxy '*' http://127.0.0.1:18180/health
