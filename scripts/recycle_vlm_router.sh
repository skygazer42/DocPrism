#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
PORT="${MINERU_VLM_ROUTER_PORT:-18100}"
HOST="${MINERU_VLM_ROUTER_HOST:-127.0.0.1}"
WAIT_SECONDS="${MINERU_VLM_RECYCLE_WAIT_SECONDS:-120}"
START_SCRIPT="${MINERU_VLM_RECYCLE_START_SCRIPT:-$ROOT/scripts/start_vlm_router_gpu45.sh}"
LOG_PATH="${MINERU_VLM_RECYCLE_LOG_PATH:-$ROOT/logs/vlm-router.log}"
HEALTH_URL="http://$HOST:$PORT/health"

port_pids() {
  ss -ltnp "sport = :$PORT" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

kill_tree() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
}

force_kill_tree() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

mkdir -p "$(dirname "$LOG_PATH")"

for pid in $(port_pids); do
  kill_tree "$pid"
done

sleep 3

for pid in $(port_pids); do
  force_kill_tree "$pid"
done

nohup "$START_SCRIPT" > "$LOG_PATH" 2>&1 &
new_pid="$!"

for _ in $(seq 1 "$WAIT_SECONDS"); do
  if curl -fsS --max-time 2 --noproxy '*' "$HEALTH_URL" >/dev/null 2>&1; then
    printf '{"status":"ok","router_pid":%s,"health_url":"%s"}\n' "$new_pid" "$HEALTH_URL"
    exit 0
  fi
  sleep 1
done

printf '{"status":"timeout","router_pid":%s,"health_url":"%s"}\n' "$new_pid" "$HEALTH_URL" >&2
exit 1
