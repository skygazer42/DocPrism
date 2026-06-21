#!/usr/bin/env bash
set -euo pipefail

kill_tree() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
}

port_pids() {
  local port="$1"
  ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
}

for pid in $(port_pids 18100) $(port_pids 18180); do
  kill_tree "$pid"
done

for pid in $(pgrep -f "/data/mineru-vlm-lab/scripts/run_enhancement_worker.py" 2>/dev/null || true); do
  kill_tree "$pid"
done

sleep 2

for pid in $(port_pids 18100) $(port_pids 18180); do
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
done
