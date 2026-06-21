#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/mineru32-bench/env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_helpers.sh"
load_env_preserving_overrides "$ROOT/production.env" \
  MINERU_ENV \
  MINERU_VLM_ROUTER_HOST \
  MINERU_VLM_ROUTER_PORT \
  MINERU_VLM_WORKER_HOST \
  MINERU_VLM_GPUS \
  MINERU_VLM_REPLICAS_PER_GPU \
  MINERU_VLM_WORKER_CONCURRENCY \
  MINERU_VLM_REPLICA_BASE_PORT \
  MINERU_VLM_PRELOAD \
  MINERU_PROCESSING_WINDOW_SIZE \
  MINERU_VLM_REPLICA_DRY_RUN \
  MINERU_VLM_REPLICA_HEALTH_ATTEMPTS

HOST="${MINERU_VLM_ROUTER_HOST:-127.0.0.1}"
PORT="${MINERU_VLM_ROUTER_PORT:-18100}"
WORKER_HOST="${MINERU_VLM_WORKER_HOST:-127.0.0.1}"
GPUS="${MINERU_VLM_GPUS:-4,5}"
REPLICAS_PER_GPU="${MINERU_VLM_REPLICAS_PER_GPU:-2}"
WORKER_CONCURRENCY="${MINERU_VLM_WORKER_CONCURRENCY:-1}"
BASE_PORT="${MINERU_VLM_REPLICA_BASE_PORT:-19000}"
PRELOAD="${MINERU_VLM_PRELOAD:-true}"
PROCESSING_WINDOW_SIZE="${MINERU_PROCESSING_WINDOW_SIZE:-128}"
DRY_RUN="${MINERU_VLM_REPLICA_DRY_RUN:-0}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

worker_pids=()
worker_urls=()

cleanup() {
  local pid
  for pid in "${worker_pids[@]:-}"; do
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local url="$1"
  local attempts="${MINERU_VLM_REPLICA_HEALTH_ATTEMPTS:-180}"
  for _ in $(seq 1 "$attempts"); do
    if curl -sf --noproxy '*' "$url/health" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "worker did not become healthy: $url" >&2
  return 1
}

router_args=(
  "$MINERU_ENV/bin/mineru-router"
  --host "$HOST"
  --port "$PORT"
  --local-gpus none
)

port="$BASE_PORT"
for gpu in "${GPU_LIST[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  [ -n "$gpu" ] || continue
  for replica in $(seq 1 "$REPLICAS_PER_GPU"); do
    worker_url="http://127.0.0.1:$port"
    worker_urls+=("$worker_url")
    router_args+=(--upstream-url "$worker_url")

    if [ "$DRY_RUN" = "1" ]; then
      echo "WORKER gpu=$gpu replica=$replica port=$port concurrency=$WORKER_CONCURRENCY"
    else
      export HOME="$ROOT"
      export TMPDIR="$ROOT/tmp"
      export PIP_CACHE_DIR="$ROOT/pip-cache"
      export MINERU_MODEL_SOURCE=local
      mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$ROOT/logs" "$ROOT/output/vlm-replicas"
      worker_output_root="$ROOT/output/vlm-replicas/gpu-${gpu}-replica-${replica}"
      worker_log="$ROOT/logs/vlm-fastapi-gpu-${gpu}-replica-${replica}.log"
      setsid env \
        HOME="$ROOT" \
        TMPDIR="$TMPDIR" \
        PIP_CACHE_DIR="$PIP_CACHE_DIR" \
        MINERU_MODEL_SOURCE=local \
        MINERU_API_OUTPUT_ROOT="$worker_output_root" \
        MINERU_API_MAX_CONCURRENT_REQUESTS="$WORKER_CONCURRENCY" \
        MINERU_PROCESSING_WINDOW_SIZE="$PROCESSING_WINDOW_SIZE" \
        MINERU_API_DISABLE_ACCESS_LOG=1 \
        MINERU_API_ENABLE_FASTAPI_DOCS=0 \
        CUDA_VISIBLE_DEVICES="$gpu" \
        "$MINERU_ENV/bin/python" -m mineru.cli.fast_api \
          --host "$WORKER_HOST" \
          --port "$port" \
          --enable-vlm-preload "$PRELOAD" \
          > "$worker_log" 2>&1 &
      worker_pids+=("$!")
      wait_for_health "$worker_url"
    fi
    port=$((port + 1))
  done
done

if [ "$HOST" = "0.0.0.0" ] || [ "$HOST" = "::" ]; then
  router_args+=(--allow-public-http-client)
fi

if [ "$DRY_RUN" = "1" ]; then
  printf 'ROUTER'
  printf ' %q' "${router_args[@]}"
  printf '\n'
  exit 0
fi

export HOME="$ROOT"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export MINERU_MODEL_SOURCE=local
export MINERU_API_OUTPUT_ROOT="$ROOT/output/vlm-replica-router"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MINERU_API_OUTPUT_ROOT" "$ROOT/logs"

"${router_args[@]}" &
router_pid="$!"
wait "$router_pid"
