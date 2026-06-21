#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/conda/envs/vllm-env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_helpers.sh"
load_env_preserving_overrides "$ROOT/production.env" \
  MINERU_ENV \
  MINERU_VLM_ROUTER_HOST \
  MINERU_VLM_ROUTER_PORT \
  MINERU_VLM_GPUS \
  MINERU_VLM_WORKER_HOST \
  MINERU_VLM_PRELOAD \
  MINERU_VLM_ROUTER_DRY_RUN \
  MINERU_VLLM_GPU_MEMORY_UTILIZATION \
  MINERU_VLLM_MAX_MODEL_LEN \
  MINERU_VLLM_KV_CACHE_MEMORY_BYTES \
  MINERU_VLLM_ENFORCE_EAGER \
  MINERU_VLM_WORKER_CONCURRENCY \
  MINERU_API_MAX_CONCURRENT_REQUESTS \
  MINERU_PROCESSING_WINDOW_SIZE \
  MINERU_VLM_UPSTREAM_URLS \
  MINERU_VLM_EXTRA_ARGS

HOST="${MINERU_VLM_ROUTER_HOST:-127.0.0.1}"
PORT="${MINERU_VLM_ROUTER_PORT:-18100}"
GPUS="${MINERU_VLM_GPUS:-4,5}"
WORKER_HOST="${MINERU_VLM_WORKER_HOST:-127.0.0.1}"
PRELOAD="${MINERU_VLM_PRELOAD:-true}"
DRY_RUN="${MINERU_VLM_ROUTER_DRY_RUN:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${MINERU_VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_MAX_MODEL_LEN="${MINERU_VLLM_MAX_MODEL_LEN:-}"
VLLM_KV_CACHE_MEMORY_BYTES="${MINERU_VLLM_KV_CACHE_MEMORY_BYTES:-}"
VLLM_ENFORCE_EAGER="${MINERU_VLLM_ENFORCE_EAGER:-}"
WORKER_CONCURRENCY="${MINERU_VLM_WORKER_CONCURRENCY:-${MINERU_API_MAX_CONCURRENT_REQUESTS:-3}}"
PROCESSING_WINDOW_SIZE="${MINERU_PROCESSING_WINDOW_SIZE:-64}"

export HOME="$ROOT"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export MINERU_MODEL_SOURCE=local
export MINERU_API_OUTPUT_ROOT="$ROOT/output/mineru-vlm-router"
export MINERU_API_MAX_CONCURRENT_REQUESTS="$WORKER_CONCURRENCY"
export MINERU_PROCESSING_WINDOW_SIZE="$PROCESSING_WINDOW_SIZE"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MINERU_API_OUTPUT_ROOT" "$ROOT/logs"

args=(
  "$MINERU_ENV/bin/mineru-router"
  --host "$HOST"
  --port "$PORT"
  --local-gpus "$GPUS"
  --worker-host "$WORKER_HOST"
  --enable-vlm-preload "$PRELOAD"
)

for upstream in ${MINERU_VLM_UPSTREAM_URLS:-}; do
  args+=(--upstream-url "$upstream")
done

if [ "$HOST" = "0.0.0.0" ] || [ "$HOST" = "::" ]; then
  args+=(--allow-public-http-client)
fi

if [ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]; then
  args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

if [ -n "$VLLM_MAX_MODEL_LEN" ]; then
  args+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi

if [ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]; then
  args+=(--kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
fi

if [[ "$VLLM_ENFORCE_EAGER" =~ ^(1|true|yes|on)$ ]]; then
  args+=(--enforce-eager)
fi

for extra_arg in ${MINERU_VLM_EXTRA_ARGS:-}; do
  args+=("$extra_arg")
done

if [[ "$DRY_RUN" =~ ^(1|true|yes|on)$ ]]; then
  printf 'CMD'
  for arg in "${args[@]}"; do
    printf ' %s' "$arg"
  done
  printf '\n'
  exit 0
fi

if [ -f "$MINERU_ENV/bin/activate" ]; then
  source "$MINERU_ENV/bin/activate"
fi
exec "${args[@]}"
