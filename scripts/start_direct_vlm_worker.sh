#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/conda/envs/vllm-env}"

if [ -f "$ROOT/production.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/production.env"
  set +a
fi

cd "$ROOT"

DIRECT_VLM_KIND_VALUE="${DIRECT_VLM_KIND-}"

args=(
  "$ROOT/scripts/run_direct_vlm_worker.py"
  --orchestrator-base-url "${DIRECT_VLM_ORCHESTRATOR_BASE_URL:-http://127.0.0.1:18180}"
  --worker-id "${DIRECT_VLM_WORKER_ID:-direct-vlm-worker-1}"
  --limit "${DIRECT_VLM_CLAIM_LIMIT:-2}"
  --kind "$DIRECT_VLM_KIND_VALUE"
  --model-path "${MINERU_VLM_MODEL_PATH:-$ROOT/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-Pro-2605-1___2B}"
  --backend "${DIRECT_VLM_BACKEND:-vllm-async-engine}"
  --gpu-memory-utilization "${DIRECT_VLM_GPU_MEMORY_UTILIZATION:-0.9}"
  --max-model-len "${DIRECT_VLM_MAX_MODEL_LEN:-8192}"
  --max-image-width "${DIRECT_VLM_MAX_IMAGE_WIDTH:-512}"
  --interval-seconds "${DIRECT_VLM_INTERVAL_SECONDS:-0.1}"
)

if [ -n "${DIRECT_VLM_MAX_NUM_BATCHED_TOKENS:-}" ]; then
  args+=(--max-num-batched-tokens "$DIRECT_VLM_MAX_NUM_BATCHED_TOKENS")
fi
if [ -n "${DIRECT_VLM_MAX_NUM_SEQS:-}" ]; then
  args+=(--max-num-seqs "$DIRECT_VLM_MAX_NUM_SEQS")
fi
if [ -n "${DIRECT_VLM_KV_CACHE_DTYPE:-}" ]; then
  args+=(--kv-cache-dtype "$DIRECT_VLM_KV_CACHE_DTYPE")
fi
if [ -n "${DIRECT_VLM_COMPILATION_CONFIG:-}" ]; then
  args+=(--compilation-config "$DIRECT_VLM_COMPILATION_CONFIG")
fi

exec "$MINERU_ENV/bin/python" "${args[@]}"
