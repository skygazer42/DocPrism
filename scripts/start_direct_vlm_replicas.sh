#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/conda/envs/vllm-env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_helpers.sh"
load_env_preserving_overrides "$ROOT/production.env" \
  MINERU_ENV \
  MINERU_VLM_GPUS \
  MINERU_VLM_MODEL_PATH \
  DIRECT_VLM_GPUS \
  DIRECT_VLM_REPLICAS_PER_GPU \
  DIRECT_VLM_CLAIM_LIMIT \
  DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION \
  DIRECT_VLM_MAX_MODEL_LEN \
  DIRECT_VLM_MAX_IMAGE_WIDTH \
  DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH \
  DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH \
  DIRECT_VLM_INTERVAL_SECONDS \
  DIRECT_VLM_ORCHESTRATOR_BASE_URL \
  DIRECT_VLM_WORKER_ID_PREFIX \
  DIRECT_VLM_BACKEND \
  DIRECT_VLM_KIND \
  DIRECT_VLM_REPLICA_STAGGER_SECONDS \
  DIRECT_VLM_REPLICA_DRY_RUN \
  DIRECT_VLM_MAX_NUM_BATCHED_TOKENS \
  DIRECT_VLM_MAX_NUM_SEQS \
  DIRECT_VLM_KV_CACHE_DTYPE \
  DIRECT_VLM_COMPILATION_CONFIG

cd "$ROOT"
mkdir -p "$ROOT/logs"

GPUS="${DIRECT_VLM_GPUS:-${MINERU_VLM_GPUS:-4,5}}"
REPLICAS_PER_GPU="${DIRECT_VLM_REPLICAS_PER_GPU:-2}"
CLAIM_LIMIT="${DIRECT_VLM_CLAIM_LIMIT:-2}"
GPU_MEMORY_UTILIZATION="${DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION:-0.42}"
MAX_MODEL_LEN="${DIRECT_VLM_MAX_MODEL_LEN:-8192}"
MAX_IMAGE_WIDTH="${DIRECT_VLM_MAX_IMAGE_WIDTH:-512}"
MAX_TABLE_IMAGE_WIDTH="${DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH:-1024}"
MAX_PAGE_IMAGE_WIDTH="${DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH:-1024}"
INTERVAL_SECONDS="${DIRECT_VLM_INTERVAL_SECONDS:-0.1}"
ORCHESTRATOR_BASE_URL="${DIRECT_VLM_ORCHESTRATOR_BASE_URL:-http://127.0.0.1:18180}"
WORKER_ID_PREFIX="${DIRECT_VLM_WORKER_ID_PREFIX:-direct-vlm}"
MODEL_PATH="${MINERU_VLM_MODEL_PATH:-$ROOT/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-Pro-2605-1___2B}"
BACKEND="${DIRECT_VLM_BACKEND:-vllm-async-engine}"
KIND_VALUE="${DIRECT_VLM_KIND-}"
STAGGER_SECONDS="${DIRECT_VLM_REPLICA_STAGGER_SECONDS:-8}"
DRY_RUN="${DIRECT_VLM_REPLICA_DRY_RUN:-0}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

build_args() {
  local worker_id="$1"
  local -n out_args="$2"
  out_args=(
    "$ROOT/scripts/run_direct_vlm_worker.py"
    --orchestrator-base-url "$ORCHESTRATOR_BASE_URL"
    --worker-id "$worker_id"
    --limit "$CLAIM_LIMIT"
    --kind "$KIND_VALUE"
    --model-path "$MODEL_PATH"
    --backend "$BACKEND"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-image-width "$MAX_IMAGE_WIDTH"
    --max-table-image-width "$MAX_TABLE_IMAGE_WIDTH"
    --max-page-image-width "$MAX_PAGE_IMAGE_WIDTH"
    --interval-seconds "$INTERVAL_SECONDS"
  )
  if [ -n "${DIRECT_VLM_MAX_NUM_BATCHED_TOKENS:-}" ]; then
    out_args+=(--max-num-batched-tokens "$DIRECT_VLM_MAX_NUM_BATCHED_TOKENS")
  fi
  if [ -n "${DIRECT_VLM_MAX_NUM_SEQS:-}" ]; then
    out_args+=(--max-num-seqs "$DIRECT_VLM_MAX_NUM_SEQS")
  fi
  if [ -n "${DIRECT_VLM_KV_CACHE_DTYPE:-}" ]; then
    out_args+=(--kv-cache-dtype "$DIRECT_VLM_KV_CACHE_DTYPE")
  fi
  if [ -n "${DIRECT_VLM_COMPILATION_CONFIG:-}" ]; then
    out_args+=(--compilation-config "$DIRECT_VLM_COMPILATION_CONFIG")
  fi
}

for gpu in "${GPU_LIST[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  [ -n "$gpu" ] || continue
  for replica in $(seq 1 "$REPLICAS_PER_GPU"); do
    worker_id="${WORKER_ID_PREFIX}-gpu${gpu}-${replica}"
    args=()
    build_args "$worker_id" args
    if [ "$DRY_RUN" = "1" ]; then
      echo "DIRECT_WORKER gpu=$gpu replica=$replica worker_id=$worker_id limit=$CLAIM_LIMIT gpu_memory_utilization=$GPU_MEMORY_UTILIZATION max_model_len=$MAX_MODEL_LEN"
      printf 'CMD CUDA_VISIBLE_DEVICES=%q %q' "$gpu" "$MINERU_ENV/bin/python"
      printf ' %q' "${args[@]}"
      printf '\n'
      continue
    fi
    log="$ROOT/logs/direct-vlm-${worker_id}.log"
    nohup env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      MINERU_VLM_LAB_ROOT="$ROOT" \
      "$MINERU_ENV/bin/python" "${args[@]}" \
      > "$log" 2>&1 &
    echo "DIRECT_WORKER gpu=$gpu replica=$replica worker_id=$worker_id pid=$! log=$log"
    sleep "$STAGGER_SECONDS"
  done
done
