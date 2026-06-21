#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"

export MINERU_VLM_LAB_ROOT="$ROOT"
export MINERU_VLM_GPUS="${MINERU_VLM_GPUS:-4,5}"
export MINERU_VLM_ROUTER_HOST="${MINERU_VLM_ROUTER_HOST:-127.0.0.1}"
export MINERU_VLM_ROUTER_PORT="${MINERU_VLM_ROUTER_PORT:-18100}"
export MINERU_VLM_PRELOAD="${MINERU_VLM_PRELOAD:-true}"

exec "$ROOT/scripts/start_vlm_router.sh"
