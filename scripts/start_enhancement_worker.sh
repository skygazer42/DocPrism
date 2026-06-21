#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/mineru32-bench/env}"

if [ -f "$ROOT/production.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/production.env"
  set +a
fi

source "$MINERU_ENV/bin/activate"
cd "$ROOT"

args=(
  --orchestrator-base-url "${ENHANCEMENT_ORCHESTRATOR_BASE_URL:-http://127.0.0.1:18180}" \
  --mineru-vlm-base-url "${ENHANCEMENT_MINERU_VLM_BASE_URL:-http://127.0.0.1:18100}" \
  --worker-id "${ENHANCEMENT_WORKER_ID:-table-vlm-worker-1}" \
  --limit "${ENHANCEMENT_CLAIM_LIMIT:-2}" \
  --concurrency "${ENHANCEMENT_CONCURRENCY:-2}" \
  --vlm-timeout-seconds "${ENHANCEMENT_VLM_TIMEOUT_SECONDS:-120}" \
  --timeout-cleanup-command "${ENHANCEMENT_TIMEOUT_CLEANUP_COMMAND:-}" \
  --timeout-cleanup-cooldown-seconds "${ENHANCEMENT_TIMEOUT_CLEANUP_COOLDOWN_SECONDS:-60}" \
  --timeout-cleanup-command-timeout-seconds "${ENHANCEMENT_TIMEOUT_CLEANUP_COMMAND_TIMEOUT_SECONDS:-180}" \
  --scratch-root "${ENHANCEMENT_SCRATCH_ROOT:-$ROOT/work/enhancement-worker}" \
  --interval-seconds "${ENHANCEMENT_INTERVAL_SECONDS:-2}"
)

if [ -n "${ENHANCEMENT_LEASE_TIMEOUT_SECONDS:-}" ]; then
  args+=(--lease-timeout-seconds "$ENHANCEMENT_LEASE_TIMEOUT_SECONDS")
fi

if [ -n "${ENHANCEMENT_KIND:-}" ]; then
  args+=(--kind "$ENHANCEMENT_KIND")
fi

exec "$MINERU_ENV/bin/python" "$ROOT/scripts/run_enhancement_worker.py" "${args[@]}"
