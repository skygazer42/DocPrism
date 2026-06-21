#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/mineru32-bench/env}"

source "$MINERU_ENV/bin/activate"
export HOME="$ROOT"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export MINERU_MODEL_SOURCE=modelscope
mkdir -p "$ROOT" "$TMPDIR" "$PIP_CACHE_DIR" "$ROOT/logs"

"$MINERU_ENV/bin/mineru-models-download" -s modelscope -m vlm
