#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINERU_VLM_LAB_ROOT:-/data/mineru-vlm-lab}"
MINERU_ENV="${MINERU_ENV:-/data/mineru32-bench/env}"
MODEL_ID="${EMBEDDING_MODEL_ID:-BAAI/bge-small-zh-v1.5}"
DEVICE="${EMBEDDING_DEVICE:-auto}"

source "$MINERU_ENV/bin/activate"
export HOME="$ROOT"
export TMPDIR="$ROOT/tmp"
export MODELSCOPE_CACHE="$ROOT/.cache/modelscope"
mkdir -p "$TMPDIR" "$MODELSCOPE_CACHE" "$ROOT/logs"

MODEL_PATH="$("$MINERU_ENV/bin/python" - <<PY
import contextlib
import sys
from modelscope import snapshot_download
with contextlib.redirect_stdout(sys.stderr):
    path = snapshot_download("${MODEL_ID}", cache_dir="${MODELSCOPE_CACHE}")
print(path)
PY
)"

cat > "$ROOT/embedding.env" <<EOF
EMBEDDING_PROVIDER=transformers
EMBEDDING_MODEL_PATH=$MODEL_PATH
EMBEDDING_MODEL=$MODEL_ID
EMBEDDING_DIM=0
EMBEDDING_DEVICE=$DEVICE
EMBEDDING_MAX_LENGTH=512
EMBEDDING_BATCH_SIZE=64
MAX_CONCURRENT_EMBEDDING_BATCHES=1
PRELOAD_EMBEDDING=true
EOF

echo "embedding_model_path=$MODEL_PATH"
echo "embedding_env=$ROOT/embedding.env"
