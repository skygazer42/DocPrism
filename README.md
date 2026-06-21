# MinerU VLM Lab

Independent VLM-only PDF parsing lab. This project does not depend on DocPilot
and does not call MinerU pipeline.

## Runtime Shape

See `docs/production-requirements.md` for the 64-page production SLA and
acceptance checks.

See `docs/deployment-plan-20260621.md` for the current 6-GPU RTX 4090
deployment plan, benchmark results, Markdown/content validation, and GPU
shutdown procedure.

- Editable pages: extracted locally with PyMuPDF.
- Complex or image-heavy pages: sent to MinerU VLM through `mineru-router`.
- MinerU VLM workers: vLLM-backed hot workers on local GPUs, managed by
  `mineru-router`.
- GPU saturation mode: raise vLLM KV cache and page-level concurrency first,
  then add GPUs or upstream router nodes. Transformers is not the target path
  for production throughput in this lab.
- API: synchronous `/parse` plus async `/api/v1/jobs`.
- Storage: normalized page/block records and embeddings are persisted in SQLite.
- Observability: parse responses and async job status include phase timings for
  routing, VLM, embedding, storage, and total wall time.
- Embedding: default local deterministic hash embeddings for throughput tests;
  OpenAI-compatible HTTP embeddings can be enabled with environment variables.

## Deploy

```bash
cd /data/mineru-vlm-lab
./scripts/download_vlm_model.sh
./scripts/download_embedding_model.sh
python -m pip install -r requirements.txt
MINERU_ENV=/data/conda/envs/vllm-env \
MINERU_VLM_GPUS=4,5 \
MINERU_VLLM_GPU_MEMORY_UTILIZATION=0.90 \
MINERU_VLLM_MAX_MODEL_LEN=8192 \
nohup ./scripts/start_vlm_router_gpu45.sh > logs/vlm-router.log 2>&1 &

MINERU_VLM_BASE_URL=http://127.0.0.1:18100 \
EMBEDDING_PROVIDER=transformers \
EMBEDDING_DEVICE=cpu \
PRELOAD_EMBEDDING=false \
nohup ./scripts/start_orchestrator.sh > logs/orchestrator.log 2>&1 &
```

For the current 6-card RTX 4090 node, copy
`configs/production-6gpu-4090.env.example` to `production.env` and use
`scripts/start_direct_vlm_replicas.sh`. For 8-card or multi-node deployments,
copy one of the templates under `configs/` to `production.env`.

Health:

```bash
curl http://127.0.0.1:18100/health
curl http://127.0.0.1:18180/health
```

Parse:

```bash
curl -X POST http://127.0.0.1:18180/parse \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "persist=true"
```

Async parse:

```bash
curl -X POST http://127.0.0.1:18180/api/v1/jobs \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2"

curl http://127.0.0.1:18180/api/v1/jobs/<job_id>
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>/blocks
curl http://127.0.0.1:18180/api/v1/stats
```

## Environment

- `MINERU_VLM_GPUS=4,5`: GPUs used by `mineru-router`.
- `MINERU_VLLM_GPU_MEMORY_UTILIZATION=0.90`: vLLM memory target used by
  MinerU VLM workers. Higher values reserve more room for KV cache.
- `MINERU_VLLM_MAX_MODEL_LEN=8192`: cap request context so KV cache is sized
  for the PDF crop/page workload instead of an unnecessarily long context.
- `MINERU_VLLM_KV_CACHE_MEMORY_BYTES`: optional explicit per-GPU KV cache size.
- `MINERU_PROCESSING_WINDOW_SIZE=128`: MinerU processing window advertised by
  each worker.
- `MAX_CONCURRENT_VLM_PAGES=2`: page-level VLM fan-out from the orchestrator.
- `FAST_TEXT_CHUNK_CHARS=5000`: target chunk size for editable text pages before
  embedding. Use larger chunks for throughput-oriented page-level indexing and
  smaller chunks when retrieval granularity matters more than latency.
- `MINERU_VLM_LAB_DB_PATH=/data/mineru-vlm-lab/storage/mineru-vlm-lab.sqlite3`
- `EMBEDDING_PROVIDER=hash`: local deterministic embedding.
- `EMBEDDING_PROVIDER=transformers`: local Hugging Face/ModelScope model loaded
  from `EMBEDDING_MODEL_PATH`; `scripts/download_embedding_model.sh` writes
  this into `embedding.env`.
- `EMBEDDING_PROVIDER=openai_compatible`: use an embedding service with
  `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, and `EMBEDDING_MODEL`.
- `EMBEDDING_DEVICE=auto|cpu|cuda:4`: device for the local transformers model.
- `ENHANCEMENT_CONCURRENCY=2`: concurrent block-level crop VLM calls per
  enhancement worker.
- `ENHANCEMENT_LEASE_TIMEOUT_SECONDS=300`: lets another worker reclaim
  `processing` enhancement tasks whose lease is older than this. Keep it larger
  than `ENHANCEMENT_VLM_TIMEOUT_SECONDS`.
- `ENHANCEMENT_VLM_TIMEOUT_SECONDS=120`: per-crop VLM deadline; timed-out crops
  are marked failed so the worker pool is not pinned by one bad block.
- `ENHANCEMENT_TIMEOUT_CLEANUP_COMMAND=/data/mineru-vlm-lab/scripts/recycle_vlm_router.sh`:
  optional local cleanup hook after a crop timeout. Use it when the MinerU
  backend keeps timed-out requests in `processing`; the hook recycles only the
  VLM router, not the orchestrator.
- `ENHANCEMENT_TIMEOUT_CLEANUP_COOLDOWN_SECONDS=90`: minimum gap between
  cleanup runs inside one worker process.
- `ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH=640`: default VLM crop width for table
  enhancement tasks.
- `ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH=768`: per-task override used only
  for scan/noisy/large complex tables.
- `DEFERRED_TABLE_MIN_COMPLEX_BLOCKS=5`: editable caption tables below this
  complexity stay on the native PyMuPDF text path instead of entering optional
  VLM.

Enhancement queue:

```bash
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>/enhancements
curl -X POST http://127.0.0.1:18180/api/v1/enhancements/claim \
  -H "Content-Type: application/json" \
  -d '{"limit": 8, "worker_id": "vlm-table-worker-1"}'
curl -X POST http://127.0.0.1:18180/api/v1/enhancements/<task_id>/complete \
  -H "Content-Type: application/json" \
  -d '{"worker_id": "vlm-table-worker-1", "result": {"markdown": "| A | B |"}}'
```

Editable text-dense table candidates remain on the PyMuPDF fast path. The
service emits queued table enhancement tasks with PNG crop paths for downstream
VLM workers instead of blocking the main parse response.

Run the built-in crop worker:

```bash
python scripts/run_enhancement_worker.py --once --limit 2 --concurrency 2 --lease-timeout-seconds 300 --vlm-timeout-seconds 120
nohup ./scripts/start_enhancement_worker.sh > logs/enhancement-worker.log 2>&1 &
```

Recycle only the local VLM router if the backend keeps stale in-flight tasks:

```bash
./scripts/recycle_vlm_router.sh
```

## Test

```bash
python -m pytest -q
```

Benchmark:

```bash
python scripts/bench_jobs.py --pdf /path/to/file.pdf --jobs 8 --concurrency 8
/data/mineru32-bench/env/bin/python scripts/bench_parse_and_enhancements.py \
  --base-url http://127.0.0.1:18180 \
  --pdf /path/to/file.pdf \
  --wait-enhancements all \
  --timeout 360 \
  --poll-interval 0.05
```
