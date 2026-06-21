# VLM-Only Production Architecture

## Goal

The production target is 64 pages in less than 10 seconds including downstream
storage and embedding. A single full MinerU pipeline request cannot meet this
target. The service must split work by page and block, avoid unnecessary OCR,
and keep GPU workers warm.

## Routing

1. Open the PDF with PyMuPDF.
2. For each page, collect text density, block count, image count, image area,
   and simple table hints.
3. Editable text-dense pages go through PyMuPDF fast path.
4. Scanned, image-heavy, low-text, or low-confidence pages go to VLM.
5. Editable text-dense table candidates stay on the fast path; production table
   enhancement should crop table blocks and call VLM at block level, not submit
   an otherwise editable whole page.
6. VLM tasks are submitted as page-level jobs to a warm multi-GPU worker pool.

## Worker Model

- `mineru-router` starts one vLLM-backed VLM worker per GPU.
- The production target is vLLM hot service, not transformers. Increase
  `MINERU_VLLM_GPU_MEMORY_UTILIZATION` or set
  `MINERU_VLLM_KV_CACHE_MEMORY_BYTES` to reserve enough KV cache, and keep
  `MINERU_VLLM_MAX_MODEL_LEN` aligned with page/crop workloads.
- Multi-node deployments should aggregate warm local GPU routers through
  upstream URLs instead of cold-starting per request.
- The orchestrator does page-level scheduling rather than submitting one whole
  PDF to one GPU.
- The VLM path uses MinerU `vlm-auto-engine` backed by vLLM. It does not call
  MinerU pipeline mode.

## Async Consumers

The parse path emits normalized page/block records. The current implementation
stores those records in SQLite and computes embeddings in batches. The default
embedding provider is a deterministic local hash provider for smoke tests. The
production path should keep embedding off the VLM GPUs, either with CPU
batching, a separate embedding GPU pool, or an OpenAI-compatible embedding
service via environment variables.

## API Contract

1. `POST /api/v1/jobs` uploads a PDF and returns `job_id`, `status_url`, and
   `blocks_url`.
2. `GET /api/v1/jobs/{job_id}` returns status, page counts, route counts, block
   count, embedding count, elapsed time, and error details.
3. `GET /api/v1/jobs/{job_id}/blocks` returns normalized blocks joined with
   embedding metadata.
4. `GET /api/v1/jobs/{job_id}/enhancements` returns queued table/image
   enhancement tasks with crop paths.
5. `GET /api/v1/stats` returns DB-backed runtime counters for jobs, pages,
   blocks, embeddings, and enhancement backlog states.
6. `POST /api/v1/enhancements/claim` leases queued enhancement tasks to a
   worker, and `/complete` or `/fail` records the worker result. The claim API
   accepts `lease_timeout_seconds` so another worker can recover stale
   `processing` tasks if the original worker exits before writing a terminal
   status. The bundled `scripts/run_enhancement_worker.py` worker wraps PNG
   crops as one-page PDFs, submits them to MinerU VLM, parses the returned zip,
   and completes the task with markdown/content-list payloads. Each crop call
   has an independent deadline so a pathological VLM block is failed and
   recorded instead of pinning the consumer pool indefinitely. If the MinerU
   backend continues the timed-out request internally, the worker can run a
   configured cleanup command such as `scripts/recycle_vlm_router.sh` with a
   cooldown to release local GPU workers without restarting the orchestrator.
7. `POST /parse` remains available for synchronous experiments and can persist
   results when `persist=true`.

## Storage Model

- `jobs`: lifecycle, counts, elapsed time, source file path.
- `pages`: per-page route decision and routing signals.
- `blocks`: normalized text/table/markdown units from PyMuPDF or VLM output.
- `embeddings`: embedding provider/model/dim and vector payload per block.
- `enhancement_tasks`: queued nonblocking VLM enhancement work such as editable
  table-candidate crops.

## Throughput Notes

For editable PDFs, the critical path is PyMuPDF extraction, block normalization,
batched embedding, and SQLite writes. For complex pages, the VLM router stays
warm on multiple GPUs and the orchestrator submits page-level VLM requests with
bounded concurrency. This keeps OCR/VLM off the common editable-page path while
still preserving a single API and storage contract.

Live GPU4/GPU5 vLLM tests showed that `gpu_memory_utilization=0.90` keeps both
cards hot and raises the per-card KV cache to 17.82 GiB for the current
MinerU2.5 VLM model. That fixes the earlier low-VRAM symptom: the old process
was using a non-vLLM environment and falling back to transformers.
