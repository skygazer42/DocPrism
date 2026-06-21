# Production Requirements

## Target

Process a 64-page PDF in less than 10 seconds, including:

- PDF intake and page routing.
- Editable-page extraction through PyMuPDF.
- VLM calls for scan-heavy, image-heavy, table/formula/image blocks, or
  low-confidence pages.
- Normalized block persistence.
- Embedding generation and persistence.

The target assumes a realistic production mix where most PDFs have an editable
text layer. VLM-heavy scanned documents have a separate capacity target and must
be measured by VLM page/crop throughput.

## Non-Goals

- Do not depend on DocPilot.
- Do not call MinerU pipeline mode for the main path.
- Do not use transformers as the production VLM runtime.
- Do not cold-start VLM per request.

## Required Runtime Shape

1. Keep MinerU VLM as a warm vLLM service through `mineru-router`.
2. Download and stage MinerU VLM weights from ModelScope.
3. Route each page before OCR/VLM:
   - `fast_pymupdf`: editable, text-dense pages with enough native text.
   - `vlm_page`: scanned, image-heavy, or low-confidence pages.
   - `enhancement_task`: editable pages with table/image candidates that can
     return quickly while block-level VLM workers process crops.
4. Use bounded page-level VLM fan-out, sized to the router's warm GPU capacity.
5. Use block-level consumers for table/image/formula enhancement crops.
6. Keep embedding off VLM GPUs or route it to a separate embedding service.
7. Persist phase timings so benchmarks can explain whether time is spent in
   routing, VLM, embedding, or storage.
8. Chunk editable text before embedding. For throughput-oriented production
   indexing, use `FAST_TEXT_CHUNK_CHARS=5000` as the current baseline; lower it
   only when retrieval granularity is more important than the 64-page latency
   SLA.

## Scaling Rules

- Prefer route optimization before adding GPUs. Editable pages should not touch
  OCR/VLM.
- Prefer vLLM KV cache tuning before adding more VLM processes per card.
- Use `MINERU_VLLM_GPU_MEMORY_UTILIZATION` for coarse KV cache sizing.
- Use `MINERU_VLLM_KV_CACHE_MEMORY_BYTES` when profiling shows that a fixed KV
  cache budget is better than utilization-based sizing.
- Use `MINERU_VLLM_MAX_MODEL_LEN` to avoid over-sizing context for page/crop
  workloads.
- Scale from page-level concurrency to multi-GPU, then to multi-node router
  aggregation.
- Keep table crop width adaptive. The current default is 640 px for normal
  editable table enhancement tasks, with a per-task 768 px override only for
  scan, noisy native tables, or large/complex tables.
- Keep optional table fan-out bounded. Ordinary editable caption tables stay on
  the native PyMuPDF text path; only complex or low-confidence table regions
  should enter the VLM crop queue.

## Acceptance Checks

For every benchmark run, record:

- PDF page count.
- Fast page count.
- VLM page count.
- Queued enhancement task count.
- Block count.
- Embedding count.
- Total elapsed seconds.
- `routing_seconds`, `vlm_seconds`, `embedding_seconds`, `storage_seconds`.
- Router health, per-GPU completed/failed counts, and GPU memory usage.

The 64-page production SLA only passes when `elapsed_seconds < 10` and
`embedding_count == block_count` for the persisted job.

The current editable-PDF hot baseline on GPU4/GPU5 with CPU embedding and
`FAST_TEXT_CHUNK_CHARS=5000` is:

- 64 pages.
- 64 fast PyMuPDF pages.
- 0 VLM pages.
- 64 blocks.
- 64 embeddings.
- 6.308 seconds total.
- 0.737 seconds routing, 5.520 seconds embedding, 0.046 seconds storage.

## Current Evidence

On the 6-card RTX 4090 node, the current best direct-vLLM profile is:

- GPUs `0,1,2,3,4,5`.
- `DIRECT_VLM_REPLICAS_PER_GPU=1`.
- `DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION=0.90`.
- `DIRECT_VLM_CLAIM_LIMIT=2`.
- `DIRECT_VLM_MAX_MODEL_LEN=8192`.
- `DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH=640`.
- `DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH=1024`.

Measured before the adaptive fan-out change:

- ResNet 12 pages with all optional VLM enhancements waited: 3.489 seconds,
  8 optional VLM tasks, 0 required tasks.
- ATLAS 64 pages with all optional VLM enhancements waited: 5.891 seconds,
  parse wall 2.749 seconds, total throughput 10.863 pages/second, parse-only
  throughput 23.283 pages/second, 13 optional VLM tasks, 0 required tasks.

The 26 pages/second target should be treated as production parse throughput
under a realistic editable-PDF mix plus selective asynchronous VLM. It is not a
credible target for full-page/full-block synchronous VLM on every page.

## Source Notes

- MinerU advanced CLI parameters allow passing supported vLLM arguments through
  MinerU CLI/server entrypoints.
- vLLM tuning guidance states that increasing `gpu_memory_utilization` gives
  more memory for KV cache.
- vLLM cache config supports `kv_cache_memory_bytes` as an explicit KV cache
  budget.
