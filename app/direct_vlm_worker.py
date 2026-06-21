from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


DEFAULT_MODEL_PATH = (
    "/data/mineru-vlm-lab/.cache/modelscope/hub/models/"
    "OpenDataLab/MinerU2___5-Pro-2605-1___2B"
)


def load_mineru_predictor(
    *,
    model_path: str,
    backend: str,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
    compilation_config: str | None = None,
) -> Any:
    from mineru.backend.vlm.vlm_analyze import ModelSingleton

    kwargs = build_vllm_engine_kwargs(
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        compilation_config=compilation_config,
    )
    return ModelSingleton().get_model(backend, model_path, None, **kwargs)


def build_vllm_engine_kwargs(
    *,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
    compilation_config: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if compilation_config:
        kwargs["compilation_config"] = compilation_config
    return kwargs


def load_crop_image(crop_path: Path, *, max_image_width: int | None) -> Image.Image:
    with Image.open(crop_path) as image:
        loaded = image.convert("RGB")
    if max_image_width and max_image_width > 0 and loaded.width > max_image_width:
        ratio = max_image_width / loaded.width
        size = (max_image_width, max(1, round(loaded.height * ratio)))
        loaded = loaded.resize(size, Image.Resampling.LANCZOS)
    loaded.load()
    return loaded


def resolve_task_max_image_width(
    task: dict[str, Any],
    *,
    max_image_width: int | None,
    max_table_image_width: int | None,
    max_page_image_width: int | None,
) -> int | None:
    metadata = task.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("page_vlm") and max_page_image_width:
        return max_page_image_width
    if isinstance(metadata, dict):
        task_width = _positive_int(metadata.get("max_image_width"))
        if task_width:
            return task_width
    if task.get("kind") == "table_candidate" and max_table_image_width:
        return max_table_image_width
    return max_image_width


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__iter__"):
        try:
            return [_jsonable(item) for item in value]
        except TypeError:
            pass
    return str(value)


def _looks_like_content_item(value: dict[str, Any]) -> bool:
    return bool({"type", "content", "text", "html", "markdown", "latex"} & set(value))


def flatten_content_items(value: Any) -> list[dict[str, Any]]:
    value = _jsonable(value)
    if isinstance(value, dict):
        if _looks_like_content_item(value):
            return [value]
        items: list[dict[str, Any]] = []
        for child in value.values():
            items.extend(flatten_content_items(child))
        return items
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for child in value:
            items.extend(flatten_content_items(child))
        return items
    return []


def markdown_from_items(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        for key in ("markdown", "content", "text", "html", "latex"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
                break
    return "\n\n".join(parts)


def split_batch_results(raw_results: Any, task_count: int) -> list[Any]:
    normalized = _jsonable(raw_results)
    if task_count == 1:
        return [normalized]
    if isinstance(normalized, list) and len(normalized) == task_count:
        return normalized
    raise ValueError(f"VLM returned {type(normalized).__name__} for {task_count} tasks")


def build_task_result(
    *,
    task: dict[str, Any],
    raw_result: Any,
    image_size: tuple[int, int],
    batch_elapsed_seconds: float,
    max_image_width: int | None,
) -> dict[str, Any]:
    content_items = flatten_content_items(raw_result)
    return {
        "backend": "mineru_direct_pil",
        "elapsed_seconds": batch_elapsed_seconds,
        "crop_path": task["crop_path"],
        "image_size": [image_size[0], image_size[1]],
        "max_image_width": max_image_width,
        "raw_result": _jsonable(raw_result),
        "content_items": content_items,
        "markdown": markdown_from_items(content_items),
    }


async def run_direct_worker_once(
    *,
    orchestrator_base_url: str,
    worker_id: str,
    limit: int,
    kind: str | None = "image_candidate",
    lease_timeout_seconds: float | None = None,
    job_id: str | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    backend: str = "vllm-async-engine",
    gpu_memory_utilization: float | None = 0.9,
    max_model_len: int | None = 8192,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
    compilation_config: str | None = None,
    max_image_width: int | None = 512,
    max_table_image_width: int | None = 1024,
    max_page_image_width: int | None = 1024,
    image_analysis: bool = True,
    predictor: Any | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, int]:
    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=30.0)
        client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    assert client is not None
    try:
        claim_payload: dict[str, Any] = {"limit": limit, "worker_id": worker_id}
        if job_id:
            claim_payload["job_id"] = job_id
        if kind:
            claim_payload["kind"] = kind
        if lease_timeout_seconds is not None:
            claim_payload["lease_timeout_seconds"] = lease_timeout_seconds
        try:
            claim_response = await client.post(
                f"{orchestrator_base_url.rstrip('/')}/api/v1/enhancements/claim",
                json=claim_payload,
            )
            claim_response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"claimed": 0, "completed": 0, "failed": 0, "claim_errors": 1}
        tasks = claim_response.json().get("tasks", [])
        if not tasks:
            return {"claimed": 0, "completed": 0, "failed": 0}

        if predictor is None:
            predictor = load_mineru_predictor(
                model_path=model_path,
                backend=backend,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                kv_cache_dtype=kv_cache_dtype,
                compilation_config=compilation_config,
            )

        task_max_image_widths = [
            resolve_task_max_image_width(
                task,
                max_image_width=max_image_width,
                max_table_image_width=max_table_image_width,
                max_page_image_width=max_page_image_width,
            )
            for task in tasks
        ]
        images = [
            load_crop_image(Path(task["crop_path"]), max_image_width=task_width)
            for task, task_width in zip(tasks, task_max_image_widths)
        ]
        image_sizes = [image.size for image in images]
        started = time.monotonic()
        try:
            raw_results = await predictor.aio_batch_two_step_extract(
                images=images,
                image_analysis=image_analysis,
            )
            batch_elapsed = round(time.monotonic() - started, 3)
            split_results = split_batch_results(raw_results, len(tasks))
        except Exception as exc:
            await asyncio.gather(
                *(
                    _fail_task(
                        client=client,
                        orchestrator_base_url=orchestrator_base_url,
                        worker_id=worker_id,
                        task_id=task["task_id"],
                        error=str(exc),
                    )
                    for task in tasks
                ),
                return_exceptions=True,
            )
            return {"claimed": len(tasks), "completed": 0, "failed": len(tasks)}

        completed = 0
        failed = 0
        for task, image_size, raw_result, task_width in zip(tasks, image_sizes, split_results, task_max_image_widths):
            result = build_task_result(
                task=task,
                raw_result=raw_result,
                image_size=image_size,
                batch_elapsed_seconds=batch_elapsed,
                max_image_width=task_width,
            )
            try:
                complete_response = await client.post(
                    f"{orchestrator_base_url.rstrip('/')}/api/v1/enhancements/{task['task_id']}/complete",
                    json={"worker_id": worker_id, "result": result},
                )
                complete_response.raise_for_status()
                completed += 1
            except Exception as exc:
                failed += 1
                try:
                    await _fail_task(
                        client=client,
                        orchestrator_base_url=orchestrator_base_url,
                        worker_id=worker_id,
                        task_id=task["task_id"],
                        error=str(exc),
                    )
                except httpx.HTTPError:
                    pass
        return {"claimed": len(tasks), "completed": completed, "failed": failed}
    finally:
        if owns_client:
            await client.aclose()


async def _fail_task(
    *,
    client: httpx.AsyncClient,
    orchestrator_base_url: str,
    worker_id: str,
    task_id: str,
    error: str,
) -> None:
    fail_response = await client.post(
        f"{orchestrator_base_url.rstrip('/')}/api/v1/enhancements/{task_id}/fail",
        json={"worker_id": worker_id, "error": error[:4000]},
    )
    fail_response.raise_for_status()


async def run_direct_worker_loop(
    *,
    orchestrator_base_url: str,
    worker_id: str,
    limit: int,
    interval_seconds: float,
    once: bool,
    kind: str | None = "image_candidate",
    lease_timeout_seconds: float | None = None,
    job_id: str | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    backend: str = "vllm-async-engine",
    gpu_memory_utilization: float | None = 0.9,
    max_model_len: int | None = 8192,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
    compilation_config: str | None = None,
    max_image_width: int | None = 512,
    max_table_image_width: int | None = 1024,
    max_page_image_width: int | None = 1024,
    image_analysis: bool = True,
) -> None:
    predictor = load_mineru_predictor(
        model_path=model_path,
        backend=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        compilation_config=compilation_config,
    )
    while True:
        try:
            summary = await run_direct_worker_once(
                orchestrator_base_url=orchestrator_base_url,
                worker_id=worker_id,
                limit=limit,
                kind=kind,
                lease_timeout_seconds=lease_timeout_seconds,
                job_id=job_id,
                model_path=model_path,
                backend=backend,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                kv_cache_dtype=kv_cache_dtype,
                compilation_config=compilation_config,
                max_image_width=max_image_width,
                max_table_image_width=max_table_image_width,
                max_page_image_width=max_page_image_width,
                image_analysis=image_analysis,
                predictor=predictor,
            )
        except Exception as exc:
            summary = {"claimed": 0, "completed": 0, "failed": 0, "loop_errors": 1, "error": str(exc)}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if once:
            return
        await asyncio.sleep(interval_seconds)


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume image enhancement crops with direct MinerU VLM batches.")
    parser.add_argument("--orchestrator-base-url", default=os.getenv("DIRECT_VLM_ORCHESTRATOR_BASE_URL", "http://127.0.0.1:18180"))
    parser.add_argument("--worker-id", default=os.getenv("DIRECT_VLM_WORKER_ID", "direct-vlm-worker-1"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("DIRECT_VLM_CLAIM_LIMIT", "4")))
    parser.add_argument("--kind", default=os.getenv("DIRECT_VLM_KIND", "") or None)
    parser.add_argument("--lease-timeout-seconds", type=float, default=_optional_float_env("DIRECT_VLM_LEASE_TIMEOUT_SECONDS"))
    parser.add_argument("--job-id", default=os.getenv("DIRECT_VLM_JOB_ID") or None)
    parser.add_argument("--model-path", default=os.getenv("MINERU_VLM_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--backend", default=os.getenv("DIRECT_VLM_BACKEND", "vllm-async-engine"))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("DIRECT_VLM_GPU_MEMORY_UTILIZATION", "0.9")),
    )
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("DIRECT_VLM_MAX_MODEL_LEN", "8192")))
    parser.add_argument("--max-num-batched-tokens", type=int, default=_optional_int_env("DIRECT_VLM_MAX_NUM_BATCHED_TOKENS"))
    parser.add_argument("--max-num-seqs", type=int, default=_optional_int_env("DIRECT_VLM_MAX_NUM_SEQS"))
    parser.add_argument("--kv-cache-dtype", default=os.getenv("DIRECT_VLM_KV_CACHE_DTYPE") or None)
    parser.add_argument("--compilation-config", default=os.getenv("DIRECT_VLM_COMPILATION_CONFIG") or None)
    parser.add_argument("--max-image-width", type=int, default=_optional_int_env("DIRECT_VLM_MAX_IMAGE_WIDTH") or 512)
    parser.add_argument(
        "--max-table-image-width",
        type=int,
        default=_optional_int_env("DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH") or 1024,
    )
    parser.add_argument(
        "--max-page-image-width",
        type=int,
        default=_optional_int_env("DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH") or 1024,
    )
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("DIRECT_VLM_INTERVAL_SECONDS", "0.1")))
    parser.add_argument("--no-image-analysis", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asyncio.run(
        run_direct_worker_loop(
            orchestrator_base_url=args.orchestrator_base_url,
            worker_id=args.worker_id,
            limit=args.limit,
            interval_seconds=args.interval_seconds,
            once=args.once,
            kind=args.kind,
            lease_timeout_seconds=args.lease_timeout_seconds,
            job_id=args.job_id,
            model_path=args.model_path,
            backend=args.backend,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            kv_cache_dtype=args.kv_cache_dtype,
            compilation_config=args.compilation_config,
            max_image_width=args.max_image_width,
            max_table_image_width=args.max_table_image_width,
            max_page_image_width=args.max_page_image_width,
            image_analysis=not args.no_image_analysis,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
