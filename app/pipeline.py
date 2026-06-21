from __future__ import annotations

import asyncio
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import io
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import fitz
import httpx

from app.embedding import build_embedding_provider
from app.models import BlockRecord, EnhancementTask, PageRoute, ParseSummary, StorageStats
from app.routing import (
    TABLE_CAPTION_PATTERN,
    classify_page,
    collect_page_signal,
    extract_fast_page,
    looks_like_table_text,
    should_extract_native_tables,
)
from app.settings import Settings
from app.storage import SQLiteStore


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _flatten_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        flattened: list[dict[str, Any]] = []
        for item in value:
            flattened.extend(_flatten_items(item))
        return flattened
    return []


def _item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "html", "latex", "caption", "markdown"):
        text = _clean_text(item.get(key))
        if text:
            return text
    return ""


def _item_bbox(item: dict[str, Any]) -> list[float] | None:
    raw = item.get("bbox") or item.get("poly") or item.get("position")
    if not isinstance(raw, list):
        return None
    values: list[float] = []
    for value in raw[:4]:
        try:
            values.append(round(float(value), 2))
        except (TypeError, ValueError):
            return None
    return values if len(values) == 4 else None


def normalize_vlm_archive(response_bytes: bytes, job_id: str, page_index: int) -> tuple[list[BlockRecord], dict[str, Any]]:
    metadata: dict[str, Any] = {"zip_entries": []}
    blocks: list[BlockRecord] = []
    with zipfile.ZipFile(io.BytesIO(response_bytes)) as archive:
        names = archive.namelist()
        metadata["zip_entries"] = names[:40]
        content_names = [name for name in names if name.endswith("content_list.json") or name.endswith("content_list_v2.json")]
        for name in content_names:
            try:
                payload = json.loads(archive.read(name).decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            for item in _flatten_items(payload):
                text = _item_text(item)
                if not text:
                    continue
                block_index = len(blocks)
                blocks.append(
                    BlockRecord(
                        job_id=job_id,
                        page_index=page_index,
                        block_index=block_index,
                        source="mineru_vlm",
                        kind=str(item.get("type") or item.get("kind") or "text"),
                        text=text,
                        bbox=_item_bbox(item),
                        metadata={"archive": name, "raw": item},
                    )
                )
        if not blocks:
            md_names = [name for name in names if name.endswith(".md")]
            for name in md_names:
                markdown = archive.read(name).decode("utf-8", errors="replace")
                text = _clean_text(markdown)
                if text:
                    blocks.append(
                        BlockRecord(
                            job_id=job_id,
                            page_index=page_index,
                            block_index=len(blocks),
                            source="mineru_vlm",
                            kind="markdown",
                            text=text,
                            bbox=None,
                            metadata={"archive": name},
                        )
                    )
                    metadata["markdown_preview"] = markdown[:2000]
                    break
    return blocks, metadata


async def call_vlm_page(
    client: httpx.AsyncClient,
    settings: Settings,
    pdf_path: Path,
    job_id: str,
    page_index: int,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], list[BlockRecord]]:
    data = {
        "output_dir": str(out_dir),
        "backend": "vlm-auto-engine",
        "parse_method": "auto",
        "return_md": "true",
        "return_middle_json": "true",
        "return_content_list": "true",
        "return_images": "false",
        "response_format_zip": "true",
        "return_original_file": "false",
        "start_page_id": str(page_index),
        "end_page_id": str(page_index),
    }
    started = time.monotonic()
    async with semaphore:
        with pdf_path.open("rb") as pdf_file:
            response = await client.post(
                f"{settings.mineru_vlm_base_url}/file_parse",
                data=data,
                files={"files": (pdf_path.name, pdf_file, "application/pdf")},
            )
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "page_index": page_index,
        "status_code": response.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "content_type": response.headers.get("content-type"),
        "bytes": len(response.content),
    }
    page_dir = out_dir / f"page-{page_index:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    if response.status_code >= 400:
        result["error"] = response.text[:2000]
        return result, []
    zip_path = page_dir / "vlm_result.zip"
    zip_path.write_bytes(response.content)
    result["zip_path"] = str(zip_path)
    try:
        blocks, metadata = normalize_vlm_archive(response.content, job_id, page_index)
        result.update(metadata)
        result["block_count"] = len(blocks)
        return result, blocks
    except zipfile.BadZipFile:
        result["error"] = "VLM response was not a zip archive"
        return result, []


async def embed_blocks(settings: Settings, blocks: list[BlockRecord]) -> dict[str, dict[str, Any]]:
    provider = build_embedding_provider(settings.embedding_provider, settings.embedding_dim)
    if not blocks:
        return {}

    batch_size = max(1, settings.embedding_batch_size)
    max_embedding_workers = 1 if provider.provider == "transformers" else max(1, settings.max_concurrent_embedding_batches)
    semaphore = asyncio.Semaphore(max_embedding_workers)
    batches = [blocks[index : index + batch_size] for index in range(0, len(blocks), batch_size)]

    async def embed_batch(batch: list[BlockRecord]) -> dict[str, dict[str, Any]]:
        async with semaphore:
            vectors = await asyncio.to_thread(provider.embed, [block.embedding_text for block in batch])
        return {
            block.block_id: {
                "embedding_id": f"emb-{block.block_id}",
                "provider": provider.provider,
                "model": provider.model,
                "dim": provider.dim,
                "vector": vector,
            }
            for block, vector in zip(batch, vectors)
        }

    merged: dict[str, dict[str, Any]] = {}
    for result in await asyncio.gather(*(embed_batch(batch) for batch in batches)):
        merged.update(result)
    return merged


def extract_fast_page_job(
    pdf_path: Path,
    page_index: int,
    request_id: str,
    route_reason: str,
    enhancement_root: Path,
) -> tuple[dict[str, Any], list[BlockRecord], list[EnhancementTask]]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        fast_page, page_blocks = extract_fast_page(page, page_index, request_id, route_reason)
        enhancement_tasks = create_enhancement_tasks(page, page_blocks, enhancement_root)
        return fast_page, page_blocks, enhancement_tasks
    finally:
        doc.close()


def extract_async_vlm_page_job(
    pdf_path: Path,
    page_index: int,
    request_id: str,
    route_reason: str,
    enhancement_root: Path,
) -> tuple[dict[str, Any], list[BlockRecord], list[EnhancementTask]]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        page_rect = page.rect
        block = BlockRecord(
            job_id=request_id,
            page_index=page_index,
            block_index=0,
            source="async_page_vlm",
            kind="image",
            text=f"[page vlm page={page_index}]",
            bbox=[
                round(float(page_rect.x0), 2),
                round(float(page_rect.y0), 2),
                round(float(page_rect.x1), 2),
                round(float(page_rect.y1), 2),
            ],
            metadata={
                "route_reason": route_reason,
                "image_candidate": True,
                "force_vlm": True,
                "page_vlm": True,
                "image_asset_only": False,
            },
        )
        tasks = create_enhancement_tasks(page, [block], enhancement_root)
        page_result = {
            "page_index": page_index,
            "status": "queued",
            "mode": "async_direct_vlm",
            "route_reason": route_reason,
            "block_count": 1,
            "enhancement_task_count": len(tasks),
        }
        return page_result, [block], tasks
    finally:
        doc.close()


def extract_hybrid_ocr_page_job(
    pdf_path: Path,
    page_index: int,
    request_id: str,
    route_reason: str,
    enhancement_root: Path,
) -> tuple[dict[str, Any], list[BlockRecord], list[EnhancementTask]]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        ocr_result = _run_tesseract_page_ocr(page, enhancement_root, page_index)
        min_text_chars = _int_env("SCAN_OCR_MIN_TEXT_CHARS", 80)
        if not ocr_result.get("available") or len(str(ocr_result.get("text") or "").strip()) < min_text_chars:
            return extract_async_vlm_page_job(pdf_path, page_index, request_id, route_reason, enhancement_root)

        page_rect = page.rect
        blocks = _build_scan_ocr_blocks(
            request_id=request_id,
            page_index=page_index,
            route_reason=route_reason,
            page_rect=page_rect,
            ocr_lines=ocr_result["lines"],
        )
        tasks = create_enhancement_tasks(page, blocks, enhancement_root)
        page_result = {
            "page_index": page_index,
            "status": "queued" if tasks else "completed",
            "mode": "hybrid_ocr",
            "ocr_engine": "tesseract",
            "route_reason": route_reason,
            "block_count": len(blocks),
            "enhancement_task_count": len(tasks),
            "ocr_text_chars": len(str(ocr_result.get("text") or "")),
        }
        return page_result, blocks, tasks
    finally:
        doc.close()


async def parse_pdf_file(
    *,
    pdf_path: Path,
    file_name: str,
    request_id: str,
    settings: Settings,
    max_concurrent_vlm_pages: int,
    persist: bool,
    store: SQLiteStore | None = None,
    run_embedding: bool = True,
) -> ParseSummary:
    started = time.monotonic()
    timings: dict[str, float] = {
        "routing_seconds": 0.0,
        "vlm_seconds": 0.0,
        "embedding_seconds": 0.0,
        "storage_seconds": 0.0,
    }
    out_dir = settings.work_root / request_id / "vlm"
    fast_pages: list[dict[str, Any]] = []
    deferred_fast_jobs: list[tuple[int, str]] = []
    vlm_page_indexes: list[int] = []
    routes: list[PageRoute] = []
    blocks: list[BlockRecord] = []
    enhancement_tasks: list[EnhancementTask] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    try:
        routing_started = time.monotonic()
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            signal = collect_page_signal(page, page_index)
            route, reason = classify_page(signal)
            routes.append(
                PageRoute(
                    page_index=page_index,
                    route=route,
                    reason=reason,
                    text_chars=signal.text_chars,
                    block_count=signal.block_count,
                    image_count=signal.image_count,
                    image_area_ratio=round(signal.image_area_ratio, 4),
                )
            )
            if route == "fast_pymupdf":
                page_text = page.get_text("text") or ""
                if should_extract_native_tables(page_text, reason):
                    deferred_fast_jobs.append((page_index, reason))
                else:
                    fast_page, page_blocks = extract_fast_page(page, page_index, request_id, reason)
                    fast_pages.append(fast_page)
                    blocks.extend(page_blocks)
                    enhancement_tasks.extend(
                        create_enhancement_tasks(page, page_blocks, settings.work_root / request_id / "enhancements")
                    )
            else:
                vlm_page_indexes.append(page_index)
    finally:
        doc.close()

    if deferred_fast_jobs:
        max_fast_workers = max(1, int(os.getenv("MAX_CONCURRENT_FAST_PAGES", "8")))
        loop = asyncio.get_running_loop()

        with ProcessPoolExecutor(max_workers=max_fast_workers) as pool:

            async def run_fast_job(
                page_index: int, reason: str
            ) -> tuple[dict[str, Any], list[BlockRecord], list[EnhancementTask]]:
                return await loop.run_in_executor(
                    pool,
                    partial(
                        extract_fast_page_job,
                        pdf_path,
                        page_index,
                        request_id,
                        reason,
                        settings.work_root / request_id / "enhancements",
                    ),
                )

            for fast_page, page_blocks, page_enhancement_tasks in await asyncio.gather(
                *(run_fast_job(page_index, reason) for page_index, reason in deferred_fast_jobs)
            ):
                fast_pages.append(fast_page)
                blocks.extend(page_blocks)
                enhancement_tasks.extend(page_enhancement_tasks)
    fast_pages.sort(key=lambda item: int(item["page_index"]))
    blocks.sort(key=lambda block: (block.page_index, block.block_index))
    enhancement_tasks.sort(key=lambda task: (task.page_index, task.task_id))
    timings["routing_seconds"] = round(time.monotonic() - routing_started, 3)

    vlm_pages: list[dict[str, Any]] = []
    if vlm_page_indexes:
        vlm_started = time.monotonic()
        max_page_workers = max(1, int(os.getenv("MAX_CONCURRENT_ASYNC_VLM_PAGE_RENDERS", str(max(1, max_concurrent_vlm_pages)))))
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=max_page_workers) as pool:

            async def run_vlm_page_job(
                page_index: int,
            ) -> tuple[dict[str, Any], list[BlockRecord], list[EnhancementTask]]:
                reason = routes[page_index].reason
                extractor = extract_hybrid_ocr_page_job if _scan_ocr_enabled() else extract_async_vlm_page_job
                return await loop.run_in_executor(
                    pool,
                    partial(
                        extractor,
                        pdf_path,
                        page_index,
                        request_id,
                        reason,
                        settings.work_root / request_id / "enhancements",
                    ),
                )

            for result, page_blocks, page_enhancement_tasks in await asyncio.gather(
                *(run_vlm_page_job(page_index) for page_index in vlm_page_indexes)
            ):
                vlm_pages.append(result)
                offset = len([block for block in blocks if block.page_index == result["page_index"]])
                for index, block in enumerate(page_blocks):
                    blocks.append(
                        BlockRecord(
                            job_id=block.job_id,
                            page_index=block.page_index,
                            block_index=offset + index,
                            source=block.source,
                            kind=block.kind,
                            text=block.text,
                            bbox=block.bbox,
                            metadata=block.metadata,
                    )
                )
                enhancement_tasks.extend(page_enhancement_tasks)
        timings["vlm_seconds"] = round(time.monotonic() - vlm_started, 3)

    embeddings: dict[str, dict[str, Any]] = {}
    if run_embedding:
        embedding_started = time.monotonic()
        embeddings = await embed_blocks(settings, blocks)
        timings["embedding_seconds"] = round(time.monotonic() - embedding_started, 3)
    storage = StorageStats(
        stored=False,
        block_count=len(blocks),
        embedding_count=len(embeddings),
        enhancement_task_count=len(enhancement_tasks),
    )
    if persist:
        if store is None:
            store = SQLiteStore(settings.db_path)
            store.init_schema()
        storage_started = time.monotonic()
        preliminary_elapsed = round(storage_started - started, 3)
        storage = store.store_outputs(
            job_id=request_id,
            file_name=file_name,
            source_path=str(pdf_path),
            page_count=len(routes),
            fast_page_count=len(fast_pages),
            vlm_page_count=len(vlm_pages),
            elapsed_seconds=preliminary_elapsed,
            timings={**timings, "total_seconds": preliminary_elapsed},
            routes=routes,
            blocks=blocks,
            embeddings=embeddings,
            enhancement_tasks=enhancement_tasks,
        )
        timings["storage_seconds"] = round(time.monotonic() - storage_started, 3)

    elapsed = round(time.monotonic() - started, 3)
    timings["total_seconds"] = elapsed
    if persist and store is not None:
        store.update_job_elapsed(request_id, elapsed, timings)

    block_views: list[dict[str, Any]] = []
    for block in blocks:
        view = block.to_dict()
        embedding = embeddings.get(block.block_id)
        if embedding:
            view.update(
                {
                    "embedding_id": embedding["embedding_id"],
                    "embedding_provider": embedding["provider"],
                    "embedding_model": embedding["model"],
                    "embedding_dim": embedding["dim"],
                }
            )
        block_views.append(view)

    return ParseSummary(
        request_id=request_id,
        file_name=file_name,
        page_count=len(routes),
        elapsed_seconds=elapsed,
        timings=timings,
        fast_page_count=len(fast_pages),
        vlm_page_count=len(vlm_pages),
        block_count=len(blocks),
        embedding_count=len(embeddings),
        enhancement_task_count=len(enhancement_tasks),
        routes=routes,
        fast_pages=fast_pages,
        vlm_pages=vlm_pages,
        blocks=block_views,
        enhancement_tasks=[task.to_dict() for task in enhancement_tasks],
        storage=storage,
    )


def create_enhancement_tasks(page: fitz.Page, blocks: list[BlockRecord], output_dir: Path) -> list[EnhancementTask]:
    tasks: list[EnhancementTask] = []
    candidate_blocks = [
        block
        for block in blocks
        if block.bbox
        and (
            block.metadata.get("table_candidate")
            or block.metadata.get("image_candidate")
            or block.metadata.get("native_table_needs_vlm")
        )
    ]
    if not candidate_blocks:
        return tasks
    output_dir.mkdir(parents=True, exist_ok=True)
    page_rect = page.rect
    for block in candidate_blocks:
        assert block.bbox is not None
        kind = "image_candidate" if block.metadata.get("image_candidate") else "table_candidate"
        crop_padding = _enhancement_crop_padding(kind)
        rect = fitz.Rect(block.bbox)
        rect.x0 = max(page_rect.x0, rect.x0 - crop_padding)
        rect.y0 = max(page_rect.y0, rect.y0 - crop_padding)
        rect.x1 = min(page_rect.x1, rect.x1 + crop_padding)
        rect.y1 = min(page_rect.y1, rect.y1 + crop_padding)
        crop_path = output_dir / f"page-{block.page_index:04d}-block-{block.block_index:04d}.png"
        crop_scale = _enhancement_crop_scale(kind)
        pix = page.get_pixmap(matrix=fitz.Matrix(crop_scale, crop_scale), clip=rect, alpha=False)
        pix.save(str(crop_path))
        crop_bbox = [round(float(rect.x0), 2), round(float(rect.y0), 2), round(float(rect.x1), 2), round(float(rect.y1), 2)]
        block.metadata["crop_path"] = str(crop_path)
        block.metadata["crop_bbox"] = crop_bbox
        block.metadata["crop_padding"] = crop_padding
        block.metadata["crop_scale"] = crop_scale
        if kind == "image_candidate" and not _should_queue_image_vlm(block):
            block.metadata["image_asset_only"] = True
            block.metadata["enhancement_priority"] = "asset_only"
            continue
        if kind == "image_candidate":
            block.metadata["image_asset_only"] = False
        priority = _enhancement_priority(block)
        max_image_width = _enhancement_max_image_width(block, kind, crop_bbox, priority)
        block.metadata["enhancement_priority"] = priority
        if max_image_width is not None:
            block.metadata["max_image_width"] = max_image_width
        tasks.append(
            EnhancementTask(
                job_id=block.job_id,
                block_id=block.block_id,
                page_index=block.page_index,
                kind=kind,
                status="queued",
                crop_path=str(crop_path),
                bbox=crop_bbox,
                metadata={
                    "source": block.source,
                    "route_reason": block.metadata.get("route_reason"),
                    "image_vlm_mode": block.metadata.get("image_vlm_mode"),
                    "image_asset_only": bool(block.metadata.get("image_asset_only")),
                    "vector_graphics_candidate": bool(block.metadata.get("vector_graphics_candidate")),
                    "native_table": bool(block.metadata.get("native_table")),
                    "native_table_needs_vlm": bool(block.metadata.get("native_table_needs_vlm")),
                    "scan_ocr": bool(block.metadata.get("scan_ocr")),
                    "force_vlm": bool(block.metadata.get("force_vlm")),
                    "page_vlm": bool(block.metadata.get("page_vlm")),
                    "enhancement_priority": priority,
                    "crop_padding": crop_padding,
                    "crop_scale": crop_scale,
                    "max_image_width": max_image_width,
                },
            )
        )
    return tasks


def _enhancement_priority(block: BlockRecord) -> str:
    if block.metadata.get("image_asset_only"):
        return "asset_only"
    if block.metadata.get("force_vlm") or block.metadata.get("page_vlm"):
        return "required"
    if block.metadata.get("scan_ocr"):
        return "required"
    if block.metadata.get("native_table_needs_vlm"):
        if block.metadata.get("native_table") or not block.metadata.get("native_table_deferred"):
            return "required"
    if block.metadata.get("table_candidate") and not _clean_text(block.text):
        return "required"
    return "optional"


def _enhancement_max_image_width(block: BlockRecord, kind: str, crop_bbox: list[float], priority: str) -> int | None:
    if kind != "table_candidate":
        return None
    default_width = max(1, _int_env("ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH", 640))
    if priority != "required":
        return default_width
    if _uses_complex_table_image_width(block, crop_bbox):
        return max(1, _int_env("ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH", 768))
    return default_width


def _uses_complex_table_image_width(block: BlockRecord, crop_bbox: list[float]) -> bool:
    if block.metadata.get("force_vlm") or block.metadata.get("page_vlm"):
        return True
    if block.metadata.get("scan_ocr") or block.metadata.get("native_table_needs_vlm"):
        return True
    rows = block.metadata.get("table_rows")
    if isinstance(rows, list):
        min_rows = _int_env("ENHANCEMENT_COMPLEX_TABLE_MIN_ROWS", 8)
        if len(rows) >= min_rows:
            return True
        if any(isinstance(row, list) and len(row) >= _int_env("ENHANCEMENT_COMPLEX_TABLE_MIN_COLUMNS", 6) for row in rows):
            return True
    text = _clean_text(block.text)
    if len(text) >= _int_env("ENHANCEMENT_COMPLEX_TABLE_MIN_CHARS", 900):
        return True
    line_count = len([line for line in str(block.text).splitlines() if line.strip()])
    if line_count >= _int_env("ENHANCEMENT_COMPLEX_TABLE_MIN_LINES", 10):
        return True
    width = max(0.0, float(crop_bbox[2]) - float(crop_bbox[0]))
    height = max(0.0, float(crop_bbox[3]) - float(crop_bbox[1]))
    min_area = _float_env("ENHANCEMENT_COMPLEX_TABLE_MIN_AREA", 0.0)
    return min_area > 0 and width * height >= min_area


def _scan_ocr_enabled() -> bool:
    mode = os.getenv("SCAN_OCR_MODE", "auto").strip().lower()
    return mode not in {"0", "false", "no", "off", "none", "disabled", "page_vlm"}


def _run_tesseract_page_ocr(page: fitz.Page, output_dir: Path, page_index: int) -> dict[str, Any]:
    mode = os.getenv("SCAN_OCR_MODE", "auto").strip().lower()
    if mode not in {"auto", "tesseract", "ocr", "hybrid", "hybrid_ocr"}:
        return {"available": False, "reason": f"unsupported_scan_ocr_mode:{mode}"}
    binary = shutil.which(os.getenv("SCAN_OCR_TESSERACT_BIN", "tesseract"))
    if not binary:
        return {"available": False, "reason": "tesseract_not_found"}

    output_dir.mkdir(parents=True, exist_ok=True)
    scale = _float_env("SCAN_OCR_RENDER_SCALE", 2.0)
    image_path = output_dir / f"page-{page_index:04d}-ocr.png"
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pix.save(str(image_path))

    cmd = [binary, str(image_path), "stdout"]
    lang = os.getenv("SCAN_OCR_LANG", "").strip()
    if lang:
        cmd.extend(["-l", lang])
    cmd.extend(["--psm", os.getenv("SCAN_OCR_TESSERACT_PSM", "6"), "tsv"])
    timeout = _float_env("SCAN_OCR_TIMEOUT_SECONDS", 30.0)
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"tesseract_error:{exc}"}
    if result.returncode != 0:
        return {"available": False, "reason": f"tesseract_failed:{result.stderr[:400]}"}
    lines = _parse_tesseract_tsv(result.stdout, scale)
    return {
        "available": bool(lines),
        "reason": "ok" if lines else "no_ocr_lines",
        "lines": lines,
        "text": "\n".join(line["text"] for line in lines),
    }


def _parse_tesseract_tsv(value: str, scale: float) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(value), delimiter="\t")
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    min_conf = _float_env("SCAN_OCR_MIN_CONFIDENCE", 20.0)
    for row in reader:
        if row.get("level") != "5":
            continue
        text = " ".join(str(row.get("text") or "").split())
        if not text:
            continue
        try:
            conf = float(row.get("conf") or -1)
        except ValueError:
            conf = -1.0
        if conf >= 0 and conf < min_conf:
            continue
        try:
            left = float(row.get("left") or 0) / scale
            top = float(row.get("top") or 0) / scale
            width = float(row.get("width") or 0) / scale
            height = float(row.get("height") or 0) / scale
            block_num = int(row.get("block_num") or 0)
            par_num = int(row.get("par_num") or 0)
            line_num = int(row.get("line_num") or 0)
            word_num = int(row.get("word_num") or 0)
        except ValueError:
            continue
        grouped.setdefault((block_num, par_num, line_num), []).append(
            {
                "text": text,
                "word_num": word_num,
                "bbox": [left, top, left + width, top + height],
            }
        )

    lines: list[dict[str, Any]] = []
    for key, words in sorted(grouped.items()):
        words = sorted(words, key=lambda item: (item["bbox"][0], item["word_num"]))
        bbox = _union_bboxes([word["bbox"] for word in words])
        if bbox is None:
            continue
        lines.append(
            {
                "key": key,
                "text": _ocr_line_text(words),
                "bbox": [round(value, 2) for value in bbox],
            }
        )
    return lines


def _ocr_line_text(words: list[dict[str, Any]]) -> str:
    if not words:
        return ""
    pieces = [str(words[0]["text"])]
    for previous, current in zip(words, words[1:]):
        gap = float(current["bbox"][0]) - float(previous["bbox"][2])
        previous_width = max(1.0, float(previous["bbox"][2]) - float(previous["bbox"][0]))
        previous_chars = max(1, len(str(previous["text"])))
        char_width = max(3.5, previous_width / previous_chars)
        spaces = max(1, min(12, round(gap / char_width))) if gap > char_width * 1.5 else 1
        pieces.append(" " * spaces + str(current["text"]))
    return "".join(pieces)


def _build_scan_ocr_blocks(
    *,
    request_id: str,
    page_index: int,
    route_reason: str,
    page_rect: fitz.Rect,
    ocr_lines: list[dict[str, Any]],
) -> list[BlockRecord]:
    table_line_indexes = _scan_ocr_table_line_indexes(ocr_lines)
    blocks: list[BlockRecord] = []
    page_bbox = [
        round(float(page_rect.x0), 2),
        round(float(page_rect.y0), 2),
        round(float(page_rect.x1), 2),
        round(float(page_rect.y1), 2),
    ]
    blocks.append(
        BlockRecord(
            job_id=request_id,
            page_index=page_index,
            block_index=len(blocks),
            source="scan_ocr_tesseract",
            kind="image",
            text=f"[scan page asset page={page_index}]",
            bbox=page_bbox,
            metadata={
                "route_reason": route_reason,
                "scan_ocr": True,
                "image_candidate": True,
                "image_asset_only": True,
            },
        )
    )

    if table_line_indexes:
        table_lines = [ocr_lines[index] for index in sorted(table_line_indexes)]
        bbox = _union_bboxes([line["bbox"] for line in table_lines])
        if bbox is not None:
            blocks.append(
                BlockRecord(
                    job_id=request_id,
                    page_index=page_index,
                    block_index=len(blocks),
                    source="scan_ocr_tesseract",
                    kind="native_table",
                    text="\n".join(line["text"] for line in table_lines),
                    bbox=[round(value, 2) for value in bbox],
                    metadata={
                        "route_reason": route_reason,
                        "scan_ocr": True,
                        "table_candidate": True,
                        "native_table_needs_vlm": True,
                    },
                )
            )

    text_lines = [line for index, line in enumerate(ocr_lines) if index not in table_line_indexes]
    if text_lines:
        bbox = _union_bboxes([line["bbox"] for line in text_lines])
        if bbox is not None:
            blocks.append(
                BlockRecord(
                    job_id=request_id,
                    page_index=page_index,
                    block_index=len(blocks),
                    source="scan_ocr_tesseract",
                    kind="text",
                    text="\n".join(line["text"] for line in text_lines),
                    bbox=[round(value, 2) for value in bbox],
                    metadata={
                        "route_reason": route_reason,
                        "scan_ocr": True,
                    },
                )
            )
    return blocks


def _scan_ocr_table_line_indexes(lines: list[dict[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for index, line in enumerate(lines):
        text = str(line.get("text") or "")
        if TABLE_CAPTION_PATTERN.search(text):
            indexes.add(index)
            previous_y = float(line["bbox"][1])
            for next_index in range(index + 1, min(len(lines), index + 9)):
                next_y = float(lines[next_index]["bbox"][1])
                if next_y - previous_y > _float_env("SCAN_OCR_TABLE_MAX_LINE_GAP", 30.0):
                    break
                indexes.add(next_index)
                previous_y = next_y
            continue
        if looks_like_table_text(text):
            indexes.add(index)
    return indexes


def _union_bboxes(bboxes: list[list[float]]) -> list[float] | None:
    if not bboxes:
        return None
    x0 = min(float(bbox[0]) for bbox in bboxes)
    y0 = min(float(bbox[1]) for bbox in bboxes)
    x1 = max(float(bbox[2]) for bbox in bboxes)
    y1 = max(float(bbox[3]) for bbox in bboxes)
    return [x0, y0, x1, y1]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _enhancement_crop_padding(kind: str) -> float:
    if kind == "table_candidate":
        value = os.getenv("ENHANCEMENT_TABLE_CROP_PADDING", "40")
    else:
        value = os.getenv("ENHANCEMENT_CROP_PADDING", "8")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 40.0 if kind == "table_candidate" else 8.0


def _enhancement_crop_scale(kind: str) -> float:
    if kind == "table_candidate":
        value = os.getenv("ENHANCEMENT_TABLE_CROP_SCALE", "4")
    else:
        value = os.getenv("ENHANCEMENT_CROP_SCALE", "2")
    try:
        return max(1.0, float(value))
    except ValueError:
        return 4.0 if kind == "table_candidate" else 2.0


def _image_vlm_mode() -> str:
    value = os.getenv("ENHANCEMENT_IMAGE_VLM_MODE", "none").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "all"
    if value in {"0", "false", "no", "off", "asset", "asset-only"}:
        return "none"
    return value


def _should_queue_image_vlm(block: BlockRecord) -> bool:
    mode = _image_vlm_mode()
    block.metadata["image_vlm_mode"] = mode
    if block.metadata.get("force_vlm"):
        return True
    if mode == "all":
        return True
    if mode == "none":
        return False
    if mode in {"complex_uncaptioned", "complex-uncaptioned", "uncaptioned", "no-caption"}:
        return bool(
            block.metadata.get("force_vlm")
            or (
                block.metadata.get("vector_graphics_candidate")
                and not block.metadata.get("figure_caption_candidate")
            )
        )
    return bool(block.metadata.get("vector_graphics_candidate") or block.metadata.get("force_vlm"))
