from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.models import (
    BlocksResponse,
    EnhancementClaimRequest,
    EnhancementClaimResponse,
    EnhancementCompleteRequest,
    EnhancementFailRequest,
    EnhancementsResponse,
    JobCreated,
    ParseSummary,
    RuntimeStatsResponse,
)
from app.embedding import build_embedding_provider
from app.pipeline import parse_pdf_file
from app.settings import Settings, load_settings
from app.storage import SQLiteStore


settings: Settings = load_settings()
store = SQLiteStore(settings.db_path)
running_tasks: set[asyncio.Task[Any]] = set()


def ensure_dirs() -> None:
    settings.work_root.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    store.init_schema()
    store.mark_interrupted_jobs()


def save_upload(upload: UploadFile, request_id: str) -> Path:
    suffix = Path(upload.filename or "input.pdf").suffix or ".pdf"
    target = settings.work_root / request_id / f"input{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return target


async def run_job(
    *,
    job_id: str,
    file_name: str,
    pdf_path: Path,
    max_concurrent_vlm_pages: int,
    run_embedding: bool,
) -> None:
    store.mark_processing(job_id)
    try:
        await parse_pdf_file(
            pdf_path=pdf_path,
            file_name=file_name,
            request_id=job_id,
            settings=settings,
            max_concurrent_vlm_pages=max_concurrent_vlm_pages,
            persist=True,
            store=store,
            run_embedding=run_embedding,
        )
    except Exception as exc:
        store.mark_failed(job_id, str(exc))


app = FastAPI(title="MinerU VLM Lab", version="0.2.0")


@app.on_event("startup")
async def startup() -> None:
    ensure_dirs()
    if settings.preload_embedding:
        provider = build_embedding_provider(settings.embedding_provider, settings.embedding_dim)
        await asyncio.to_thread(provider.embed, ["embedding warmup"])


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in list(running_tasks):
        if not task.done():
            task.cancel()


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        try:
            mineru_health = (await client.get(f"{settings.mineru_vlm_base_url}/health")).json()
        except Exception as exc:
            mineru_health = {"status": "unavailable", "error": str(exc)}
    return {
        "status": "ok",
        "mode": "vlm-only",
        "mineru_vlm_base_url": settings.mineru_vlm_base_url,
        "db_path": str(settings.db_path),
        "embedding_provider": settings.embedding_provider,
        "embedding_dim": settings.embedding_dim,
        "preload_embedding": settings.preload_embedding,
        "mineru_vlm": mineru_health,
    }


@app.get("/api/v1/stats", response_model=RuntimeStatsResponse)
async def runtime_stats() -> RuntimeStatsResponse:
    return RuntimeStatsResponse(**store.runtime_stats())


@app.post("/api/v1/jobs", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    file: UploadFile = File(...),
    max_concurrent_vlm_pages: int = Form(settings.max_concurrent_vlm_pages),
    run_embedding: bool = Form(True),
) -> JobCreated:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    job_id = uuid.uuid4().hex
    pdf_path = save_upload(file, job_id)
    store.start_job(job_id, file.filename, str(pdf_path), status="queued")
    task = asyncio.create_task(
        run_job(
            job_id=job_id,
            file_name=file.filename,
            pdf_path=pdf_path,
            max_concurrent_vlm_pages=max_concurrent_vlm_pages,
            run_embedding=run_embedding,
        )
    )
    running_tasks.add(task)
    task.add_done_callback(running_tasks.discard)
    return JobCreated(
        job_id=job_id,
        status="queued",
        status_url=f"/api/v1/jobs/{job_id}",
        blocks_url=f"/api/v1/jobs/{job_id}/blocks",
    )


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=json.loads(job.model_dump_json()))


@app.get("/api/v1/jobs/{job_id}/blocks", response_model=BlocksResponse)
async def get_job_blocks(job_id: str) -> BlocksResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    blocks = store.list_blocks(job_id)
    return BlocksResponse(job_id=job_id, total=len(blocks), blocks=blocks)


@app.get("/api/v1/jobs/{job_id}/enhancements", response_model=EnhancementsResponse)
async def get_job_enhancements(job_id: str) -> EnhancementsResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    tasks = store.list_enhancement_tasks(job_id)
    return EnhancementsResponse(job_id=job_id, total=len(tasks), tasks=tasks)


@app.post("/api/v1/enhancements/claim", response_model=EnhancementClaimResponse)
async def claim_enhancements(request: EnhancementClaimRequest) -> EnhancementClaimResponse:
    tasks = store.claim_enhancement_tasks(
        request.limit,
        request.worker_id,
        request.job_id,
        request.kind,
        lease_timeout_seconds=request.lease_timeout_seconds,
    )
    return EnhancementClaimResponse(total=len(tasks), tasks=tasks)


@app.post("/api/v1/enhancements/{task_id}/complete")
async def complete_enhancement(task_id: str, request: EnhancementCompleteRequest) -> JSONResponse:
    task = store.complete_enhancement_task(task_id, request.worker_id, request.result)
    if task is None:
        raise HTTPException(status_code=404, detail="Enhancement task not found")
    return JSONResponse(content=task)


@app.post("/api/v1/enhancements/{task_id}/fail")
async def fail_enhancement(task_id: str, request: EnhancementFailRequest) -> JSONResponse:
    task = store.fail_enhancement_task(task_id, request.worker_id, request.error)
    if task is None:
        raise HTTPException(status_code=404, detail="Enhancement task not found")
    return JSONResponse(content=task)


@app.post("/parse", response_model=ParseSummary)
async def parse_pdf(
    file: UploadFile = File(...),
    max_concurrent_vlm_pages: int = Form(settings.max_concurrent_vlm_pages),
    persist: bool = Form(False),
    run_embedding: bool = Form(True),
) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    request_id = uuid.uuid4().hex
    pdf_path = save_upload(file, request_id)
    if persist:
        store.start_job(request_id, file.filename, str(pdf_path), status="processing")
    try:
        summary = await parse_pdf_file(
            pdf_path=pdf_path,
            file_name=file.filename,
            request_id=request_id,
            settings=settings,
            max_concurrent_vlm_pages=max_concurrent_vlm_pages,
            persist=persist,
            store=store if persist else None,
            run_embedding=run_embedding,
        )
    except ValueError as exc:
        if persist:
            store.mark_failed(request_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if persist:
            store.mark_failed(request_id, str(exc))
        raise
    return JSONResponse(content=json.loads(summary.model_dump_json()))
