#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx


TERMINAL_TASK_STATUSES = {"completed", "failed"}
WAIT_POLICIES = {"none", "required", "all"}


def enhancement_priority(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    value = str(metadata.get("enhancement_priority") or "").strip().lower()
    if value in {"required", "optional", "asset_only"}:
        return value
    if metadata.get("image_asset_only"):
        return "asset_only"
    if metadata.get("scan_ocr") or metadata.get("page_vlm") or metadata.get("force_vlm"):
        return "required"
    return "optional"


def pending_wait_tasks(tasks: list[dict[str, Any]], wait_policy: str) -> list[dict[str, Any]]:
    if wait_policy not in WAIT_POLICIES:
        raise ValueError(f"unsupported wait policy: {wait_policy}")
    if wait_policy == "none":
        return []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("status") or "") in TERMINAL_TASK_STATUSES:
            continue
        priority = enhancement_priority(task)
        if wait_policy == "all" and priority != "asset_only":
            pending.append(task)
        elif wait_policy == "required" and priority == "required":
            pending.append(task)
    return pending


def build_result_summary(
    *,
    pdf_path: Path,
    wait_enhancements: str,
    job: dict[str, Any],
    tasks: list[dict[str, Any]],
    parse_wall_seconds: float,
    enhancement_wall_seconds: float,
    total_wall_seconds: float,
) -> dict[str, Any]:
    page_count = int(job.get("page_count") or 0)
    task_status_counts: dict[str, int] = {}
    priority_counts = {"required": 0, "optional": 0, "asset_only": 0}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        task_status_counts[status] = task_status_counts.get(status, 0) + 1
        priority = enhancement_priority(task)
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    return {
        "pdf_path": str(pdf_path),
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "wait_enhancements": wait_enhancements,
        "page_count": page_count,
        "fast_page_count": int(job.get("fast_page_count") or 0),
        "vlm_page_count": int(job.get("vlm_page_count") or 0),
        "block_count": int(job.get("block_count") or 0),
        "embedding_count": int(job.get("embedding_count") or 0),
        "enhancement_task_count": int(job.get("enhancement_task_count") or len(tasks)),
        "required_enhancement_count": int(priority_counts.get("required", 0)),
        "optional_enhancement_count": int(priority_counts.get("optional", 0)),
        "asset_only_enhancement_count": int(priority_counts.get("asset_only", 0)),
        "pending_wait_enhancement_count": len(pending_wait_tasks(tasks, wait_enhancements)),
        "service_elapsed_seconds": job.get("elapsed_seconds"),
        "timings": job.get("timings") or {},
        "parse_wall_seconds": round(parse_wall_seconds, 3),
        "parse_pages_per_second": round(page_count / parse_wall_seconds, 3) if parse_wall_seconds > 0 else 0.0,
        "enhancement_wall_seconds": round(enhancement_wall_seconds, 3),
        "total_wall_seconds": round(total_wall_seconds, 3),
        "total_pages_per_second": round(page_count / total_wall_seconds, 3) if total_wall_seconds > 0 else 0.0,
        "task_status_counts": task_status_counts,
    }


async def submit_and_measure(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    pdf_path = Path(args.pdf)
    timeout = httpx.Timeout(connect=30.0, read=args.timeout, write=args.timeout, pool=30.0)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        with pdf_path.open("rb") as pdf_file:
            response = await client.post(
                f"{base_url}/api/v1/jobs",
                files={"file": (pdf_path.name, pdf_file, "application/pdf")},
                data={"run_embedding": str(args.run_embedding).lower()},
            )
        response.raise_for_status()
        job_id = response.json()["job_id"]

        while True:
            job_response = await client.get(f"{base_url}/api/v1/jobs/{job_id}")
            job_response.raise_for_status()
            job = job_response.json()
            if job["status"] in {"completed", "failed"}:
                parse_wall = time.perf_counter() - started
                break
            await asyncio.sleep(args.poll_interval)

        enhancement_started = time.perf_counter()
        tasks: list[dict[str, Any]] = []
        while True:
            tasks_response = await client.get(f"{base_url}/api/v1/jobs/{job_id}/enhancements")
            tasks_response.raise_for_status()
            tasks = tasks_response.json().get("tasks", [])
            if not pending_wait_tasks(tasks, args.wait_enhancements):
                break
            if time.perf_counter() - enhancement_started > args.timeout:
                break
            await asyncio.sleep(args.poll_interval)
        enhancement_wall = 0.0 if args.wait_enhancements == "none" else time.perf_counter() - enhancement_started
        total_wall = time.perf_counter() - started

    return build_result_summary(
        pdf_path=pdf_path,
        wait_enhancements=args.wait_enhancements,
        job=job,
        tasks=tasks,
        parse_wall_seconds=parse_wall,
        enhancement_wall_seconds=enhancement_wall,
        total_wall_seconds=total_wall,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parse latency and enhancement wait policies.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--wait-enhancements", choices=sorted(WAIT_POLICIES), default="required")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--run-embedding", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = asyncio.run(submit_and_measure(parse_args()))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if result.get("status") != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
