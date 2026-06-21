#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


async def submit_and_wait(client: httpx.AsyncClient, base_url: str, pdf_path: Path, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    with pdf_path.open("rb") as pdf_file:
        response = await client.post(
            f"{base_url}/api/v1/jobs",
            files={"file": (pdf_path.name, pdf_file, "application/pdf")},
        )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    while True:
        status_response = await client.get(f"{base_url}/api/v1/jobs/{job_id}")
        status_response.raise_for_status()
        job = status_response.json()
        if job["status"] in {"completed", "failed"}:
            job["client_wall_seconds"] = round(time.perf_counter() - started, 3)
            job["client_index"] = index
            return job
        await asyncio.sleep(0.05)


async def run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    pdf_path = Path(args.pdf)
    timeout = httpx.Timeout(connect=30.0, read=args.timeout, write=args.timeout, pool=30.0)
    limits = httpx.Limits(max_connections=max(args.concurrency * 2, 10), max_keepalive_connections=max(args.concurrency, 10))
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        async def guarded(index: int) -> dict[str, Any]:
            async with semaphore:
                return await submit_and_wait(client, base_url, pdf_path, index)

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(index) for index in range(args.jobs)))
        total = round(time.perf_counter() - started, 3)

    completed = [result for result in results if result["status"] == "completed"]
    failed = [result for result in results if result["status"] != "completed"]
    service_times = [result.get("elapsed_seconds") or 0.0 for result in completed]
    wall_times = [result.get("client_wall_seconds") or 0.0 for result in completed]
    pages = sum(result.get("page_count") or 0 for result in completed)
    blocks = sum(result.get("block_count") or 0 for result in completed)

    print(f"jobs={args.jobs} concurrency={args.concurrency} completed={len(completed)} failed={len(failed)} total_wall_seconds={total}")
    if completed:
        print(f"pages={pages} blocks={blocks} pages_per_second={round(pages / total, 3)}")
        print(f"service_elapsed_avg={round(mean(service_times), 3)} client_wall_avg={round(mean(wall_times), 3)}")
        print(f"service_elapsed_max={round(max(service_times), 3)} client_wall_max={round(max(wall_times), 3)}")
        timing_keys = ["routing_seconds", "vlm_seconds", "embedding_seconds", "storage_seconds", "total_seconds"]
        for key in timing_keys:
            values = [float(result.get("timings", {}).get(key) or 0.0) for result in completed]
            if values:
                print(f"{key}_avg={round(mean(values), 3)} {key}_max={round(max(values), 3)}")
    for result in results:
        print(
            "job",
            result["client_index"],
            result["job_id"],
            result["status"],
            f"pages={result.get('page_count')}",
            f"blocks={result.get('block_count')}",
            f"embeddings={result.get('embedding_count')}",
            f"enhancements={result.get('enhancement_task_count')}",
            f"elapsed={result.get('elapsed_seconds')}",
            f"timings={result.get('timings')}",
            f"wall={result.get('client_wall_seconds')}",
            f"error={result.get('error')}",
        )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MinerU VLM Lab async jobs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
