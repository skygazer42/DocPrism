from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import fitz
import httpx


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


def image_to_pdf(image_path: Path, pdf_path: Path) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    image_doc = fitz.open(str(image_path))
    try:
        pdf_bytes = image_doc.convert_to_pdf()
    finally:
        image_doc.close()
    pdf_doc = fitz.open("pdf", pdf_bytes)
    try:
        pdf_doc.save(pdf_path)
    finally:
        pdf_doc.close()
    return pdf_path


def parse_vlm_zip(payload: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"zip_entries": [], "markdown": "", "content_items": []}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        result["zip_entries"] = names[:80]
        md_names = [name for name in names if name.endswith(".md")]
        if md_names:
            result["markdown"] = archive.read(md_names[0]).decode("utf-8", errors="replace")
        content_names = [
            name
            for name in names
            if name.endswith("content_list.json") or name.endswith("content_list_v2.json")
        ]
        for name in content_names:
            try:
                content = json.loads(archive.read(name).decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(content, list):
                result["content_items"].extend(_flatten_content_items(content))
    return result


def _flatten_content_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for item in value:
            items.extend(_flatten_content_items(item))
        return items
    return []


class TimeoutCleanupRunner:
    def __init__(
        self,
        command: str | None,
        *,
        cooldown_seconds: float = 60.0,
        command_timeout_seconds: float = 180.0,
    ) -> None:
        self.command = command.strip() if command else ""
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.command_timeout_seconds = max(1.0, command_timeout_seconds)
        self._lock = asyncio.Lock()
        self._last_started_at = 0.0

    async def run(self) -> dict[str, Any] | None:
        if not self.command:
            return None
        async with self._lock:
            now = time.monotonic()
            if self._last_started_at and now - self._last_started_at < self.cooldown_seconds:
                return {"status": "skipped", "reason": "cooldown"}
            self._last_started_at = now
            started = time.monotonic()
            proc = await asyncio.create_subprocess_shell(
                self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.command_timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                return {
                    "status": "timeout",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "stdout": stdout.decode("utf-8", errors="replace")[-2000:],
                    "stderr": stderr.decode("utf-8", errors="replace")[-2000:],
                }
            return {
                "status": "completed" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stdout": stdout.decode("utf-8", errors="replace")[-2000:],
                "stderr": stderr.decode("utf-8", errors="replace")[-2000:],
            }


async def call_vlm_for_crop(
    *,
    client: httpx.AsyncClient,
    mineru_vlm_base_url: str,
    task: dict[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    crop_path = Path(task["crop_path"])
    task_scratch = scratch_root / task["task_id"]
    task_scratch.mkdir(parents=True, exist_ok=True)
    pdf_path = image_to_pdf(crop_path, task_scratch / "crop.pdf")
    data = {
        "output_dir": str(task_scratch / "vlm-output"),
        "backend": "vlm-auto-engine",
        "parse_method": "auto",
        "return_md": "true",
        "return_middle_json": "true",
        "return_content_list": "true",
        "return_images": "false",
        "response_format_zip": "true",
        "return_original_file": "false",
        "start_page_id": "0",
        "end_page_id": "0",
    }
    started = time.monotonic()
    with pdf_path.open("rb") as pdf_file:
        response = await client.post(
            f"{mineru_vlm_base_url.rstrip('/')}/file_parse",
            data=data,
            files={"files": (pdf_path.name, pdf_file, "application/pdf")},
        )
    elapsed = round(time.monotonic() - started, 3)
    base_result: dict[str, Any] = {
        "backend": "mineru_vlm",
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
        "content_type": response.headers.get("content-type"),
        "bytes": len(response.content),
        "crop_path": str(crop_path),
        "pdf_path": str(pdf_path),
    }
    if response.status_code >= 400:
        base_result["error"] = response.text[:4000]
        return base_result
    try:
        base_result.update(parse_vlm_zip(response.content))
    except zipfile.BadZipFile:
        base_result["error"] = "VLM response was not a zip archive"
    return base_result


async def run_worker_once(
    *,
    orchestrator_base_url: str,
    mineru_vlm_base_url: str,
    worker_id: str,
    limit: int,
    scratch_root: Path,
    concurrency: int = 1,
    lease_timeout_seconds: float | None = None,
    vlm_timeout_seconds: float | None = 120.0,
    timeout_cleanup_command: str | None = None,
    timeout_cleanup_cooldown_seconds: float = 60.0,
    timeout_cleanup_command_timeout_seconds: float = 180.0,
    job_id: str | None = None,
    kind: str | None = None,
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
        claim_response = await client.post(
            f"{orchestrator_base_url.rstrip('/')}/api/v1/enhancements/claim",
            json=claim_payload,
        )
        claim_response.raise_for_status()
        tasks = claim_response.json().get("tasks", [])
        semaphore = asyncio.Semaphore(max(1, concurrency))
        timeout_cleanup = TimeoutCleanupRunner(
            timeout_cleanup_command,
            cooldown_seconds=timeout_cleanup_cooldown_seconds,
            command_timeout_seconds=timeout_cleanup_command_timeout_seconds,
        )

        async def process_task(task: dict[str, Any]) -> str:
            async with semaphore:
                return await _process_task(
                    client=client,
                    orchestrator_base_url=orchestrator_base_url,
                    mineru_vlm_base_url=mineru_vlm_base_url,
                    worker_id=worker_id,
                    scratch_root=scratch_root,
                    vlm_timeout_seconds=vlm_timeout_seconds,
                    timeout_cleanup=timeout_cleanup,
                    task=task,
                )

        outcomes = await asyncio.gather(*(process_task(task) for task in tasks))
        return {"claimed": len(tasks), "completed": outcomes.count("completed"), "failed": outcomes.count("failed")}
    finally:
        if owns_client:
            await client.aclose()


async def _process_task(
    *,
    client: httpx.AsyncClient,
    orchestrator_base_url: str,
    mineru_vlm_base_url: str,
    worker_id: str,
    scratch_root: Path,
    vlm_timeout_seconds: float | None,
    timeout_cleanup: TimeoutCleanupRunner,
    task: dict[str, Any],
) -> str:
    try:
        vlm_call = call_vlm_for_crop(
            client=client,
            mineru_vlm_base_url=mineru_vlm_base_url,
            task=task,
            scratch_root=scratch_root,
        )
        if vlm_timeout_seconds is not None and vlm_timeout_seconds > 0:
            result = await asyncio.wait_for(vlm_call, timeout=vlm_timeout_seconds)
        else:
            result = await vlm_call
        if result.get("status_code", 500) >= 400 or result.get("error"):
            await _fail_task(
                client=client,
                orchestrator_base_url=orchestrator_base_url,
                worker_id=worker_id,
                task_id=task["task_id"],
                error=result.get("error") or str(result),
            )
            return "failed"
        complete_response = await client.post(
            f"{orchestrator_base_url.rstrip('/')}/api/v1/enhancements/{task['task_id']}/complete",
            json={"worker_id": worker_id, "result": result},
        )
        complete_response.raise_for_status()
        return "completed"
    except asyncio.TimeoutError:
        timeout_value = f"{vlm_timeout_seconds:g}" if vlm_timeout_seconds is not None else "unknown"
        await _fail_task(
            client=client,
            orchestrator_base_url=orchestrator_base_url,
            worker_id=worker_id,
            task_id=task["task_id"],
            error=f"VLM crop parse timed out after {timeout_value}s",
        )
        cleanup_result = await timeout_cleanup.run()
        if cleanup_result and cleanup_result.get("status") != "skipped":
            print(json.dumps({"timeout_cleanup": cleanup_result}, ensure_ascii=False), file=sys.stderr, flush=True)
        return "failed"
    except Exception as exc:
        try:
            await _fail_task(
                client=client,
                orchestrator_base_url=orchestrator_base_url,
                worker_id=worker_id,
                task_id=task["task_id"],
                error=str(exc),
            )
        finally:
            return "failed"


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
        json={"worker_id": worker_id, "error": error},
    )
    fail_response.raise_for_status()


async def run_worker_loop(
    *,
    orchestrator_base_url: str,
    mineru_vlm_base_url: str,
    worker_id: str,
    limit: int,
    scratch_root: Path,
    interval_seconds: float,
    once: bool,
    concurrency: int = 1,
    lease_timeout_seconds: float | None = None,
    vlm_timeout_seconds: float | None = 120.0,
    timeout_cleanup_command: str | None = None,
    timeout_cleanup_cooldown_seconds: float = 60.0,
    timeout_cleanup_command_timeout_seconds: float = 180.0,
    job_id: str | None = None,
    kind: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> None:
    while True:
        summary = await run_worker_once(
            orchestrator_base_url=orchestrator_base_url,
            mineru_vlm_base_url=mineru_vlm_base_url,
            worker_id=worker_id,
            limit=limit,
            concurrency=concurrency,
            lease_timeout_seconds=lease_timeout_seconds,
            vlm_timeout_seconds=vlm_timeout_seconds,
            timeout_cleanup_command=timeout_cleanup_command,
            timeout_cleanup_cooldown_seconds=timeout_cleanup_cooldown_seconds,
            timeout_cleanup_command_timeout_seconds=timeout_cleanup_command_timeout_seconds,
            scratch_root=scratch_root,
            job_id=job_id,
            kind=kind,
            client=client,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if once:
            return
        await asyncio.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume table enhancement crops with MinerU VLM.")
    parser.add_argument("--orchestrator-base-url", default="http://127.0.0.1:18180")
    parser.add_argument("--mineru-vlm-base-url", default="http://127.0.0.1:18100")
    parser.add_argument("--worker-id", default="table-vlm-worker-1")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--lease-timeout-seconds",
        type=float,
        default=_optional_float_env("ENHANCEMENT_LEASE_TIMEOUT_SECONDS"),
    )
    parser.add_argument(
        "--vlm-timeout-seconds",
        type=float,
        default=float(os.getenv("ENHANCEMENT_VLM_TIMEOUT_SECONDS", "120")),
    )
    parser.add_argument("--timeout-cleanup-command", default=os.getenv("ENHANCEMENT_TIMEOUT_CLEANUP_COMMAND", ""))
    parser.add_argument(
        "--timeout-cleanup-cooldown-seconds",
        type=float,
        default=float(os.getenv("ENHANCEMENT_TIMEOUT_CLEANUP_COOLDOWN_SECONDS", "60")),
    )
    parser.add_argument(
        "--timeout-cleanup-command-timeout-seconds",
        type=float,
        default=float(os.getenv("ENHANCEMENT_TIMEOUT_CLEANUP_COMMAND_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument("--scratch-root", default="/data/mineru-vlm-lab/work/enhancement-worker")
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--kind", default=os.getenv("ENHANCEMENT_KIND") or None)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asyncio.run(
        run_worker_loop(
            orchestrator_base_url=args.orchestrator_base_url,
            mineru_vlm_base_url=args.mineru_vlm_base_url,
            worker_id=args.worker_id,
            limit=args.limit,
            concurrency=args.concurrency,
            lease_timeout_seconds=args.lease_timeout_seconds,
            vlm_timeout_seconds=args.vlm_timeout_seconds,
            timeout_cleanup_command=args.timeout_cleanup_command,
            timeout_cleanup_cooldown_seconds=args.timeout_cleanup_cooldown_seconds,
            timeout_cleanup_command_timeout_seconds=args.timeout_cleanup_command_timeout_seconds,
            scratch_root=Path(args.scratch_root),
            interval_seconds=args.interval_seconds,
            once=args.once,
            job_id=args.job_id,
            kind=args.kind,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
