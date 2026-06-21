from __future__ import annotations

import io
import json
import asyncio
import shlex
import sys
import zipfile
from pathlib import Path

import fitz
import httpx
import pytest

from app.enhancement_worker import image_to_pdf, run_worker_loop, run_worker_once


def make_png(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=200, height=120)
    page.insert_textbox(fitz.Rect(12, 12, 188, 108), "A   B\n1   2", fontsize=16, fontname="cour")
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(str(path))
    doc.close()
    return path


def make_vlm_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("input/vlm/input.md", "| A | B |\n|---|---|\n| 1 | 2 |\n")
        archive.writestr("input/vlm/input_content_list.json", json.dumps([{"type": "table", "html": "<table></table>"}]))
    return buffer.getvalue()


def test_image_to_pdf_wraps_crop(tmp_path):
    image_path = make_png(tmp_path / "crop.png")
    pdf_path = image_to_pdf(image_path, tmp_path / "crop.pdf")

    doc = fitz.open(pdf_path)
    try:
        assert doc.page_count == 1
        assert doc.load_page(0).rect.width > 0
        assert doc.load_page(0).rect.height > 0
    finally:
        doc.close()


@pytest.mark.asyncio
async def test_worker_claims_crop_calls_vlm_and_completes(tmp_path):
    crop_path = make_png(tmp_path / "crop.png")
    completed_payloads: list[dict] = []
    vlm_seen = {"called": False}

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            return httpx.Response(
                200,
                json={
                    "total": 1,
                    "tasks": [
                        {
                            "task_id": "task-1",
                            "job_id": "job-1",
                            "block_id": "block-1",
                            "page_index": 0,
                            "kind": "table_candidate",
                            "status": "processing",
                            "crop_path": str(crop_path),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {},
                        }
                    ],
                },
            )
        if str(request.url) == "http://vlm/file_parse":
            body = await request.aread()
            assert b"crop.pdf" in body
            vlm_seen["called"] = True
            return httpx.Response(200, content=make_vlm_zip(), headers={"content-type": "application/zip"})
        if str(request.url) == "http://orchestrator/api/v1/enhancements/task-1/complete":
            payload = json.loads((await request.aread()).decode("utf-8"))
            completed_payloads.append(payload)
            return httpx.Response(200, json={"task_id": "task-1", "status": "completed", "result": payload["result"]})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_worker_once(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=1,
            scratch_root=tmp_path / "scratch",
            client=client,
        )

    assert summary == {"claimed": 1, "completed": 1, "failed": 0}
    assert vlm_seen["called"] is True
    assert completed_payloads[0]["worker_id"] == "worker-1"
    result = completed_payloads[0]["result"]
    assert result["status_code"] == 200
    assert result["markdown"] == "| A | B |\n|---|---|\n| 1 | 2 |\n"
    assert result["content_items"][0]["type"] == "table"


@pytest.mark.asyncio
async def test_worker_loop_passes_job_id_to_claim(tmp_path):
    claim_payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            claim_payloads.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"total": 0, "tasks": []})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        await run_worker_loop(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=1,
            scratch_root=tmp_path / "scratch",
            interval_seconds=0.01,
            once=True,
            job_id="job-filter",
            client=client,
        )

    assert claim_payloads == [{"limit": 1, "worker_id": "worker-1", "job_id": "job-filter"}]


@pytest.mark.asyncio
async def test_worker_passes_lease_timeout_to_claim_when_configured(tmp_path):
    claim_payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            claim_payloads.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"total": 0, "tasks": []})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        await run_worker_once(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=4,
            scratch_root=tmp_path / "scratch",
            lease_timeout_seconds=30,
            client=client,
        )

    assert claim_payloads == [{"limit": 4, "worker_id": "worker-1", "lease_timeout_seconds": 30}]


@pytest.mark.asyncio
async def test_worker_passes_kind_to_claim_when_configured(tmp_path):
    claim_payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            claim_payloads.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"total": 0, "tasks": []})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        await run_worker_once(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=3,
            scratch_root=tmp_path / "scratch",
            kind="table_candidate",
            client=client,
        )

    assert claim_payloads == [{"limit": 3, "worker_id": "worker-1", "kind": "table_candidate"}]


@pytest.mark.asyncio
async def test_worker_processes_claimed_tasks_concurrently(tmp_path):
    crop_one = make_png(tmp_path / "crop-one.png")
    crop_two = make_png(tmp_path / "crop-two.png")
    active = 0
    max_active = 0
    completed: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            return httpx.Response(
                200,
                json={
                    "total": 2,
                    "tasks": [
                        {
                            "task_id": "task-1",
                            "job_id": "job-1",
                            "block_id": "block-1",
                            "page_index": 0,
                            "kind": "table_candidate",
                            "status": "processing",
                            "crop_path": str(crop_one),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {},
                        },
                        {
                            "task_id": "task-2",
                            "job_id": "job-1",
                            "block_id": "block-2",
                            "page_index": 0,
                            "kind": "table_candidate",
                            "status": "processing",
                            "crop_path": str(crop_two),
                            "bbox": [100, 0, 200, 100],
                            "metadata": {},
                        },
                    ],
                },
            )
        if str(request.url) == "http://vlm/file_parse":
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return httpx.Response(200, content=make_vlm_zip(), headers={"content-type": "application/zip"})
        if str(request.url).startswith("http://orchestrator/api/v1/enhancements/") and str(request.url).endswith("/complete"):
            completed.append(str(request.url).split("/")[-2])
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_worker_once(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=2,
            concurrency=2,
            scratch_root=tmp_path / "scratch",
            client=client,
        )

    assert summary == {"claimed": 2, "completed": 2, "failed": 0}
    assert sorted(completed) == ["task-1", "task-2"]
    assert max_active == 2


@pytest.mark.asyncio
async def test_worker_fails_task_when_vlm_call_times_out(tmp_path):
    crop_path = make_png(tmp_path / "crop.png")
    failed_payloads: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            return httpx.Response(
                200,
                json={
                    "total": 1,
                    "tasks": [
                        {
                            "task_id": "task-timeout",
                            "job_id": "job-1",
                            "block_id": "block-1",
                            "page_index": 0,
                            "kind": "table_candidate",
                            "status": "processing",
                            "crop_path": str(crop_path),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {},
                        }
                    ],
                },
            )
        if str(request.url) == "http://vlm/file_parse":
            await asyncio.sleep(0.2)
            return httpx.Response(200, content=make_vlm_zip(), headers={"content-type": "application/zip"})
        if str(request.url) == "http://orchestrator/api/v1/enhancements/task-timeout/fail":
            failed_payloads.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"task_id": "task-timeout", "status": "failed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_worker_once(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=1,
            scratch_root=tmp_path / "scratch",
            vlm_timeout_seconds=0.01,
            client=client,
        )

    assert summary == {"claimed": 1, "completed": 0, "failed": 1}
    assert failed_payloads[0]["worker_id"] == "worker-1"
    assert "timed out after 0.01s" in failed_payloads[0]["error"]


@pytest.mark.asyncio
async def test_worker_runs_timeout_cleanup_command_once_for_concurrent_timeouts(tmp_path):
    crop_one = make_png(tmp_path / "crop-one.png")
    crop_two = make_png(tmp_path / "crop-two.png")
    marker = tmp_path / "cleanup-marker.txt"
    failed: list[str] = []
    cleanup_script = (
        "from pathlib import Path\n"
        f"p = Path({str(marker)!r})\n"
        "p.write_text((p.read_text() if p.exists() else '') + 'ran\\n')\n"
    )
    cleanup_command = f"{shlex.quote(sys.executable)} -c {shlex.quote(cleanup_script)}"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            return httpx.Response(
                200,
                json={
                    "total": 2,
                    "tasks": [
                        {
                            "task_id": "task-timeout-1",
                            "job_id": "job-1",
                            "block_id": "block-1",
                            "page_index": 0,
                            "kind": "table_candidate",
                            "status": "processing",
                            "crop_path": str(crop_one),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {},
                        },
                        {
                            "task_id": "task-timeout-2",
                            "job_id": "job-1",
                            "block_id": "block-2",
                            "page_index": 0,
                            "kind": "table_candidate",
                            "status": "processing",
                            "crop_path": str(crop_two),
                            "bbox": [100, 0, 200, 100],
                            "metadata": {},
                        },
                    ],
                },
            )
        if str(request.url) == "http://vlm/file_parse":
            await asyncio.sleep(0.2)
            return httpx.Response(200, content=make_vlm_zip(), headers={"content-type": "application/zip"})
        if str(request.url).startswith("http://orchestrator/api/v1/enhancements/") and str(request.url).endswith("/fail"):
            failed.append(str(request.url).split("/")[-2])
            return httpx.Response(200, json={"status": "failed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_worker_once(
            orchestrator_base_url="http://orchestrator",
            mineru_vlm_base_url="http://vlm",
            worker_id="worker-1",
            limit=2,
            concurrency=2,
            scratch_root=tmp_path / "scratch",
            vlm_timeout_seconds=0.01,
            timeout_cleanup_command=cleanup_command,
            timeout_cleanup_cooldown_seconds=60,
            client=client,
        )

    assert summary == {"claimed": 2, "completed": 0, "failed": 2}
    assert sorted(failed) == ["task-timeout-1", "task-timeout-2"]
    assert marker.read_text() == "ran\n"
