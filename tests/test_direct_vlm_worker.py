from __future__ import annotations

import json
from pathlib import Path

import fitz
import httpx
import pytest

from app.direct_vlm_worker import build_vllm_engine_kwargs, parse_args, run_direct_worker_once


def make_crop(path: Path, *, label: str) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=240, height=120)
    page.draw_rect(fitz.Rect(20, 20, 220, 100), color=(0.1, 0.35, 0.8), fill=(0.85, 0.9, 1.0))
    page.insert_textbox(fitz.Rect(35, 42, 205, 78), label, fontsize=18, fontname="helv")
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(str(path))
    doc.close()
    return path


class FakePredictor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def aio_batch_two_step_extract(self, *, images, image_analysis: bool):
        self.calls.append(
            {
                "count": len(images),
                "sizes": [image.size for image in images],
                "image_analysis": image_analysis,
            }
        )
        return [
            [{"type": "text", "content": "first crop"}],
            [{"type": "text", "content": "second crop"}],
        ]


def test_direct_worker_defaults_to_general_limit_four(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_direct_vlm_worker.py"])
    args = parse_args()

    assert args.limit == 4
    assert args.kind is None


def test_build_vllm_engine_kwargs_includes_scheduler_and_kv_options():
    kwargs = build_vllm_engine_kwargs(
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        kv_cache_dtype="fp8",
        compilation_config='{"level": 3}',
    )

    assert kwargs == {
        "gpu_memory_utilization": 0.95,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 16,
        "kv_cache_dtype": "fp8",
        "compilation_config": '{"level": 3}',
    }


@pytest.mark.asyncio
async def test_direct_worker_batches_claimed_images_and_completes_results(tmp_path):
    crop_one = make_crop(tmp_path / "crop-one.png", label="one")
    crop_two = make_crop(tmp_path / "crop-two.png", label="two")
    predictor = FakePredictor()
    claim_payloads: list[dict] = []
    completed: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            claim_payloads.append(json.loads((await request.aread()).decode("utf-8")))
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
                            "kind": "image_candidate",
                            "status": "processing",
                            "crop_path": str(crop_one),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {},
                        },
                        {
                            "task_id": "task-2",
                            "job_id": "job-1",
                            "block_id": "block-2",
                            "page_index": 1,
                            "kind": "image_candidate",
                            "status": "processing",
                            "crop_path": str(crop_two),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {},
                        },
                    ],
                },
            )
        if str(request.url).startswith("http://orchestrator/api/v1/enhancements/") and str(request.url).endswith("/complete"):
            completed.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_direct_worker_once(
            orchestrator_base_url="http://orchestrator",
            worker_id="direct-gpu4",
            limit=2,
            kind="image_candidate",
            predictor=predictor,
            max_image_width=64,
            client=client,
        )

    assert summary == {"claimed": 2, "completed": 2, "failed": 0}
    assert claim_payloads == [{"limit": 2, "worker_id": "direct-gpu4", "kind": "image_candidate"}]
    assert predictor.calls == [{"count": 2, "sizes": [(64, 32), (64, 32)], "image_analysis": True}]
    assert [payload["worker_id"] for payload in completed] == ["direct-gpu4", "direct-gpu4"]
    assert completed[0]["result"]["backend"] == "mineru_direct_pil"
    assert completed[0]["result"]["content_items"][0]["content"] == "first crop"
    assert completed[0]["result"]["markdown"] == "first crop"


@pytest.mark.asyncio
async def test_direct_worker_uses_table_specific_image_width(tmp_path):
    crop_one = make_crop(tmp_path / "image-crop.png", label="image")
    crop_two = make_crop(tmp_path / "table-crop.png", label="table")
    predictor = FakePredictor()
    completed: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
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
                            "kind": "image_candidate",
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
                            "bbox": [0, 0, 100, 100],
                            "metadata": {"native_table_needs_vlm": True},
                        },
                    ],
                },
            )
        if str(request.url).startswith("http://orchestrator/api/v1/enhancements/") and str(request.url).endswith("/complete"):
            completed.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_direct_worker_once(
            orchestrator_base_url="http://orchestrator",
            worker_id="direct-gpu4",
            limit=2,
            kind=None,
            predictor=predictor,
            max_image_width=64,
            max_table_image_width=160,
            client=client,
        )

    assert summary == {"claimed": 2, "completed": 2, "failed": 0}
    assert predictor.calls == [{"count": 2, "sizes": [(64, 32), (160, 80)], "image_analysis": True}]
    assert completed[0]["result"]["max_image_width"] == 64
    assert completed[1]["result"]["max_image_width"] == 160


@pytest.mark.asyncio
async def test_direct_worker_honors_task_level_image_width_override(tmp_path):
    crop = make_crop(tmp_path / "large-table-crop.png", label="table")
    predictor = FakePredictor()
    completed: list[dict] = []

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
                            "crop_path": str(crop),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {"max_image_width": 192},
                        }
                    ],
                },
            )
        if str(request.url).startswith("http://orchestrator/api/v1/enhancements/") and str(request.url).endswith("/complete"):
            completed.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_direct_worker_once(
            orchestrator_base_url="http://orchestrator",
            worker_id="direct-gpu4",
            limit=1,
            kind=None,
            predictor=predictor,
            max_image_width=64,
            max_table_image_width=160,
            client=client,
        )

    assert summary == {"claimed": 1, "completed": 1, "failed": 0}
    assert predictor.calls == [{"count": 1, "sizes": [(192, 96)], "image_analysis": True}]
    assert completed[0]["result"]["max_image_width"] == 192


@pytest.mark.asyncio
async def test_direct_worker_uses_page_specific_image_width(tmp_path):
    crop_one = make_crop(tmp_path / "image-crop.png", label="image")
    crop_two = make_crop(tmp_path / "page-crop.png", label="page")
    predictor = FakePredictor()
    completed: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
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
                            "kind": "image_candidate",
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
                            "kind": "image_candidate",
                            "status": "processing",
                            "crop_path": str(crop_two),
                            "bbox": [0, 0, 100, 100],
                            "metadata": {"page_vlm": True},
                        },
                    ],
                },
            )
        if str(request.url).startswith("http://orchestrator/api/v1/enhancements/") and str(request.url).endswith("/complete"):
            completed.append(json.loads((await request.aread()).decode("utf-8")))
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_direct_worker_once(
            orchestrator_base_url="http://orchestrator",
            worker_id="direct-gpu4",
            limit=2,
            kind=None,
            predictor=predictor,
            max_image_width=64,
            max_page_image_width=200,
            client=client,
        )

    assert summary == {"claimed": 2, "completed": 2, "failed": 0}
    assert predictor.calls == [{"count": 2, "sizes": [(64, 32), (200, 100)], "image_analysis": True}]
    assert completed[0]["result"]["max_image_width"] == 64
    assert completed[1]["result"]["max_image_width"] == 200


@pytest.mark.asyncio
async def test_direct_worker_keeps_running_when_claim_endpoint_is_temporarily_unavailable():
    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://orchestrator/api/v1/enhancements/claim":
            raise httpx.ConnectError("orchestrator restarting", request=request)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        summary = await run_direct_worker_once(
            orchestrator_base_url="http://orchestrator",
            worker_id="direct-gpu4",
            limit=2,
            kind="image_candidate",
            predictor=FakePredictor(),
            client=client,
        )

    assert summary == {"claimed": 0, "completed": 0, "failed": 0, "claim_errors": 1}
