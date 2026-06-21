from __future__ import annotations

import os
import time

from pathlib import Path

from conftest import make_editable_pdf, make_editable_pdf_with_image, make_scanned_pdf, make_table_candidate_pdf


def wait_for_done(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def make_vector_graphics_candidate_pdf(path: Path) -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 50
    for line in range(12):
        page.insert_text(
            (50, y),
            f"Line {line + 1}: editable paper page with native text and a vector architecture figure. " * 2,
            fontsize=10,
        )
        y += 16
    for index in range(50):
        x = 70 + (index % 10) * 42
        y = 180 + (index // 10) * 34
        page.draw_rect(fitz.Rect(x, y, x + 26, y + 18), color=(0, 0, 0), width=0.5)
    doc.save(path)
    doc.close()
    return path


def make_captioned_vector_graphics_candidate_pdf(path: Path) -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 50
    for line in range(10):
        page.insert_text(
            (50, y),
            f"Line {line + 1}: editable paper page with native text and a captioned vector figure. " * 2,
            fontsize=10,
        )
        y += 16
    for index in range(8):
        x = 90 + index * 48
        page.draw_rect(fitz.Rect(x, 220, x + 32, 250), color=(0, 0, 0), width=0.5)
        page.draw_line((x + 32, 235), (x + 48, 235), color=(0, 0, 0), width=0.5)
    page.insert_text(
        (70, 300),
        "Figure 1: Caption already describes this vector architecture diagram.",
        fontsize=10,
    )
    doc.save(path)
    doc.close()
    return path


def test_async_job_routes_editable_pdf_into_blocks_embeddings_and_storage(app_client, tmp_path):
    pdf_path = make_editable_pdf(tmp_path / "editable.pdf", pages=2)

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/api/v1/jobs",
            files={"file": ("editable.pdf", pdf_file, "application/pdf")},
            data={"max_concurrent_vlm_pages": "2"},
        )

    assert response.status_code == 202, response.text
    created = response.json()
    assert created["status"] in {"queued", "processing"}
    assert created["status_url"] == f"/api/v1/jobs/{created['job_id']}"
    assert created["blocks_url"] == f"/api/v1/jobs/{created['job_id']}/blocks"

    job = wait_for_done(app_client, created["job_id"])
    assert job["status"] == "completed"
    assert job["page_count"] == 2
    assert job["fast_page_count"] == 2
    assert job["vlm_page_count"] == 0
    assert job["block_count"] >= 2
    assert job["embedding_count"] == job["block_count"]
    assert job["elapsed_seconds"] < 2.0
    assert job["timings"]["total_seconds"] == job["elapsed_seconds"]
    assert set(job["timings"]) >= {"routing_seconds", "vlm_seconds", "embedding_seconds", "storage_seconds"}

    blocks_response = app_client.get(f"/api/v1/jobs/{created['job_id']}/blocks")
    assert blocks_response.status_code == 200, blocks_response.text
    blocks = blocks_response.json()["blocks"]
    assert len(blocks) == job["block_count"]
    assert all(block["source"] == "fast_pymupdf" for block in blocks)
    assert all(block["embedding_id"] for block in blocks)
    assert all(block["embedding_dim"] == 16 for block in blocks)


def test_sync_parse_can_persist_blocks_and_embeddings(app_client, tmp_path):
    pdf_path = make_editable_pdf(tmp_path / "sync.pdf", pages=1)

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("sync.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["page_count"] == 1
    assert summary["fast_page_count"] == 1
    assert summary["vlm_page_count"] == 0
    assert summary["block_count"] >= 1
    assert summary["embedding_count"] == summary["block_count"]
    assert summary["timings"]["total_seconds"] == summary["elapsed_seconds"]
    assert summary["timings"]["storage_seconds"] >= 0
    assert summary["storage"]["stored"] is True
    assert summary["storage"]["db_path"].endswith("mineru-vlm-lab.sqlite3")

    blocks_response = app_client.get(f"/api/v1/jobs/{summary['request_id']}/blocks")
    assert blocks_response.status_code == 200, blocks_response.text
    assert blocks_response.json()["total"] == summary["block_count"]


def test_sync_parse_can_skip_embeddings_for_parse_only_benchmarks(app_client, tmp_path):
    pdf_path = make_editable_pdf(tmp_path / "parse-only.pdf", pages=1)

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("parse-only.pdf", pdf_file, "application/pdf")},
            data={"persist": "true", "run_embedding": "false"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["page_count"] == 1
    assert summary["fast_page_count"] == 1
    assert summary["block_count"] >= 1
    assert summary["embedding_count"] == 0
    assert summary["timings"]["embedding_seconds"] == 0
    assert summary["storage"]["stored"] is True

    blocks_response = app_client.get(f"/api/v1/jobs/{summary['request_id']}/blocks")
    assert blocks_response.status_code == 200, blocks_response.text
    blocks = blocks_response.json()["blocks"]
    assert len(blocks) == summary["block_count"]
    assert all(block["embedding_id"] is None for block in blocks)


def test_scan_pages_queue_direct_vlm_page_tasks_without_router(app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("SCAN_OCR_MODE", "off")
    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", pages=2)

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("scan.pdf", pdf_file, "application/pdf")},
            data={"persist": "true", "run_embedding": "false"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["page_count"] == 2
    assert summary["fast_page_count"] == 0
    assert summary["vlm_page_count"] == 2
    assert summary["block_count"] == 2
    assert summary["enhancement_task_count"] == 2
    assert all(block["kind"] == "image" for block in summary["blocks"])
    assert all(block["metadata"]["page_vlm"] is True for block in summary["blocks"])
    assert all(block["metadata"]["force_vlm"] is True for block in summary["blocks"])
    assert all(Path(block["metadata"]["crop_path"]).exists() for block in summary["blocks"])
    assert all(task["kind"] == "image_candidate" for task in summary["enhancement_tasks"])
    assert all(task["metadata"]["page_vlm"] is True for task in summary["enhancement_tasks"])


def test_scan_ocr_mode_extracts_text_and_queues_table_region_without_full_page_vlm(
    app_client, tmp_path, monkeypatch
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tesseract = fake_bin / "tesseract"
    fake_tesseract.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext')\n"
        "rows = [\n"
        "  (5,1,1,1,1,1,90,90,65,18,96,'Table'),\n"
        "  (5,1,1,1,1,2,165,90,20,18,96,'1'),\n"
        "  (5,1,1,1,1,3,195,90,80,18,96,'Summary'),\n"
        "  (5,1,1,1,2,1,90,125,70,18,96,'Metric'),\n"
        "  (5,1,1,1,2,2,245,125,55,18,96,'Value'),\n"
        "  (5,1,1,1,3,1,90,155,70,18,96,'Alpha'),\n"
        "  (5,1,1,1,3,2,245,155,20,18,96,'42'),\n"
        "  (5,1,1,1,4,1,90,230,75,18,96,'Scanned'),\n"
        "  (5,1,1,1,4,2,175,230,65,18,96,'body'),\n"
        "  (5,1,1,1,4,3,250,230,50,18,96,'text'),\n"
        "]\n"
        "for row in rows:\n"
        "    print('\\t'.join(map(str, row)))\n",
        encoding="utf-8",
    )
    fake_tesseract.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("SCAN_OCR_MODE", "tesseract")
    monkeypatch.setenv("SCAN_OCR_MIN_TEXT_CHARS", "10")
    pdf_path = make_scanned_pdf(tmp_path / "scan-ocr.pdf", pages=1)

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("scan-ocr.pdf", pdf_file, "application/pdf")},
            data={"persist": "true", "run_embedding": "false"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["page_count"] == 1
    assert summary["fast_page_count"] == 0
    assert summary["vlm_page_count"] == 1
    assert summary["vlm_pages"][0]["mode"] == "hybrid_ocr"
    assert summary["vlm_pages"][0]["ocr_engine"] == "tesseract"
    assert summary["enhancement_task_count"] == 1
    assert not any(block["metadata"].get("page_vlm") for block in summary["blocks"])
    assert any(block["source"] == "scan_ocr_tesseract" and "Scanned body text" in block["text"] for block in summary["blocks"])

    task = summary["enhancement_tasks"][0]
    assert task["kind"] == "table_candidate"
    assert task["metadata"]["scan_ocr"] is True
    assert task["metadata"]["page_vlm"] is False
    assert Path(task["crop_path"]).exists()


def test_scan_ocr_auto_falls_back_to_full_page_vlm_when_ocr_engine_is_missing(app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "missing-bin"))
    monkeypatch.setenv("SCAN_OCR_MODE", "auto")
    pdf_path = make_scanned_pdf(tmp_path / "scan-no-ocr.pdf", pages=1)

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("scan-no-ocr.pdf", pdf_file, "application/pdf")},
            data={"persist": "true", "run_embedding": "false"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["vlm_pages"][0]["mode"] == "async_direct_vlm"
    assert summary["block_count"] == 1
    assert summary["enhancement_task_count"] == 1
    assert summary["blocks"][0]["metadata"]["page_vlm"] is True


def test_fast_path_merges_dense_editable_lines_before_embedding(app_client, tmp_path):
    import fitz

    pdf_path = tmp_path / "dense-lines.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for line in range(24):
        page.insert_text(
            (50, 70 + line * 18),
            f"Line {line + 1}: editable native text should be chunked before embedding for production throughput.",
            fontsize=10,
        )
    doc.save(pdf_path)
    doc.close()

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("dense-lines.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["fast_page_count"] == 1
    assert summary["vlm_page_count"] == 0
    assert summary["block_count"] < 24
    assert summary["embedding_count"] == summary["block_count"]


def test_fast_path_preserves_vector_graphics_asset_without_default_vlm_task(app_client, tmp_path):
    pdf_path = make_vector_graphics_candidate_pdf(tmp_path / "vector-graphics.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("vector-graphics.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["fast_page_count"] == 1
    vector_blocks = [
        block
        for block in summary["blocks"]
        if block["metadata"].get("vector_graphics_candidate")
    ]
    assert len(vector_blocks) == 1
    assert vector_blocks[0]["metadata"]["image_asset_only"] is True
    assert Path(vector_blocks[0]["metadata"]["crop_path"]).exists()
    vector_tasks = [
        task
        for task in summary["enhancement_tasks"]
        if task["metadata"].get("vector_graphics_candidate")
    ]
    assert vector_tasks == []


def test_fast_path_can_queue_vector_graphics_vlm_when_enabled(app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_IMAGE_VLM_MODE", "complex")
    pdf_path = make_vector_graphics_candidate_pdf(tmp_path / "vector-graphics-vlm.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("vector-graphics-vlm.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    vector_tasks = [
        task
        for task in summary["enhancement_tasks"]
        if task["metadata"].get("vector_graphics_candidate")
    ]
    assert len(vector_tasks) == 1
    assert vector_tasks[0]["kind"] == "image_candidate"


def test_complex_uncaptioned_mode_skips_captioned_vector_graphics_vlm(app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_IMAGE_VLM_MODE", "complex_uncaptioned")
    pdf_path = make_captioned_vector_graphics_candidate_pdf(tmp_path / "captioned-vector.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("captioned-vector.pdf", pdf_file, "application/pdf")},
            data={"persist": "true", "run_embedding": "false"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    vector_blocks = [
        block
        for block in summary["blocks"]
        if block["metadata"].get("vector_graphics_candidate")
    ]
    assert len(vector_blocks) == 1
    assert vector_blocks[0]["metadata"]["figure_caption_candidate"] is True
    assert vector_blocks[0]["metadata"]["image_asset_only"] is True
    assert Path(vector_blocks[0]["metadata"]["crop_path"]).exists()
    vector_tasks = [
        task
        for task in summary["enhancement_tasks"]
        if task["metadata"].get("vector_graphics_candidate")
    ]
    assert vector_tasks == []


def test_complex_uncaptioned_mode_queues_uncaptioned_vector_graphics_vlm(app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_IMAGE_VLM_MODE", "complex_uncaptioned")
    pdf_path = make_vector_graphics_candidate_pdf(tmp_path / "uncaptioned-vector.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("uncaptioned-vector.pdf", pdf_file, "application/pdf")},
            data={"persist": "true", "run_embedding": "false"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    vector_tasks = [
        task
        for task in summary["enhancement_tasks"]
        if task["metadata"].get("vector_graphics_candidate")
    ]
    assert len(vector_tasks) == 1
    assert vector_tasks[0]["kind"] == "image_candidate"


def test_editable_table_candidate_creates_nonblocking_enhancement_task(app_client, tmp_path):
    pdf_path = make_table_candidate_pdf(tmp_path / "table.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("table.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["page_count"] == 1
    assert summary["fast_page_count"] == 1
    assert summary["vlm_page_count"] == 0
    assert summary["enhancement_task_count"] >= 1
    assert summary["enhancement_tasks"][0]["status"] == "queued"
    assert summary["enhancement_tasks"][0]["kind"] == "table_candidate"
    assert summary["enhancement_tasks"][0]["crop_path"].endswith(".png")

    tasks_response = app_client.get(f"/api/v1/jobs/{summary['request_id']}/enhancements")
    assert tasks_response.status_code == 200, tasks_response.text
    tasks = tasks_response.json()["tasks"]
    assert len(tasks) == summary["enhancement_task_count"]
    assert tasks[0]["block_id"] == summary["enhancement_tasks"][0]["block_id"]

    claim_response = app_client.post(
        "/api/v1/enhancements/claim",
        json={"limit": 1, "worker_id": "test-worker", "job_id": summary["request_id"]},
    )
    assert claim_response.status_code == 200, claim_response.text
    claimed = claim_response.json()["tasks"]
    assert len(claimed) == 1
    assert claimed[0]["status"] == "processing"

    complete_response = app_client.post(
        f"/api/v1/enhancements/{claimed[0]['task_id']}/complete",
        json={"worker_id": "test-worker", "result": {"markdown": "| A | B |"}},
    )
    assert complete_response.status_code == 200, complete_response.text
    assert complete_response.json()["status"] == "completed"

    tasks_response = app_client.get(f"/api/v1/jobs/{summary['request_id']}/enhancements")
    tasks = tasks_response.json()["tasks"]
    completed = [task for task in tasks if task["task_id"] == claimed[0]["task_id"]][0]
    assert completed["status"] == "completed"
    assert completed["result"]["markdown"] == "| A | B |"


def test_expired_enhancement_claim_can_be_recovered_by_another_worker(app_client, tmp_path):
    pdf_path = make_table_candidate_pdf(tmp_path / "recoverable-table.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("recoverable-table.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    first_claim = app_client.post(
        "/api/v1/enhancements/claim",
        json={"limit": 1, "worker_id": "dead-worker", "job_id": summary["request_id"]},
    )
    assert first_claim.status_code == 200, first_claim.text
    first_task = first_claim.json()["tasks"][0]
    assert first_task["status"] == "processing"
    assert first_task["worker_id"] == "dead-worker"

    recovered_claim = app_client.post(
        "/api/v1/enhancements/claim",
        json={
            "limit": 1,
            "worker_id": "recovery-worker",
            "job_id": summary["request_id"],
            "lease_timeout_seconds": 0,
        },
    )

    assert recovered_claim.status_code == 200, recovered_claim.text
    recovered_tasks = recovered_claim.json()["tasks"]
    assert len(recovered_tasks) == 1
    assert recovered_tasks[0]["task_id"] == first_task["task_id"]
    assert recovered_tasks[0]["status"] == "processing"
    assert recovered_tasks[0]["worker_id"] == "recovery-worker"


def test_stats_endpoint_reports_jobs_pages_embeddings_and_enhancement_backlog(app_client, tmp_path):
    editable_pdf = make_editable_pdf(tmp_path / "stats-editable.pdf", pages=2)
    table_pdf = make_table_candidate_pdf(tmp_path / "stats-table.pdf")

    with editable_pdf.open("rb") as pdf_file:
        editable_response = app_client.post(
            "/parse",
            files={"file": ("stats-editable.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )
    assert editable_response.status_code == 200, editable_response.text

    with table_pdf.open("rb") as pdf_file:
        table_response = app_client.post(
            "/parse",
            files={"file": ("stats-table.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )
    assert table_response.status_code == 200, table_response.text
    table_summary = table_response.json()

    claim_response = app_client.post(
        "/api/v1/enhancements/claim",
        json={"limit": 1, "worker_id": "stats-worker", "job_id": table_summary["request_id"]},
    )
    assert claim_response.status_code == 200, claim_response.text

    stats_response = app_client.get("/api/v1/stats")

    assert stats_response.status_code == 200, stats_response.text
    stats = stats_response.json()
    assert stats["jobs"]["total"] == 2
    assert stats["jobs"]["completed"] == 2
    assert stats["pages"]["total"] == 3
    assert stats["pages"]["fast"] == 3
    assert stats["pages"]["vlm"] == 0
    assert stats["blocks"]["total"] >= 3
    assert stats["embeddings"]["total"] == stats["blocks"]["total"]
    assert stats["enhancements"]["total"] == table_summary["enhancement_task_count"]
    assert stats["enhancements"]["processing"] == 1
    assert stats["enhancements"]["queued"] == table_summary["enhancement_task_count"] - 1


def test_fast_path_preserves_raster_image_asset_without_default_vlm_task(app_client, tmp_path):
    pdf_path = make_editable_pdf_with_image(tmp_path / "image-page.pdf")

    with pdf_path.open("rb") as pdf_file:
        response = app_client.post(
            "/parse",
            files={"file": ("image-page.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )

    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["page_count"] == 1
    assert summary["fast_page_count"] == 1
    image_blocks = [block for block in summary["blocks"] if block["kind"] == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["bbox"]
    assert image_blocks[0]["metadata"]["image_candidate"] is True
    assert image_blocks[0]["metadata"]["image_asset_only"] is True
    assert Path(image_blocks[0]["metadata"]["crop_path"]).exists()
    image_tasks = [task for task in summary["enhancement_tasks"] if task["kind"] == "image_candidate"]
    assert image_tasks == []


def test_enhancement_claim_can_filter_by_kind(app_client, tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_IMAGE_VLM_MODE", "complex")
    table_pdf = make_table_candidate_pdf(tmp_path / "claim-table.pdf")
    image_pdf = make_vector_graphics_candidate_pdf(tmp_path / "claim-image.pdf")

    with table_pdf.open("rb") as pdf_file:
        table_response = app_client.post(
            "/parse",
            files={"file": ("claim-table.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )
    assert table_response.status_code == 200, table_response.text

    with image_pdf.open("rb") as pdf_file:
        image_response = app_client.post(
            "/parse",
            files={"file": ("claim-image.pdf", pdf_file, "application/pdf")},
            data={"persist": "true"},
        )
    assert image_response.status_code == 200, image_response.text

    image_claim = app_client.post(
        "/api/v1/enhancements/claim",
        json={"limit": 5, "worker_id": "image-worker", "kind": "image_candidate"},
    )
    assert image_claim.status_code == 200, image_claim.text
    image_tasks = image_claim.json()["tasks"]
    assert len(image_tasks) == 1
    assert image_tasks[0]["kind"] == "image_candidate"

    table_claim = app_client.post(
        "/api/v1/enhancements/claim",
        json={"limit": 5, "worker_id": "table-worker", "kind": "table_candidate"},
    )
    assert table_claim.status_code == 200, table_claim.text
    table_tasks = table_claim.json()["tasks"]
    assert len(table_tasks) >= 1
    assert all(task["kind"] == "table_candidate" for task in table_tasks)
