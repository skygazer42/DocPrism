from __future__ import annotations

import torch
import fitz
from PIL import Image

from app.embedding import HashEmbeddingProvider, TransformersEmbeddingProvider, build_embedding_provider
from app.models import BlockRecord, PageSignal
from app.markdown_export import render_markdown
from app.pipeline import create_enhancement_tasks
from app.routing import classify_page, extract_fast_page, looks_like_table_text, native_table_needs_vlm, native_table_to_markdown


def test_page_router_separates_editable_scan_and_table_pages():
    assert classify_page(
        PageSignal(
            page_index=0,
            text_chars=1200,
            block_count=10,
            image_count=0,
            image_area_ratio=0.01,
            has_table_hint=False,
        )
    ) == ("fast_pymupdf", "editable_text_dense")

    assert classify_page(
        PageSignal(
            page_index=1,
            text_chars=0,
            block_count=0,
            image_count=1,
            image_area_ratio=0.92,
            has_table_hint=False,
        )
    ) == ("vlm", "scan_or_image_heavy")

    assert classify_page(
        PageSignal(
            page_index=2,
            text_chars=120,
            block_count=8,
            image_count=0,
            image_area_ratio=0.0,
            has_table_hint=True,
        )
    ) == ("vlm", "table_hint")

    assert classify_page(
        PageSignal(
            page_index=3,
            text_chars=1200,
            block_count=20,
            image_count=0,
            image_area_ratio=0.0,
            has_table_hint=True,
        )
    ) == ("fast_pymupdf", "editable_text_dense_table_candidate")


def test_page_level_table_reason_does_not_mark_every_text_block_as_table():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 60, 545, 180),
        "This is normal paragraph text before the table. It should stay native text only.",
        fontsize=12,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 240, 545, 420),
        (
            "Name          Qty          Price          Total\n"
            "Alpha         12           3.50           42.00\n"
            "Beta          5            9.00           45.00\n"
        ),
        fontsize=12,
        fontname="cour",
    )

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense_table_candidate")

    text_blocks = [block for block in blocks if block.kind == "text"]
    table_blocks = [block for block in text_blocks if block.metadata.get("table_candidate")]
    assert len(text_blocks) == 2
    assert len(table_blocks) == 1
    assert "Alpha" in table_blocks[0].text
    doc.close()


def test_formula_spacing_is_not_treated_as_table_candidate():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 80, 545, 160),
        "mT2(pT,1, pT,2, qT) = min    qT,1+qT,2=qT    max[ mT(pT,1, qT,1), mT(pT,2, qT,2) ]",
        fontsize=12,
        fontname="helv",
    )

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense_table_candidate")

    assert blocks
    assert all(not block.metadata.get("table_candidate") for block in blocks)
    doc.close()


def test_figure_caption_allows_vector_asset_below_global_threshold():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 60, 545, 160),
        "Editable body text before a vector chart. " * 20,
        fontsize=10,
        fontname="helv",
    )
    for index in range(4):
        x = 80 + index * 55
        page.draw_rect(fitz.Rect(x, 190, x + 35, 240), color=(0, 0, 0), width=0.7)
        page.draw_line((x, 240), (x + 45, 170), color=(0, 0, 0), width=0.7)
    page.insert_text((80, 270), "Figure 1. A compact vector chart.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(50, 320, 545, 760),
        "Editable body text after the vector chart. " * 30,
        fontsize=10,
        fontname="helv",
    )

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense")

    image_blocks = [block for block in blocks if block.kind == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0].metadata["figure_caption_candidate"] is True
    assert image_blocks[0].metadata["vector_graphics_candidate"] is True
    doc.close()


def test_table_text_detection_requires_multiple_table_rows():
    formula = (
        "mT2(pT,1, pT,2, qT) = min    qT,1+qT,2=qT    "
        "max[ mT(pT,1, qT,1), mT(pT,2, qT,2) ]"
    )
    table = (
        "Name          Qty          Price          Total\n"
        "Alpha         12           3.50           42.00\n"
        "Beta          5            9.00           45.00\n"
    )

    assert looks_like_table_text(formula) is False
    assert looks_like_table_text(table) is True


def add_grid_table(page: fitz.Page) -> None:
    page.insert_text((50, 55), "Table 1. ImageNet validation errors.", fontsize=11)
    xs = [50, 180, 280, 390, 500]
    ys = [80, 112, 144, 176]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), color=(0, 0, 0), width=0.7)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), color=(0, 0, 0), width=0.7)
    rows = [
        ["Method", "Top-1", "Top-5", "Params"],
        ["ResNet-50", "22.85", "6.71", "25.6M"],
        ["ResNet-101", "21.75", "6.05", "44.5M"],
    ]
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            page.insert_text((xs[col_index] + 4, ys[row_index] + 20), value, fontsize=10)


def test_fast_path_emits_native_table_markdown_for_grid_tables():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    add_grid_table(page)

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense")

    native_tables = [block for block in blocks if block.kind == "native_table"]
    assert len(native_tables) == 1
    assert native_tables[0].metadata["native_table"] is True
    assert "| Method | Top-1 | Top-5 | Params |" in native_tables[0].text
    assert "| ResNet-50 | 22.85 | 6.71 | 25.6M |" in native_tables[0].text
    assert all("ResNet-50" not in block.text for block in blocks if block.kind == "text")
    doc.close()


def test_native_table_markdown_expands_multiline_cells_into_rows():
    markdown = native_table_to_markdown(
        [
            ["system", "net", "mAP", "cat dog"],
            [
                "baseline\nbaseline+++",
                "VGG-16\nResNet-101",
                "73.2\n85.6",
                "84.7 86.4\n89.9 90.3",
            ],
        ]
    )

    assert "| baseline | VGG-16 | 73.2 | 84.7 86.4 |" in markdown
    assert "| baseline+++ | ResNet-101 | 85.6 | 89.9 90.3 |" in markdown
    assert "| baseline baseline+++ |" not in markdown


def test_markdown_export_reflows_inline_text_table_fragments():
    text = "\n".join(
        [
            "Body before the result table.",
            "Table 2. Segmentation results (IOU) on the ISBI cell tracking challenge 2015.",
            "Name PhC-U373 DIC-HeLa",
            "IMCB-SG (2014) 0.2669 0.2935 KTH-SE (2014) 0.7953 0.4607 u-net (2015) 0.9203 0.7756",
            "Body after the table should stay as text.",
        ]
    )

    markdown = render_markdown(
        title="U-Net",
        source_pdf="u-net.pdf",
        job_id="job-1",
        page_count=1,
        blocks=[
            {
                "block_id": "block-1",
                "page_index": 0,
                "block_index": 0,
                "kind": "text",
                "text": text,
                "metadata": {},
            }
        ],
        enhancement_tasks=[],
    )

    assert "| Name | PhC-U373 | DIC-HeLa |" in markdown
    assert "| IMCB-SG (2014) | 0.2669 | 0.2935 |" in markdown
    assert "| KTH-SE (2014) | 0.7953 | 0.4607 |" in markdown
    assert "| u-net (2015) | 0.9203 | 0.7756 |" in markdown
    assert "Body after the table should stay as text." in markdown


def test_native_table_noise_detection_flags_corrupt_network_tables():
    noisy = "|  | 3 3× ×3 3, 6 64 | 1×1, 64   3×3, 64 × 1×1, 256 |"
    clean = "| method | top-1 err. |\n| --- | --- |\n| ResNet-50 | 22.85 |"

    assert native_table_needs_vlm(noisy) is True
    assert native_table_needs_vlm(clean) is False


def test_noisy_native_table_creates_table_enhancement_task(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    block = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="fast_pymupdf",
        kind="native_table",
        text="| noisy | table |\n| --- | --- |\n| 3 3× ×3 3 | bad |",
        bbox=[50, 80, 300, 180],
        metadata={"native_table": True, "native_table_needs_vlm": True},
    )

    tasks = create_enhancement_tasks(page, [block], tmp_path / "enhancements")

    assert len(tasks) == 1
    assert tasks[0].kind == "table_candidate"
    assert tasks[0].metadata["native_table_needs_vlm"] is True
    assert block.metadata["crop_path"].endswith(".png")
    assert block.metadata["crop_bbox"][0] <= 20
    assert block.metadata["crop_bbox"][1] <= 50
    with Image.open(block.metadata["crop_path"]) as crop:
        assert crop.width >= 900
    doc.close()


def test_enhancement_priority_marks_required_scan_tables(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    block = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="scan_ocr_tesseract",
        kind="native_table",
        text="Name Qty\nAlpha 12",
        bbox=[50, 80, 300, 180],
        metadata={"scan_ocr": True, "table_candidate": True, "native_table_needs_vlm": True},
    )

    tasks = create_enhancement_tasks(page, [block], tmp_path / "enhancements")

    assert block.metadata["enhancement_priority"] == "required"
    assert tasks[0].metadata["enhancement_priority"] == "required"
    doc.close()


def test_enhancement_priority_marks_editable_complex_tables_optional(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    block = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="fast_pymupdf",
        kind="text",
        text="Table 1. Complex table\nResNet-50 22.85",
        bbox=[50, 80, 300, 180],
        metadata={"table_candidate": True, "native_table_deferred": True},
    )

    tasks = create_enhancement_tasks(page, [block], tmp_path / "enhancements")

    assert block.metadata["enhancement_priority"] == "optional"
    assert tasks[0].metadata["enhancement_priority"] == "optional"
    doc.close()


def test_enhancement_task_marks_simple_and_complex_table_image_widths(tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH", "640")
    monkeypatch.setenv("ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH", "768")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    simple = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="fast_pymupdf",
        kind="text",
        text="Table 1. Accuracy\nResNet-50 22.85 6.71\nResNet-101 21.75 6.05",
        bbox=[50, 80, 300, 180],
        metadata={"table_candidate": True, "native_table_deferred": True},
    )
    complex_table = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=1,
        source="fast_pymupdf",
        kind="native_table",
        text="| noisy | table |\n| --- | --- |\n| 3 3× ×3 3 | bad |",
        bbox=[50, 220, 300, 340],
        metadata={"native_table": True, "native_table_needs_vlm": True},
    )

    tasks = create_enhancement_tasks(page, [simple, complex_table], tmp_path / "enhancements")

    assert [task.metadata["max_image_width"] for task in tasks] == [640, 768]
    assert simple.metadata["max_image_width"] == 640
    assert complex_table.metadata["max_image_width"] == 768
    doc.close()


def test_optional_wide_table_stays_on_default_image_width(tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH", "640")
    monkeypatch.setenv("ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH", "768")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    block = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="fast_pymupdf",
        kind="text",
        text="Table 2. Short wide table\nATLAS 95% CL",
        bbox=[20, 80, 575, 520],
        metadata={"table_candidate": True, "native_table_deferred": True},
    )

    tasks = create_enhancement_tasks(page, [block], tmp_path / "enhancements")

    assert tasks[0].metadata["max_image_width"] == 640
    doc.close()


def test_optional_multiline_table_stays_on_default_image_width(tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH", "640")
    monkeypatch.setenv("ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH", "768")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    block = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="fast_pymupdf",
        kind="text",
        text="\n".join(
            [
                "Table 5. Ablation results.",
                "setting A B C",
                "row 1 10 20 30",
                "row 2 11 21 31",
                "row 3 12 22 32",
                "row 4 13 23 33",
                "row 5 14 24 34",
                "row 6 15 25 35",
                "row 7 16 26 36",
                "row 8 17 27 37",
                "row 9 18 28 38",
            ]
        ),
        bbox=[20, 80, 575, 520],
        metadata={"table_candidate": True, "native_table_deferred": True},
    )

    tasks = create_enhancement_tasks(page, [block], tmp_path / "enhancements")

    assert tasks[0].metadata["enhancement_priority"] == "optional"
    assert tasks[0].metadata["max_image_width"] == 640
    doc.close()


def test_enhancement_priority_marks_captioned_image_assets_asset_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ENHANCEMENT_IMAGE_VLM_MODE", "complex_uncaptioned")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    block = BlockRecord(
        job_id="job-1",
        page_index=0,
        block_index=0,
        source="fast_pymupdf",
        kind="image",
        text="[vector graphics page=0]",
        bbox=[50, 80, 300, 180],
        metadata={
            "image_candidate": True,
            "vector_graphics_candidate": True,
            "figure_caption_candidate": True,
        },
    )

    tasks = create_enhancement_tasks(page, [block], tmp_path / "enhancements")

    assert tasks == []
    assert block.metadata["image_asset_only"] is True
    assert block.metadata["enhancement_priority"] == "asset_only"
    doc.close()


def test_native_table_extraction_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NATIVE_TABLE_EXTRACTION_MODE", "none")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    add_grid_table(page)

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense")

    assert all(block.kind != "native_table" for block in blocks)
    doc.close()


def test_deferred_native_table_mode_creates_caption_table_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv("NATIVE_TABLE_EXTRACTION_MODE", "defer")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    add_grid_table(page)

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense_table_candidate")

    native_tables = [block for block in blocks if block.kind == "native_table"]
    table_candidates = [block for block in blocks if block.metadata.get("table_candidate")]
    assert native_tables == []
    assert len(table_candidates) == 1
    assert table_candidates[0].metadata["native_table_deferred"] is True
    assert "Table 1. ImageNet validation errors." in table_candidates[0].text
    assert "ResNet-50" in table_candidates[0].text

    tasks = create_enhancement_tasks(page, blocks, tmp_path / "enhancements")
    assert len(tasks) == 1
    assert tasks[0].kind == "table_candidate"
    doc.close()


def test_deferred_complex_mode_skips_single_block_caption_table(monkeypatch):
    monkeypatch.setenv("NATIVE_TABLE_EXTRACTION_MODE", "defer")
    monkeypatch.setenv("DEFERRED_TABLE_QUEUE_MODE", "complex")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(80, 80, 500, 130),
        "method top-1 err. top-5 err. ResNet-50 22.85 6.71 ResNet-101 21.75 6.05",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 150, 545, 190),
        "Table 3. Error rates on ImageNet validation.",
        fontsize=10,
        fontname="helv",
    )

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense_table_candidate")

    assert all(not block.metadata.get("native_table_deferred") for block in blocks)
    assert any("ResNet-50" in block.text for block in blocks)
    doc.close()


def test_deferred_complex_mode_skips_three_block_simple_caption_table(monkeypatch):
    monkeypatch.setenv("NATIVE_TABLE_EXTRACTION_MODE", "defer")
    monkeypatch.setenv("DEFERRED_TABLE_QUEUE_MODE", "complex")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(80, 80, 500, 105),
        "method top-1 err. top-5 err.",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(80, 110, 500, 135),
        "ResNet-50 22.85 6.71",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(80, 140, 500, 165),
        "ResNet-101 21.75 6.05",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 190, 545, 230),
        "Table 3. Error rates on ImageNet validation.",
        fontsize=10,
        fontname="helv",
    )

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense_table_candidate")

    assert all(not block.metadata.get("native_table_deferred") for block in blocks)
    assert any("ResNet-50" in block.text for block in blocks)
    assert any("Table 3. Error rates" in block.text for block in blocks)
    doc.close()


def test_deferred_table_caption_detection_ignores_table_reference_paragraph(monkeypatch):
    monkeypatch.setenv("NATIVE_TABLE_EXTRACTION_MODE", "defer")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(80, 40, 500, 70),
        "region observed expected SR3-body W 12 10.4 SR3-body t 5 4.8",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 80, 545, 180),
        (
            "Table 10 shows the expected and observed numbers of events after the background fit. "
            "This is a normal paragraph, not the table caption itself."
        ),
        fontsize=10,
        fontname="helv",
    )

    _, blocks = extract_fast_page(page, 0, "job-1", "editable_text_dense_table_candidate")

    assert all(not block.metadata.get("native_table_deferred") for block in blocks)
    doc.close()


def test_render_markdown_includes_native_tables_assets_and_completed_vlm():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "text-1",
            "kind": "text",
            "text": "Intro text.",
            "metadata": {},
        },
        {
            "page_index": 0,
            "block_index": 1,
            "block_id": "table-1",
            "kind": "native_table",
            "text": "| Method | Top-1 |\n| --- | --- |\n| ResNet-50 | 22.85 |",
            "metadata": {"native_table": True},
        },
        {
            "page_index": 0,
            "block_index": 2,
            "block_id": "image-1",
            "kind": "image",
            "text": "[vector graphics page=0]",
            "metadata": {"crop_path": "/tmp/page-0000-block-0002.png"},
        },
    ]
    tasks = [
        {
            "block_id": "image-1",
            "status": "completed",
            "result": {"markdown": "Figure 1: residual learning block."},
        }
    ]

    markdown = render_markdown(
        title="Deep Residual Learning for Image Recognition",
        source_pdf="/tmp/resnet.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        asset_names={"image-1": "assets/page-01-figure-01.png"},
    )

    assert "# Deep Residual Learning for Image Recognition" in markdown
    assert "Intro text." in markdown
    assert "| Method | Top-1 |" in markdown
    assert "![page 1 image 1](assets/page-01-figure-01.png)" in markdown
    assert "Figure 1: residual learning block." in markdown


def test_render_markdown_prefers_completed_native_table_enhancement():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "table-1",
            "kind": "native_table",
            "text": "| noisy | table |\n| --- | --- |\n| 3 3× ×3 3 | bad |",
            "metadata": {"native_table": True, "native_table_needs_vlm": True},
        }
    ]
    tasks = [
        {
            "block_id": "table-1",
            "status": "completed",
            "result": {"markdown": "| layer | 50-layer |\n| --- | --- |\n| conv2_x | 3 blocks |"},
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
    )

    assert "| layer | 50-layer |" in markdown
    assert "3 3× ×3 3" not in markdown


def test_render_markdown_prefers_completed_text_table_candidate_enhancement():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "table-text-1",
            "kind": "text",
            "text": "Name Qty Price Total Alpha 12 3.50 42.00 Beta 5 9.00 45.00",
            "metadata": {"table_candidate": True},
        }
    ]
    tasks = [
        {
            "block_id": "table-text-1",
            "status": "completed",
            "result": {
                "markdown": (
                    "| Name | Qty | Price | Total |\n"
                    "| --- | --- | --- | --- |\n"
                    "| Alpha | 12 | 3.50 | 42.00 |\n"
                    "| Beta | 5 | 9.00 | 45.00 |"
                )
            },
        }
    ]

    markdown = render_markdown(
        title="Table Paper",
        source_pdf="/tmp/table.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
    )

    assert "| Name | Qty | Price | Total |" in markdown
    assert "| Beta | 5 | 9.00 | 45.00 |" in markdown
    assert "Name Qty Price Total Alpha" not in markdown


def test_markdown_export_wait_policy_none_uses_native_blocks():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "table-1",
            "kind": "native_table",
            "text": "| noisy | table |\n| --- | --- |\n| native fallback | ok |",
            "metadata": {"native_table": True, "native_table_needs_vlm": True},
        }
    ]
    tasks = [
        {
            "block_id": "table-1",
            "status": "completed",
            "metadata": {"enhancement_priority": "required"},
            "result": {"markdown": "| clean | table |\n| --- | --- |\n| enhanced | ok |"},
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        wait_policy="none",
    )

    assert "native fallback" in markdown
    assert "enhanced" not in markdown


def test_markdown_export_wait_policy_required_skips_optional_pending():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "required-table",
            "kind": "native_table",
            "text": "| noisy | table |\n| --- | --- |\n| native required | ok |",
            "metadata": {"native_table": True, "native_table_needs_vlm": True},
        },
        {
            "page_index": 0,
            "block_index": 1,
            "block_id": "optional-table",
            "kind": "text",
            "text": "Table 2. Native optional fallback ResNet-50 22.85",
            "metadata": {"table_candidate": True},
        },
    ]
    tasks = [
        {
            "block_id": "required-table",
            "status": "completed",
            "metadata": {"enhancement_priority": "required"},
            "result": {"markdown": "| clean | table |\n| --- | --- |\n| required enhanced | ok |"},
        },
        {
            "block_id": "optional-table",
            "status": "completed",
            "metadata": {"enhancement_priority": "optional"},
            "result": {"markdown": "| optional | table |\n| --- | --- |\n| optional enhanced | ok |"},
        },
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        wait_policy="required",
    )

    assert "required enhanced" in markdown
    assert "native required" not in markdown
    assert "Native optional fallback" in markdown
    assert "optional enhanced" not in markdown


def test_markdown_export_wait_policy_all_includes_optional_completed():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "optional-table",
            "kind": "text",
            "text": "Table 2. Native optional fallback ResNet-50 22.85",
            "metadata": {"table_candidate": True},
        }
    ]
    tasks = [
        {
            "block_id": "optional-table",
            "status": "completed",
            "metadata": {"enhancement_priority": "optional"},
            "result": {"markdown": "| optional | table |\n| --- | --- |\n| optional enhanced | ok |"},
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        wait_policy="all",
    )

    assert "optional enhanced" in markdown
    assert "Native optional fallback" not in markdown


def test_markdown_export_rejects_optional_vlm_with_new_caption_ids():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "table-text-1",
            "kind": "text",
            "text": "Table 3. Error rates on ImageNet validation. ResNet-50 22.85 6.71",
            "metadata": {"table_candidate": True},
        }
    ]
    tasks = [
        {
            "block_id": "table-text-1",
            "status": "completed",
            "metadata": {"enhancement_priority": "optional"},
            "result": {
                "markdown": (
                    "Table 3. Error rates on ImageNet validation.\n\n"
                    "| model | top-1 err. | top-5 err. |\n"
                    "| --- | --- | --- |\n"
                    "| ResNet-50 | 22.85 | 6.71 |\n\n"
                    "Figure 34. F"
                )
            },
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        wait_policy="all",
    )

    assert "Figure 34. F" not in markdown
    assert "ResNet-50 22.85 6.71" in markdown


def test_markdown_export_allows_optional_vlm_with_existing_caption_ids():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "table-text-1",
            "kind": "text",
            "text": "Table 3. Error rates on ImageNet validation. ResNet-50 22.85 6.71",
            "metadata": {"table_candidate": True},
        }
    ]
    tasks = [
        {
            "block_id": "table-text-1",
            "status": "completed",
            "metadata": {"enhancement_priority": "optional"},
            "result": {
                "markdown": (
                    "Table 3. Error rates on ImageNet validation.\n\n"
                    "| model | top-1 err. | top-5 err. |\n"
                    "| --- | --- | --- |\n"
                    "| ResNet-50 | 22.85 | 6.71 |"
                )
            },
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        wait_policy="all",
    )

    assert "| ResNet-50 | 22.85 | 6.71 |" in markdown
    assert "ResNet-50 22.85 6.71" not in markdown


def test_render_markdown_converts_vlm_html_tables_to_markdown():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "image-1",
            "kind": "image",
            "text": "[vector graphics page=0]",
            "metadata": {"crop_path": "/tmp/page-0000-block-0000.png"},
        }
    ]
    tasks = [
        {
            "block_id": "image-1",
            "status": "completed",
            "result": {
                "markdown": (
                    "Figure summary.\n"
                    "<table><tr><td>metric</td><td>mAP</td></tr>"
                    "<tr><td>ResNet-101</td><td>48.4</td></tr></table>"
                )
            },
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
        asset_names={"image-1": "assets/page-01-figure-01.png"},
    )

    assert "<table>" not in markdown
    assert "| metric | mAP |" in markdown
    assert "| ResNet-101 | 48.4 |" in markdown


def test_render_markdown_expands_vlm_html_table_spans():
    blocks = [
        {
            "page_index": 0,
            "block_index": 0,
            "block_id": "table-1",
            "kind": "native_table",
            "text": "| noisy | table |",
            "metadata": {"native_table": True, "native_table_needs_vlm": True},
        }
    ]
    tasks = [
        {
            "block_id": "table-1",
            "status": "completed",
            "result": {
                "markdown": (
                    "<table>"
                    "<tr><td>layer name</td><td>output size</td><td>18-layer</td><td>34-layer</td></tr>"
                    "<tr><td rowspan=\"2\">conv2_x</td><td rowspan=\"2\">56x56</td>"
                    "<td colspan=\"2\">max pool</td></tr>"
                    "<tr><td>2 blocks</td><td>3 blocks</td></tr>"
                    "</table>"
                )
            },
        }
    ]

    markdown = render_markdown(
        title="Paper",
        source_pdf="/tmp/paper.pdf",
        job_id="job-1",
        page_count=1,
        blocks=blocks,
        enhancement_tasks=tasks,
    )

    assert "| conv2_x | 56x56 | max pool |  |" in markdown
    assert "|  |  | 2 blocks | 3 blocks |" in markdown
    assert "| 2 blocks | 3 blocks |  |  |" not in markdown


def test_hash_embedding_provider_is_deterministic_and_dimensioned():
    provider = HashEmbeddingProvider(dim=12)
    first = provider.embed(["same text"])[0]
    second = provider.embed(["same text"])[0]
    different = provider.embed(["different text"])[0]

    assert len(first) == 12
    assert first == second
    assert first != different


def test_transformers_embedding_provider_pools_and_normalizes_fake_model(tmp_path):
    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, texts, padding, truncation, max_length, return_tensors):
            assert padding is True
            assert truncation is True
            assert max_length == 8
            assert return_tensors == "pt"
            return FakeBatch(
                {
                    "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                }
            )

    class FakeOutput:
        def __init__(self):
            self.last_hidden_state = torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
                    [[2.0, 0.0], [100.0, 100.0], [100.0, 100.0]],
                ]
            )

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **batch):
            return FakeOutput()

    provider = TransformersEmbeddingProvider(
        model_path=str(tmp_path),
        tokenizer=FakeTokenizer(),
        model_obj=FakeModel(),
        device="cpu",
        max_length=8,
    )

    vectors = provider.embed(["alpha", "beta"])

    assert provider.provider == "transformers"
    assert provider.model == str(tmp_path)
    assert provider.dim == 2
    assert len(vectors) == 2
    assert vectors[0] == [0.707107, 0.707107]
    assert vectors[1] == [1.0, 0.0]


def test_build_embedding_provider_supports_transformers(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(tmp_path))
    provider = build_embedding_provider("transformers", dim=0)

    assert isinstance(provider, TransformersEmbeddingProvider)
    assert provider.model == str(tmp_path)


def test_build_embedding_provider_reuses_transformers_instance(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(tmp_path))

    first = build_embedding_provider("transformers", dim=0)
    second = build_embedding_provider("transformers", dim=0)

    assert first is second


def test_block_record_generates_stable_ids_from_content():
    block = BlockRecord(
        job_id="job-1",
        page_index=3,
        block_index=2,
        source="fast_pymupdf",
        kind="text",
        text="hello world",
        bbox=[1.0, 2.0, 3.0, 4.0],
        metadata={"route": "editable_text_dense"},
    )

    assert block.block_id.startswith("job-1-p0003-b0002-")
    assert block.embedding_text == "hello world"
