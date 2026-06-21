from __future__ import annotations

import importlib
import sys
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient


def make_editable_pdf(path: Path, pages: int = 2) -> Path:
    doc = fitz.open()
    text = (
        "MinerU VLM lab editable fast path. "
        "This PDF has a native text layer and should not call VLM. "
    ) * 18
    for page_index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(50, 60, 545, 780),
            f"Page {page_index + 1}\n{text}",
            fontsize=12,
            fontname="helv",
        )
    doc.save(path)
    doc.close()
    return path


def make_table_candidate_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(50, 60, 545, 260),
        (
            "Section text before the table.\n"
            "Name          Qty          Price          Total\n"
            "Alpha         12           3.50           42.00\n"
            "Beta          5            9.00           45.00\n"
            "Gamma         8            2.25           18.00\n"
            "Section text after the table.\n"
        ),
        fontsize=12,
        fontname="cour",
    )
    doc.save(path)
    doc.close()
    return path


def make_editable_pdf_with_image(path: Path) -> Path:
    image_path = path.with_suffix(".png")
    image_doc = fitz.open()
    image_page = image_doc.new_page(width=120, height=80)
    image_page.draw_rect(fitz.Rect(8, 8, 112, 72), color=(0.1, 0.35, 0.8), fill=(0.75, 0.85, 1.0))
    image_page.insert_textbox(fitz.Rect(18, 24, 102, 60), "FIG", fontsize=18, fontname="helv")
    pix = image_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(str(image_path))
    image_doc.close()

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    text = (
        "Editable paper body text with a native text layer. "
        "The image on this page should stay visible as a separate block. "
    ) * 14
    page.insert_textbox(
        fitz.Rect(50, 220, 545, 780),
        text,
        fontsize=12,
        fontname="helv",
    )
    page.insert_image(fitz.Rect(360, 70, 520, 180), filename=str(image_path))
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(path: Path, pages: int = 1) -> Path:
    source = fitz.open()
    rendered_pages = []
    for page_index in range(pages):
        page = source.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(55, 70, 540, 760),
            (
                f"Scanned page {page_index + 1}. This text is rasterized into the page image. "
                "The parser should queue direct VLM OCR instead of calling the old router. "
            )
            * 12,
            fontsize=12,
            fontname="helv",
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        rendered_pages.append((page.rect, pix.tobytes("png")))

    scanned = fitz.open()
    for rect, image_bytes in rendered_pages:
        page = scanned.new_page(width=rect.width, height=rect.height)
        page.insert_image(rect, stream=image_bytes)
    scanned.save(path)
    scanned.close()
    source.close()
    return path


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MINERU_VLM_LAB_ROOT", str(tmp_path))
    monkeypatch.setenv("MINERU_VLM_LAB_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MINERU_VLM_LAB_DB_PATH", str(tmp_path / "mineru-vlm-lab.sqlite3"))
    monkeypatch.setenv("MINERU_VLM_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIM", "16")

    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            sys.modules.pop(module_name, None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as client:
        yield client
