from __future__ import annotations

import os
import re
from typing import Any

import fitz

from app.models import BlockRecord, PageSignal


TABLE_ROW_PATTERN = re.compile(r"\S+(?:[^\S\r\n]{2,}\S+){3,}")
TABLE_CAPTION_PATTERN = re.compile(r"(^|\n)\s*Table\s+\d+[\.:]?", re.IGNORECASE)
TABLE_CAPTION_START_PATTERN = re.compile(r"^\s*Table\s+\d+\s*[\.:]", re.IGNORECASE)
FIGURE_CAPTION_PATTERN = re.compile(r"(^|\n)\s*(Figure|Fig\.)\s+\d+[\.:]?", re.IGNORECASE)
FAST_TEXT_CHUNK_CHARS = int(os.getenv("FAST_TEXT_CHUNK_CHARS", "1800"))
VECTOR_GRAPHIC_DRAWING_THRESHOLD = int(os.getenv("VECTOR_GRAPHIC_DRAWING_THRESHOLD", "40"))
FIGURE_VECTOR_GRAPHIC_DRAWING_THRESHOLD = int(os.getenv("FIGURE_VECTOR_GRAPHIC_DRAWING_THRESHOLD", "3"))
DEFER_NATIVE_TABLE_MODES = {"defer", "deferred", "async", "candidate", "candidates", "vlm"}


def looks_like_table_text(value: str) -> bool:
    rows = [line for line in value.splitlines() if line.strip()]
    if any(line.count("\t") >= 2 for line in rows):
        return True
    table_rows = [line for line in rows if TABLE_ROW_PATTERN.search(line)]
    return len(table_rows) >= 2


def native_table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = [_escape_markdown_cell(cell) for cell in normalized[0]]
    body_rows: list[list[str]] = []
    for row in normalized[1:]:
        body_rows.extend(_expand_multiline_table_row(row))
    body = [[_escape_markdown_cell(cell) for cell in row] for row in body_rows]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def native_table_needs_vlm(markdown: str) -> bool:
    if any(ord(char) < 32 and char not in "\n\t" for char in markdown):
        return True
    if any("\uf000" <= char <= "\uf8ff" for char in markdown):
        return True
    if re.search(r"\d+\s+\d+×\s*×\d+", markdown):
        return True
    if re.search(r"×\s*×|,,", markdown):
        return True
    return False


def extract_native_tables(page: fitz.Page, page_text: str = "", route_reason: str = "") -> list[dict[str, Any]]:
    if not should_extract_native_tables(page_text, route_reason):
        return []
    if not hasattr(page, "find_tables"):
        return []
    try:
        table_finder = page.find_tables()
    except Exception:
        return []
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(getattr(table_finder, "tables", []) or []):
        try:
            rows = _normalize_table_rows(table.extract())
        except Exception:
            continue
        if not _is_usable_native_table(rows):
            continue
        bbox = [round(float(value), 2) for value in table.bbox]
        markdown = native_table_to_markdown(rows)
        if not markdown:
            continue
        tables.append(
            {
                "bbox": bbox,
                "text": markdown,
                "kind": "native_table",
                "rows": rows,
                "table_index": table_index,
                "native_table_needs_vlm": native_table_needs_vlm(markdown),
            }
        )
    return tables


def should_extract_native_tables(page_text: str, route_reason: str = "") -> bool:
    mode = os.getenv("NATIVE_TABLE_EXTRACTION_MODE", "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "none", "disabled"}:
        return False
    if mode in DEFER_NATIVE_TABLE_MODES:
        return False
    if mode in {"1", "true", "yes", "on", "all", "full"}:
        return True
    if "table_candidate" in route_reason:
        return True
    if TABLE_CAPTION_PATTERN.search(page_text):
        return True
    return looks_like_table_text(page_text)


def has_figure_caption(page_text: str) -> bool:
    return bool(FIGURE_CAPTION_PATTERN.search(page_text))


def image_area_ratio(page: fitz.Page) -> float:
    page_area = max(1.0, page.rect.width * page.rect.height)
    total = 0.0
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            total += max(0.0, rect.width * rect.height)
    return min(1.0, total / page_area)


def collect_page_signal(page: fitz.Page, page_index: int) -> PageSignal:
    text = page.get_text("text") or ""
    blocks = page.get_text("blocks") or []
    image_count = len(page.get_images(full=True))
    ratio = image_area_ratio(page)
    # Plain references such as "Table 2" are common in editable papers and are
    # not enough to spend a VLM page. Only strong text-layer table signals route
    # the whole page away from the fast path in this lab implementation.
    table_hint = looks_like_table_text(text)
    return PageSignal(
        page_index=page_index,
        text_chars=len(text.strip()),
        block_count=len(blocks),
        image_count=image_count,
        image_area_ratio=ratio,
        has_table_hint=table_hint,
    )


def classify_page(signal: PageSignal) -> tuple[str, str]:
    if signal.text_chars >= 500 and signal.image_area_ratio < 0.20:
        if signal.has_table_hint:
            return "fast_pymupdf", "editable_text_dense_table_candidate"
        return "fast_pymupdf", "editable_text_dense"
    if signal.text_chars >= 220 and signal.image_area_ratio < 0.35 and signal.image_count <= 2:
        if signal.has_table_hint:
            return "fast_pymupdf", "editable_text_moderate_table_candidate"
        return "fast_pymupdf", "editable_text_moderate"
    if signal.has_table_hint:
        return "vlm", "table_hint"
    if signal.image_area_ratio >= 0.35 or signal.text_chars < 80:
        return "vlm", "scan_or_image_heavy"
    return "vlm", "mixed_or_low_confidence"


def extract_fast_page(
    page: fitz.Page,
    page_index: int,
    job_id: str,
    route_reason: str = "",
) -> tuple[dict[str, Any], list[BlockRecord]]:
    blocks: list[dict[str, Any]] = []
    block_records: list[BlockRecord] = []
    text_items: list[dict[str, Any]] = []
    page_text = page.get_text("text") or ""
    native_tables = extract_native_tables(page, page_text, route_reason)
    for block in page.get_text("blocks") or []:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        raw_text = str(text)
        clean = " ".join(raw_text.split())
        if not clean:
            continue
        bbox = [round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2)]
        if any(_bbox_overlap_ratio(bbox, table["bbox"]) >= 0.75 for table in native_tables):
            continue
        table_candidate = looks_like_table_text(raw_text)
        text_items.append({"bbox": bbox, "text": clean, "raw_text": raw_text, "table_candidate": table_candidate})

    deferred_tables: list[dict[str, Any]] = []
    if not native_tables and should_defer_native_table_candidates():
        deferred_tables = deferred_table_candidates(page, text_items)
        if deferred_tables:
            text_items = [
                item
                for item in text_items
                if not any(_bbox_overlap_ratio(item["bbox"], table["bbox"]) >= 0.75 for table in deferred_tables)
            ]

    preserve_text_blocks = (
        bool(native_tables)
        or bool(deferred_tables)
        or "table_candidate" in route_reason
        or any(item["table_candidate"] for item in text_items)
    )
    if preserve_text_blocks:
        text_chunks = text_items
    else:
        text_chunks = merge_text_items(text_items, FAST_TEXT_CHUNK_CHARS)

    content_items = [dict(item, kind="text") for item in text_chunks]
    content_items.extend(deferred_tables)
    content_items.extend(native_tables)
    content_items.sort(key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))

    for item in content_items:
        block_index = len(block_records)
        bbox = item["bbox"]
        clean = item["text"]
        kind = str(item.get("kind") or "text")
        blocks.append({"bbox": bbox, "text": clean, "kind": kind})
        metadata = {"route_reason": route_reason}
        if kind == "native_table":
            metadata.update(
                {
                    "native_table": True,
                    "table_index": item.get("table_index"),
                    "table_rows": item.get("rows") or [],
                    "table_candidate": False,
                    "native_table_needs_vlm": bool(item.get("native_table_needs_vlm")),
                }
            )
        else:
            metadata["table_candidate"] = bool(item.get("table_candidate"))
            if item.get("native_table_deferred"):
                metadata["native_table_deferred"] = True
                metadata["table_caption"] = item.get("table_caption")
        block_records.append(
            BlockRecord(
                job_id=job_id,
                page_index=page_index,
                block_index=block_index,
                source="fast_pymupdf",
                kind=kind,
                text=clean,
                bbox=bbox,
                metadata=metadata,
            )
        )
    for image in page.get_images(full=True):
        xref = image[0]
        width = int(image[2]) if len(image) > 2 else None
        height = int(image[3]) if len(image) > 3 else None
        for rect in page.get_image_rects(xref):
            bbox = [round(float(rect.x0), 2), round(float(rect.y0), 2), round(float(rect.x1), 2), round(float(rect.y1), 2)]
            block_index = len(block_records)
            text = f"[image page={page_index} xref={xref}]"
            blocks.append({"bbox": bbox, "text": text, "kind": "image"})
            block_records.append(
                BlockRecord(
                    job_id=job_id,
                    page_index=page_index,
                    block_index=block_index,
                    source="fast_pymupdf",
                    kind="image",
                    text=text,
                    bbox=bbox,
                    metadata={
                        "route_reason": route_reason,
                        "image_candidate": True,
                        "xref": xref,
                        "width": width,
                        "height": height,
                    },
                )
            )
    figure_caption_candidate = has_figure_caption(page_text)
    vector_threshold = VECTOR_GRAPHIC_DRAWING_THRESHOLD
    if figure_caption_candidate:
        vector_threshold = min(vector_threshold, FIGURE_VECTOR_GRAPHIC_DRAWING_THRESHOLD)
    vector_bbox = vector_graphics_bbox(page, vector_threshold)
    if vector_bbox is not None:
        block_index = len(block_records)
        text = f"[vector graphics page={page_index}]"
        blocks.append({"bbox": vector_bbox, "text": text, "kind": "image"})
        block_records.append(
            BlockRecord(
                job_id=job_id,
                page_index=page_index,
                block_index=block_index,
                source="fast_pymupdf",
                kind="image",
                text=text,
                bbox=vector_bbox,
                metadata={
                    "route_reason": route_reason,
                    "image_candidate": True,
                    "vector_graphics_candidate": True,
                    "figure_caption_candidate": figure_caption_candidate,
                    "drawing_threshold": vector_threshold,
                },
            )
        )
    return {"page_index": page_index, "text": page_text, "blocks": blocks}, block_records


def should_defer_native_table_candidates() -> bool:
    mode = os.getenv("NATIVE_TABLE_EXTRACTION_MODE", "auto").strip().lower()
    return mode in DEFER_NATIVE_TABLE_MODES


def deferred_table_candidates(page: fitz.Page, text_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    captions = [item for item in text_items if _is_strict_table_caption(str(item.get("raw_text") or item.get("text") or ""))]
    if not captions:
        return []
    candidates: list[dict[str, Any]] = []
    for caption in captions:
        cluster = _collect_deferred_table_cluster(page, text_items, caption, "above")
        if not cluster:
            cluster = _collect_deferred_table_cluster(page, text_items, caption, "below")
        if not cluster:
            continue
        if not _should_queue_deferred_table_candidate(cluster, caption):
            continue
        items = sorted([*cluster, caption], key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))
        bbox = _union_bboxes([item["bbox"] for item in items])
        if bbox is None:
            continue
        raw_text = "\n".join(str(item.get("raw_text") or item.get("text") or "").strip() for item in items)
        clean = "\n".join(line for line in (_clean_text_line(line) for line in raw_text.splitlines()) if line)
        if not clean:
            continue
        candidates.append(
            {
                "bbox": [round(value, 2) for value in bbox],
                "text": clean,
                "raw_text": raw_text,
                "kind": "text",
                "table_candidate": True,
                "native_table_deferred": True,
                "table_caption": _clean_text_line(str(caption.get("raw_text") or caption.get("text") or "")),
            }
        )
    return _dedupe_deferred_tables(candidates)


def _should_queue_deferred_table_candidate(cluster: list[dict[str, Any]], caption: dict[str, Any]) -> bool:
    mode = os.getenv("DEFERRED_TABLE_QUEUE_MODE", "all").strip().lower()
    if mode in {"0", "false", "no", "off", "none", "disabled", "text"}:
        return False
    if mode in {"1", "true", "yes", "on", "all"}:
        return True
    raw_text = "\n".join(str(item.get("raw_text") or item.get("text") or "") for item in [*cluster, caption])
    if native_table_needs_vlm(raw_text):
        return True
    min_blocks = _int_env("DEFERRED_TABLE_MIN_COMPLEX_BLOCKS", 5)
    if len(cluster) >= min_blocks:
        return True
    min_chars = _int_env("DEFERRED_TABLE_MIN_COMPLEX_CHARS", 700)
    if len(" ".join(raw_text.split())) >= min_chars and looks_like_table_text(raw_text):
        return True
    return False


def _collect_deferred_table_cluster(
    page: fitz.Page,
    text_items: list[dict[str, Any]],
    caption: dict[str, Any],
    direction: str,
) -> list[dict[str, Any]]:
    caption_bbox = [float(value) for value in caption["bbox"]]
    if direction == "above":
        max_gap = _float_env("DEFERRED_TABLE_MAX_ABOVE_CLUSTER_GAP", 18.0)
    else:
        max_gap = _float_env("DEFERRED_TABLE_MAX_BELOW_CLUSTER_GAP", 24.0)
    first_gap = _float_env("DEFERRED_TABLE_MAX_FIRST_GAP", 64.0)
    window = _float_env("DEFERRED_TABLE_LOOKBACK_POINTS", 220.0)
    x0, x1 = _deferred_table_x_bounds(page, caption_bbox)
    pool: list[dict[str, Any]] = []
    for item in text_items:
        if item is caption:
            continue
        bbox = [float(value) for value in item["bbox"]]
        if not _horizontally_related(bbox, x0, x1):
            continue
        if direction == "above":
            gap = caption_bbox[1] - bbox[3]
            if 0 <= gap <= window:
                pool.append(item)
        else:
            gap = bbox[1] - caption_bbox[3]
            if 0 <= gap <= window:
                pool.append(item)
    if direction == "above":
        pool.sort(key=lambda item: (float(item["bbox"][3]), float(item["bbox"][0])), reverse=True)
    else:
        pool.sort(key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))

    cluster: list[dict[str, Any]] = []
    boundary = caption_bbox[1] if direction == "above" else caption_bbox[3]
    for item in pool:
        bbox = [float(value) for value in item["bbox"]]
        gap = boundary - bbox[3] if direction == "above" else bbox[1] - boundary
        if not cluster and gap > first_gap:
            break
        if cluster and gap > max_gap:
            break
        cluster.append(item)
        boundary = bbox[1] if direction == "above" else bbox[3]
    return sorted(cluster, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0])))


def _is_strict_table_caption(value: str) -> bool:
    return bool(TABLE_CAPTION_START_PATTERN.search(value.strip()))


def _deferred_table_x_bounds(page: fitz.Page, caption_bbox: list[float]) -> tuple[float, float]:
    page_width = max(1.0, float(page.rect.width))
    caption_width = max(1.0, caption_bbox[2] - caption_bbox[0])
    if caption_width / page_width >= _float_env("DEFERRED_TABLE_FULL_WIDTH_CAPTION_RATIO", 0.55):
        return float(page.rect.x0), float(page.rect.x1)
    pad = _float_env("DEFERRED_TABLE_X_PADDING", 24.0)
    return max(float(page.rect.x0), caption_bbox[0] - pad), min(float(page.rect.x1), caption_bbox[2] + pad)


def _horizontally_related(bbox: list[float], x0: float, x1: float) -> bool:
    overlap = max(0.0, min(float(bbox[2]), x1) - max(float(bbox[0]), x0))
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    return overlap / width >= _float_env("DEFERRED_TABLE_MIN_X_OVERLAP", 0.25)


def _dedupe_deferred_tables(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]))):
        if any(_bbox_overlap_ratio(candidate["bbox"], existing["bbox"]) >= 0.75 for existing in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _union_bboxes(bboxes: list[list[float]]) -> list[float] | None:
    if not bboxes:
        return None
    return [
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    ]


def _clean_text_line(value: str) -> str:
    return " ".join(str(value).split())


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def vector_graphics_bbox(page: fitz.Page, threshold: int) -> list[float] | None:
    if threshold <= 0:
        return None
    try:
        drawings = page.get_drawings()
    except Exception:
        return None
    rects: list[fitz.Rect] = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        rect = fitz.Rect(rect)
        if rect.is_empty or rect.width < 4 or rect.height < 4:
            continue
        rects.append(rect)
    if len(rects) < threshold:
        return None
    merged = fitz.Rect(rects[0])
    for rect in rects[1:]:
        merged |= rect
    page_rect = page.rect
    merged.x0 = max(page_rect.x0, merged.x0)
    merged.y0 = max(page_rect.y0, merged.y0)
    merged.x1 = min(page_rect.x1, merged.x1)
    merged.y1 = min(page_rect.y1, merged.y1)
    if merged.is_empty:
        return None
    return [round(float(merged.x0), 2), round(float(merged.y0), 2), round(float(merged.x1), 2), round(float(merged.y1), 2)]


def merge_text_items(items: list[dict[str, Any]], target_chars: int) -> list[dict[str, Any]]:
    if not items:
        return []
    target = max(200, target_chars)
    chunks: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_bbox: list[float] | None = None

    def flush() -> None:
        nonlocal current_texts, current_bbox
        if not current_texts or current_bbox is None:
            return
        chunks.append(
            {
                "bbox": [round(value, 2) for value in current_bbox],
                "text": "\n".join(current_texts),
                "table_candidate": False,
            }
        )
        current_texts = []
        current_bbox = None

    for item in items:
        text = str(item["text"])
        if current_texts and sum(len(value) for value in current_texts) + len(text) > target:
            flush()
        current_texts.append(text)
        bbox = [float(value) for value in item["bbox"]]
        if current_bbox is None:
            current_bbox = bbox
        else:
            current_bbox = [
                min(current_bbox[0], bbox[0]),
                min(current_bbox[1], bbox[1]),
                max(current_bbox[2], bbox[2]),
                max(current_bbox[3], bbox[3]),
            ]
    flush()
    return chunks


def _normalize_table_rows(raw_rows: list[list[Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_row in raw_rows or []:
        row = [_clean_cell(cell) for cell in raw_row]
        if any(row):
            rows.append(row)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    keep_columns = [index for index in range(width) if any(row[index] for row in padded)]
    return [[row[index] for index in keep_columns] for row in padded]


def _is_usable_native_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    cell_count = len(rows) * width
    non_empty = sum(1 for row in rows for cell in row if cell.strip())
    if non_empty < 4 or non_empty / max(1, cell_count) < 0.45:
        return False
    first_row_non_empty = sum(1 for cell in rows[0] if cell.strip())
    if first_row_non_empty < 1:
        return False
    joined = " ".join(cell.lower() for row in rows for cell in row)
    if "iter." in joined and "error" in joined and "%" in joined:
        return False
    return True


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    cleaned = str(value).replace("\x14", "").replace("\uf8ee", "").replace("\uf8ef", "").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in cleaned.split("\n")]
    return "\n".join(line for line in lines if line)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _expand_multiline_table_row(row: list[str]) -> list[list[str]]:
    split_cells = [_split_cell_lines(cell) for cell in row]
    multiline_lengths = [len(lines) for lines in split_cells if len(lines) > 1]
    if not multiline_lengths:
        return [row]
    target = max(multiline_lengths)
    if any(length not in {1, target} for length in (len(lines) for lines in split_cells)):
        return [row]
    expanded: list[list[str]] = []
    for line_index in range(target):
        expanded.append([lines[line_index] if len(lines) > 1 else lines[0] for lines in split_cells])
    return expanded


def _split_cell_lines(value: str) -> list[str]:
    lines = [" ".join(line.split()) for line in str(value).splitlines()]
    lines = [line for line in lines if line]
    return lines or [""]


def _bbox_overlap_ratio(inner_bbox: list[float], outer_bbox: list[float]) -> float:
    ix0 = max(float(inner_bbox[0]), float(outer_bbox[0]))
    iy0 = max(float(inner_bbox[1]), float(outer_bbox[1]))
    ix1 = min(float(inner_bbox[2]), float(outer_bbox[2]))
    iy1 = min(float(inner_bbox[3]), float(outer_bbox[3]))
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (float(inner_bbox[2]) - float(inner_bbox[0])) * (float(inner_bbox[3]) - float(inner_bbox[1])))
    return intersection / area
