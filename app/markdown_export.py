from __future__ import annotations

from collections import defaultdict
import html
import re
from typing import Any


WAIT_POLICIES = {"none", "required", "all"}
INLINE_TABLE_CAPTION_PATTERN = re.compile(r"^\s*Table\s+\d+\s*[\.:]", re.IGNORECASE)
INLINE_TABLE_HEADER_BIGRAMS = {
    ("group", "name"),
    ("warping", "error"),
    ("rand", "error"),
    ("pixel", "error"),
    ("top-1", "err."),
    ("top-5", "err."),
}
INLINE_TABLE_NUMBER = r"[-+]?(?:\d+\.\d+|\.\d+)(?:%)?|\d+%|-"


def render_markdown(
    *,
    title: str,
    source_pdf: str,
    job_id: str,
    page_count: int,
    blocks: list[dict[str, Any]],
    enhancement_tasks: list[dict[str, Any]],
    asset_names: dict[str, str] | None = None,
    wait_policy: str = "all",
) -> str:
    if wait_policy not in WAIT_POLICIES:
        raise ValueError(f"unsupported wait policy: {wait_policy}")
    asset_names = asset_names or {}
    blocks_by_id = {str(block.get("block_id") or ""): block for block in blocks}
    completed_tasks: dict[str, dict[str, Any]] = {}
    for task in enhancement_tasks:
        block_id = str(task.get("block_id") or "")
        if (
            task.get("status") == "completed"
            and isinstance(task.get("result"), dict)
            and _uses_task_for_wait_policy(task, wait_policy)
            and _passes_task_quality_gate(task, blocks_by_id.get(block_id), task["result"])
        ):
            completed_tasks[block_id] = task["result"]

    pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        pages[int(block.get("page_index") or 0)].append(block)

    lines = [
        "---",
        f'title: "{_escape_yaml(title)}"',
        f'source_pdf: "{_escape_yaml(source_pdf)}"',
        f'job_id: "{_escape_yaml(job_id)}"',
        f"page_count: {page_count}",
        f"block_count: {len(blocks)}",
        f"enhancement_task_count: {len(enhancement_tasks)}",
        "---",
        "",
        f"# {title}",
        "",
    ]

    for page_index in range(page_count):
        page_blocks = sorted(pages.get(page_index, []), key=lambda item: int(item.get("block_index") or 0))
        lines.append(f"## Page {page_index + 1}")
        lines.append("")
        for image_number, block in enumerate(
            [item for item in page_blocks if str(item.get("kind")) == "image"],
            start=1,
        ):
            block_id = str(block.get("block_id") or "")
            asset_name = asset_names.get(block_id)
            if asset_name:
                block["_markdown_image_number"] = image_number
                block["_markdown_asset_name"] = asset_name

        for block in page_blocks:
            rendered = _render_block(block, completed_tasks.get(str(block.get("block_id"))))
            if rendered:
                lines.append(rendered)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _uses_task_for_wait_policy(task: dict[str, Any], wait_policy: str) -> bool:
    if wait_policy == "none":
        return False
    priority = _enhancement_priority(task)
    if wait_policy == "required":
        return priority == "required"
    return priority != "asset_only"


def _passes_task_quality_gate(task: dict[str, Any], block: dict[str, Any] | None, result: dict[str, Any]) -> bool:
    if _enhancement_priority(task) != "optional":
        return True
    if not _is_table_enhancement(task, block):
        return True
    if block is None:
        return True
    source_caption_ids = _caption_ids(str(block.get("text") or ""))
    result_caption_ids = _caption_ids(str(result.get("markdown") or ""))
    return result_caption_ids.issubset(source_caption_ids)


def _is_table_enhancement(task: dict[str, Any], block: dict[str, Any] | None) -> bool:
    if str(task.get("kind") or "") == "table_candidate":
        return True
    if not block:
        return False
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    return bool(metadata.get("table_candidate") or metadata.get("native_table_needs_vlm"))


def _caption_ids(value: str) -> set[tuple[str, int]]:
    ids: set[tuple[str, int]] = set()
    for match in re.finditer(r"\b(Table|Fig(?:ure)?\.?)\s*(\d+)\s*[.:]", value, flags=re.IGNORECASE):
        label = "figure" if match.group(1).lower().startswith("fig") else "table"
        ids.add((label, int(match.group(2))))
    return ids


def _enhancement_priority(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    value = str(metadata.get("enhancement_priority") or "").strip().lower()
    if value in {"required", "optional", "asset_only"}:
        return value
    if metadata.get("image_asset_only"):
        return "asset_only"
    if metadata.get("scan_ocr") or metadata.get("page_vlm") or metadata.get("force_vlm"):
        return "required"
    if metadata.get("native_table_needs_vlm") and metadata.get("native_table"):
        return "required"
    return "optional"


def _render_block(block: dict[str, Any], completed_task: dict[str, Any] | None) -> str:
    kind = str(block.get("kind") or "text")
    if kind == "image":
        parts: list[str] = []
        asset_name = block.get("_markdown_asset_name")
        if asset_name:
            page_number = int(block.get("page_index") or 0) + 1
            image_number = int(block.get("_markdown_image_number") or 1)
            parts.append(f"![page {page_number} image {image_number}]({asset_name})")
        task_markdown = _clean_task_markdown(completed_task)
        if task_markdown:
            parts.append(task_markdown)
        return "\n\n".join(parts)
    text = str(block.get("text") or "").strip()
    if kind == "native_table":
        task_markdown = _clean_task_markdown(completed_task)
        if task_markdown:
            return task_markdown
        return text
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    if metadata.get("table_candidate") or metadata.get("native_table_needs_vlm"):
        task_markdown = _clean_task_markdown(completed_task)
        if task_markdown:
            return task_markdown
    return _rewrite_inline_text_tables(text)


def _clean_task_markdown(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    markdown = str(result.get("markdown") or "").strip()
    return _convert_html_tables(markdown)


def _escape_yaml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _convert_html_tables(value: str) -> str:
    return re.sub(r"<table\b[^>]*>.*?</table>", _html_table_to_markdown, value, flags=re.IGNORECASE | re.DOTALL)


def _rewrite_inline_text_tables(value: str) -> str:
    lines = value.splitlines()
    if not any(INLINE_TABLE_CAPTION_PATTERN.match(line) for line in lines):
        return value
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        table_markdown, consumed = _inline_text_table_to_markdown(lines, index)
        if table_markdown:
            rendered.append(table_markdown)
            index += consumed
            continue
        rendered.append(lines[index])
        index += 1
    return "\n".join(rendered)


def _inline_text_table_to_markdown(lines: list[str], caption_index: int) -> tuple[str, int]:
    caption = lines[caption_index].strip()
    if not INLINE_TABLE_CAPTION_PATTERN.match(caption):
        return "", 0
    header_index = caption_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    if header_index >= len(lines):
        return "", 0
    header = _split_inline_table_header(lines[header_index].strip())
    if len(header) < 2 or len(header) > 8:
        return "", 0

    rows: list[list[str]] = []
    data_index = header_index + 1
    while data_index < len(lines):
        line = lines[data_index].strip()
        if not line or INLINE_TABLE_CAPTION_PATTERN.match(line):
            break
        parsed = _parse_inline_table_data_line(line, header)
        if not parsed:
            break
        rows.extend(parsed)
        data_index += 1
    if not rows:
        return "", 0

    markdown_rows = [
        "| " + " | ".join(_escape_inline_table_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(_escape_inline_table_cell(cell) for cell in row) + " |" for row in rows
    )
    consumed = data_index - caption_index
    return caption + "\n" + "\n".join(markdown_rows), consumed


def _split_inline_table_header(value: str) -> list[str]:
    tokens = value.split()
    columns: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index].lower(), tokens[index + 1].lower()) in INLINE_TABLE_HEADER_BIGRAMS:
            columns.append(f"{tokens[index]} {tokens[index + 1]}")
            index += 2
            continue
        columns.append(tokens[index])
        index += 1
    return columns


def _parse_inline_table_data_line(value: str, header: list[str]) -> list[list[str]]:
    numeric_columns = max(1, min(3, len(header) - 1))
    descriptor_columns = len(header) - numeric_columns
    number_group = rf"({INLINE_TABLE_NUMBER})"
    number_pattern = r"\s+".join(number_group for _ in range(numeric_columns))
    pattern = re.compile(rf"(.+?)\s+{number_pattern}(?=\s+\S|$)")
    rows: list[list[str]] = []
    for match in pattern.finditer(value):
        label = " ".join(match.group(1).split())
        numbers = list(match.groups()[1:])
        descriptors = _split_inline_table_descriptor(label, descriptor_columns)
        if len(descriptors) + len(numbers) != len(header):
            continue
        rows.append([*descriptors, *numbers])
    return rows


def _split_inline_table_descriptor(value: str, width: int) -> list[str]:
    if width <= 1:
        return [value]
    rank_match = re.match(r"^(?:\.\.\.\s*)?(\d+)\.\s+(.+)$", value)
    if rank_match:
        return [rank_match.group(1), rank_match.group(2), *("" for _ in range(max(0, width - 2)))]
    return ["", value, *("" for _ in range(max(0, width - 2)))]


def _escape_inline_table_cell(value: str) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _html_table_to_markdown(match: re.Match[str]) -> str:
    table = match.group(0)
    rows: list[list[str]] = []
    rowspans: dict[int, int] = {}

    def fill_rowspans(cells: list[str], col_index: int) -> int:
        while rowspans.get(col_index, 0) > 0:
            cells.append("")
            rowspans[col_index] -= 1
            if rowspans[col_index] <= 0:
                del rowspans[col_index]
            col_index += 1
        return col_index

    for row_match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", table, flags=re.IGNORECASE | re.DOTALL):
        cells: list[str] = []
        col_index = 0
        for cell_match in re.finditer(
            r"<t[dh]\b([^>]*)>(.*?)</t[dh]>",
            row_match.group(1),
            flags=re.IGNORECASE | re.DOTALL,
        ):
            col_index = fill_rowspans(cells, col_index)
            attrs = cell_match.group(1)
            colspan = _html_span(attrs, "colspan")
            rowspan = _html_span(attrs, "rowspan")
            cells.append(_clean_html_cell(cell_match.group(2)))
            for _ in range(1, colspan):
                cells.append("")
            if rowspan > 1:
                for span_col in range(col_index, col_index + colspan):
                    rowspans[span_col] = max(rowspans.get(span_col, 0), rowspan - 1)
            col_index += colspan
        col_index = fill_rowspans(cells, col_index)
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n" + "\n".join(lines) + "\n"


def _html_span(attrs: str, name: str) -> int:
    match = re.search(rf'{name}\s*=\s*["\']?(\d+)', attrs, flags=re.IGNORECASE)
    if not match:
        return 1
    return max(1, int(match.group(1)))


def _clean_html_cell(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return " ".join(value.split()).replace("|", "\\|")
