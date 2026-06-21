from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.markdown_export import render_markdown
from app.settings import load_settings
from app.storage import SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a parsed job to Markdown.")
    parser.add_argument("job_id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--db-path", default="")
    parser.add_argument("--wait-enhancements", choices=["none", "required", "all"], default="all")
    args = parser.parse_args()

    settings = load_settings()
    db_path = Path(args.db_path) if args.db_path else settings.db_path
    store = SQLiteStore(db_path)
    job = store.get_job(args.job_id)
    if job is None:
        raise SystemExit(f"job not found: {args.job_id}")

    blocks = store.list_blocks(args.job_id)
    tasks = store.list_enhancement_tasks(args.job_id)
    output_dir = Path(args.output_dir)
    asset_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    asset_names = copy_block_assets(blocks, asset_dir)
    title = args.title or Path(job.file_name or args.job_id).stem
    markdown = render_markdown(
        title=title,
        source_pdf=job.file_name or "",
        job_id=args.job_id,
        page_count=job.page_count,
        blocks=blocks,
        enhancement_tasks=tasks,
        asset_names=asset_names,
        wait_policy=args.wait_enhancements,
    )
    output_path = output_dir / f"{Path(job.file_name or 'document').stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(output_path)


def copy_block_assets(blocks: list[dict[str, Any]], asset_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    image_index = 0
    for block in blocks:
        if block.get("kind") != "image":
            continue
        crop_path = Path(str((block.get("metadata") or {}).get("crop_path") or ""))
        if not crop_path.exists():
            continue
        image_index += 1
        page_number = int(block.get("page_index") or 0) + 1
        suffix = crop_path.suffix or ".png"
        asset_name = f"page-{page_number:02d}-figure-{image_index:02d}{suffix}"
        destination = asset_dir / asset_name
        shutil.copy2(crop_path, destination)
        names[str(block.get("block_id") or "")] = f"assets/{asset_name}"
    return names


if __name__ == "__main__":
    main()
