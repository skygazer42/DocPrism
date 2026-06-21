from __future__ import annotations

import importlib.util
from pathlib import Path


def load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_parse_and_enhancements.py"
    spec = importlib.util.spec_from_file_location("bench_parse_and_enhancements", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_summary_reports_parse_and_enhancement_counts():
    bench = load_benchmark_module()

    summary = bench.build_result_summary(
        pdf_path=Path("/tmp/paper.pdf"),
        wait_enhancements="required",
        job={
            "job_id": "job-1",
            "status": "completed",
            "page_count": 12,
            "elapsed_seconds": 1.25,
            "timings": {"routing_seconds": 1.1},
            "enhancement_task_count": 3,
        },
        tasks=[
            {"status": "completed", "metadata": {"enhancement_priority": "required"}},
            {"status": "queued", "metadata": {"enhancement_priority": "optional"}},
            {"status": "completed", "metadata": {"enhancement_priority": "asset_only"}},
        ],
        parse_wall_seconds=1.5,
        enhancement_wall_seconds=0.75,
        total_wall_seconds=2.25,
    )

    assert summary["parse_wall_seconds"] == 1.5
    assert summary["parse_pages_per_second"] == 8.0
    assert summary["enhancement_wall_seconds"] == 0.75
    assert summary["total_wall_seconds"] == 2.25
    assert summary["required_enhancement_count"] == 1
    assert summary["optional_enhancement_count"] == 1
    assert summary["asset_only_enhancement_count"] == 1
    assert summary["pending_wait_enhancement_count"] == 0


def test_benchmark_wait_policy_filters_required_tasks():
    bench = load_benchmark_module()
    tasks = [
        {"status": "completed", "metadata": {"enhancement_priority": "required"}},
        {"status": "queued", "metadata": {"enhancement_priority": "optional"}},
        {"status": "queued", "metadata": {"enhancement_priority": "required"}},
    ]

    assert bench.pending_wait_tasks(tasks, "none") == []
    assert len(bench.pending_wait_tasks(tasks, "required")) == 1
    assert len(bench.pending_wait_tasks(tasks, "all")) == 2
