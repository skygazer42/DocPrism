from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.models import BlockRecord, EnhancementTask, JobStatus, PageRoute, StorageStats


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    file_name TEXT,
                    source_path TEXT,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    fast_page_count INTEGER NOT NULL DEFAULT 0,
                    vlm_page_count INTEGER NOT NULL DEFAULT 0,
                    block_count INTEGER NOT NULL DEFAULT 0,
                    embedding_count INTEGER NOT NULL DEFAULT 0,
                    enhancement_task_count INTEGER NOT NULL DEFAULT 0,
                    elapsed_seconds REAL,
                    timings_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    job_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    route TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    text_chars INTEGER NOT NULL,
                    block_count INTEGER NOT NULL,
                    image_count INTEGER NOT NULL,
                    image_area_ratio REAL NOT NULL,
                    PRIMARY KEY (job_id, page_index)
                );
                CREATE TABLE IF NOT EXISTS blocks (
                    block_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    block_index INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    bbox_json TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    block_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS enhancement_tasks (
                    task_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    crop_path TEXT NOT NULL,
                    bbox_json TEXT,
                    metadata_json TEXT NOT NULL,
                    worker_id TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    leased_at REAL,
                    completed_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_blocks_job ON blocks(job_id, page_index, block_index);
                CREATE INDEX IF NOT EXISTS idx_embeddings_job ON embeddings(job_id);
                CREATE INDEX IF NOT EXISTS idx_enhancement_tasks_job ON enhancement_tasks(job_id, status);
                """
            )
            self._ensure_column(conn, "jobs", "enhancement_task_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "jobs", "timings_json", "TEXT")
            self._ensure_column(conn, "enhancement_tasks", "worker_id", "TEXT")
            self._ensure_column(conn, "enhancement_tasks", "result_json", "TEXT")
            self._ensure_column(conn, "enhancement_tasks", "error", "TEXT")
            self._ensure_column(conn, "enhancement_tasks", "leased_at", "REAL")
            self._ensure_column(conn, "enhancement_tasks", "completed_at", "REAL")

    def start_job(self, job_id: str, file_name: str, source_path: str, status: str = "queued") -> None:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    job_id, status, file_name, source_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, status, file_name, source_path, now, now),
            )

    def mark_processing(self, job_id: str) -> None:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'processing', started_at = COALESCE(started_at, ?), updated_at = ? WHERE job_id = ?",
                (now, now, job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error = ?, updated_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (error[:4000], now, now, job_id),
            )

    def mark_interrupted_jobs(self) -> int:
        now = time.time()
        with self._locked_connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = 'interrupted by service restart',
                    updated_at = ?,
                    completed_at = ?
                WHERE status IN ('queued', 'processing')
                """,
                (now, now),
            )
            return cursor.rowcount

    def store_outputs(
        self,
        *,
        job_id: str,
        file_name: str,
        source_path: str,
        page_count: int,
        fast_page_count: int,
        vlm_page_count: int,
        elapsed_seconds: float,
        timings: dict[str, float] | None = None,
        routes: list[PageRoute],
        blocks: list[BlockRecord],
        embeddings: dict[str, dict[str, Any]],
        enhancement_tasks: list[EnhancementTask],
    ) -> StorageStats:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute("DELETE FROM pages WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM embeddings WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM enhancement_tasks WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM blocks WHERE job_id = ?", (job_id,))
            conn.executemany(
                """
                INSERT INTO pages (
                    job_id, page_index, route, reason, text_chars, block_count, image_count, image_area_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        route.page_index,
                        route.route,
                        route.reason,
                        route.text_chars,
                        route.block_count,
                        route.image_count,
                        route.image_area_ratio,
                    )
                    for route in routes
                ],
            )
            conn.executemany(
                """
                INSERT INTO blocks (
                    block_id, job_id, page_index, block_index, source, kind, text, bbox_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        block.block_id,
                        job_id,
                        block.page_index,
                        block.block_index,
                        block.source,
                        block.kind,
                        block.text,
                        json.dumps(block.bbox, ensure_ascii=False) if block.bbox is not None else None,
                        json.dumps(block.metadata, ensure_ascii=False, sort_keys=True),
                        now,
                    )
                    for block in blocks
                ],
            )
            conn.executemany(
                """
                INSERT INTO embeddings (
                    embedding_id, block_id, job_id, provider, model, dim, vector_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        value["embedding_id"],
                        block_id,
                        job_id,
                        value["provider"],
                        value["model"],
                        value["dim"],
                        json.dumps(value["vector"], ensure_ascii=False),
                        now,
                    )
                    for block_id, value in embeddings.items()
                ],
            )
            conn.executemany(
                """
                INSERT INTO enhancement_tasks (
                    task_id, job_id, block_id, page_index, kind, status, crop_path,
                    bbox_json, metadata_json, worker_id, result_json, error,
                    created_at, updated_at, leased_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        task.task_id,
                        job_id,
                        task.block_id,
                        task.page_index,
                        task.kind,
                        task.status,
                        task.crop_path,
                        json.dumps(task.bbox, ensure_ascii=False) if task.bbox is not None else None,
                        json.dumps(task.metadata, ensure_ascii=False, sort_keys=True),
                        None,
                        None,
                        None,
                        now,
                        now,
                        None,
                        None,
                    )
                    for task in enhancement_tasks
                ],
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, file_name, source_path, page_count, fast_page_count, vlm_page_count,
                    block_count, embedding_count, enhancement_task_count, elapsed_seconds, timings_json,
                    created_at, updated_at, started_at, completed_at
                ) VALUES (?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = 'completed',
                    file_name = excluded.file_name,
                    source_path = excluded.source_path,
                    page_count = excluded.page_count,
                    fast_page_count = excluded.fast_page_count,
                    vlm_page_count = excluded.vlm_page_count,
                    block_count = excluded.block_count,
                    embedding_count = excluded.embedding_count,
                    enhancement_task_count = excluded.enhancement_task_count,
                    elapsed_seconds = excluded.elapsed_seconds,
                    timings_json = excluded.timings_json,
                    error = NULL,
                    updated_at = excluded.updated_at,
                    started_at = COALESCE(jobs.started_at, excluded.started_at),
                    completed_at = excluded.completed_at
                """,
                (
                    job_id,
                    file_name,
                    source_path,
                    page_count,
                    fast_page_count,
                    vlm_page_count,
                    len(blocks),
                    len(embeddings),
                    len(enhancement_tasks),
                    elapsed_seconds,
                    json.dumps(timings or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                    now,
                    now,
                ),
            )
        return StorageStats(
            stored=True,
            db_path=str(self.db_path),
            block_count=len(blocks),
            embedding_count=len(embeddings),
            enhancement_task_count=len(enhancement_tasks),
        )

    def update_job_elapsed(self, job_id: str, elapsed_seconds: float, timings: dict[str, float]) -> None:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET elapsed_seconds = ?, timings_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (elapsed_seconds, json.dumps(timings, ensure_ascii=False, sort_keys=True), now, job_id),
            )

    def get_job(self, job_id: str) -> JobStatus | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        timings_json = item.pop("timings_json", None)
        item["timings"] = json.loads(timings_json) if timings_json else {}
        return JobStatus(**item)

    def list_blocks(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    b.block_id, b.job_id, b.page_index, b.block_index, b.source, b.kind, b.text,
                    b.bbox_json, b.metadata_json, e.embedding_id, e.provider AS embedding_provider,
                    e.model AS embedding_model, e.dim AS embedding_dim
                FROM blocks b
                LEFT JOIN embeddings e ON e.block_id = b.block_id
                WHERE b.job_id = ?
                ORDER BY b.page_index, b.block_index
                """,
                (job_id,),
            ).fetchall()
        blocks: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["bbox"] = json.loads(item.pop("bbox_json")) if item.get("bbox_json") else None
            item["metadata"] = json.loads(item.pop("metadata_json")) if item.get("metadata_json") else {}
            blocks.append(item)
        return blocks

    def list_enhancement_tasks(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    task_id, job_id, block_id, page_index, kind, status, crop_path,
                    bbox_json, metadata_json, worker_id, result_json, error, leased_at, completed_at
                FROM enhancement_tasks
                WHERE job_id = ?
                ORDER BY page_index, task_id
                """,
                (job_id,),
            ).fetchall()
        tasks: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["bbox"] = json.loads(item.pop("bbox_json")) if item.get("bbox_json") else None
            item["metadata"] = json.loads(item.pop("metadata_json")) if item.get("metadata_json") else {}
            item["result"] = json.loads(item.pop("result_json")) if item.get("result_json") else None
            tasks.append(item)
        return tasks

    def runtime_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            job_counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
                ).fetchall()
            }
            job_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(page_count), 0) AS pages,
                    COALESCE(SUM(fast_page_count), 0) AS fast_pages,
                    COALESCE(SUM(vlm_page_count), 0) AS vlm_pages,
                    COALESCE(SUM(block_count), 0) AS blocks,
                    COALESCE(SUM(embedding_count), 0) AS embeddings,
                    COALESCE(AVG(elapsed_seconds), 0.0) AS avg_elapsed_seconds,
                    COALESCE(MAX(elapsed_seconds), 0.0) AS max_elapsed_seconds
                FROM jobs
                """
            ).fetchone()
            enhancement_counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM enhancement_tasks GROUP BY status"
                ).fetchall()
            }
            enhancement_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COALESCE(MIN(created_at), 0.0) AS oldest_created_at,
                    COALESCE(MIN(CASE WHEN status = 'queued' THEN created_at END), 0.0) AS oldest_queued_at
                FROM enhancement_tasks
                """
            ).fetchone()

        total_jobs = int(job_row["total"] or 0)
        total_pages = int(job_row["pages"] or 0)
        fast_pages = int(job_row["fast_pages"] or 0)
        vlm_pages = int(job_row["vlm_pages"] or 0)
        total_blocks = int(job_row["blocks"] or 0)
        total_embeddings = int(job_row["embeddings"] or 0)
        return {
            "jobs": {
                "total": total_jobs,
                "queued": int(job_counts.get("queued", 0)),
                "processing": int(job_counts.get("processing", 0)),
                "completed": int(job_counts.get("completed", 0)),
                "failed": int(job_counts.get("failed", 0)),
                "avg_elapsed_seconds": round(float(job_row["avg_elapsed_seconds"] or 0.0), 3),
                "max_elapsed_seconds": round(float(job_row["max_elapsed_seconds"] or 0.0), 3),
            },
            "pages": {
                "total": total_pages,
                "fast": fast_pages,
                "vlm": vlm_pages,
            },
            "blocks": {
                "total": total_blocks,
            },
            "embeddings": {
                "total": total_embeddings,
                "coverage": round(total_embeddings / total_blocks, 4) if total_blocks else 0.0,
            },
            "enhancements": {
                "total": int(enhancement_row["total"] or 0),
                "queued": int(enhancement_counts.get("queued", 0)),
                "processing": int(enhancement_counts.get("processing", 0)),
                "completed": int(enhancement_counts.get("completed", 0)),
                "failed": int(enhancement_counts.get("failed", 0)),
                "oldest_created_at": float(enhancement_row["oldest_created_at"] or 0.0),
                "oldest_queued_at": float(enhancement_row["oldest_queued_at"] or 0.0),
            },
        }

    def claim_enhancement_tasks(
        self,
        limit: int,
        worker_id: str,
        job_id: str | None = None,
        kind: str | None = None,
        *,
        lease_timeout_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        now = time.time()
        limit = max(1, min(limit, 100))
        lease_cutoff = now - max(0.0, lease_timeout_seconds) if lease_timeout_seconds is not None else None
        status_clause = "status = 'queued'"
        params: list[Any] = []
        if lease_cutoff is not None:
            status_clause = "(status = 'queued' OR (status = 'processing' AND leased_at IS NOT NULL AND leased_at <= ?))"
            params.append(lease_cutoff)
        where_clauses = [status_clause]
        if job_id:
            where_clauses.append("job_id = ?")
            params.append(job_id)
        if kind:
            where_clauses.append("kind = ?")
            params.append(kind)
        params.append(limit)
        with self._locked_connect() as conn:
            rows = conn.execute(
                f"""
                SELECT task_id
                FROM enhancement_tasks
                WHERE {' AND '.join(where_clauses)}
                ORDER BY created_at, task_id
                LIMIT ?
                """,
                params,
            ).fetchall()
            task_ids = [row["task_id"] for row in rows]
            if not task_ids:
                return []
            conn.executemany(
                """
                UPDATE enhancement_tasks
                SET status = 'processing', worker_id = ?, leased_at = ?, updated_at = ?, error = NULL
                WHERE task_id = ?
                """,
                [(worker_id, now, now, task_id) for task_id in task_ids],
            )
            claimed = conn.execute(
                f"""
                SELECT
                    task_id, job_id, block_id, page_index, kind, status, crop_path,
                    bbox_json, metadata_json, worker_id, result_json, error, leased_at, completed_at
                FROM enhancement_tasks
                WHERE task_id IN ({','.join('?' for _ in task_ids)})
                ORDER BY page_index, task_id
                """,
                task_ids,
            ).fetchall()
        return [self._enhancement_row_to_dict(row) for row in claimed]

    def complete_enhancement_task(self, task_id: str, worker_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute(
                """
                UPDATE enhancement_tasks
                SET status = 'completed', worker_id = ?, result_json = ?, error = NULL,
                    updated_at = ?, completed_at = ?
                WHERE task_id = ?
                """,
                (worker_id, json.dumps(result, ensure_ascii=False, sort_keys=True), now, now, task_id),
            )
            row = self._fetch_enhancement_row(conn, task_id)
        return self._enhancement_row_to_dict(row) if row is not None else None

    def fail_enhancement_task(self, task_id: str, worker_id: str, error: str) -> dict[str, Any] | None:
        now = time.time()
        with self._locked_connect() as conn:
            conn.execute(
                """
                UPDATE enhancement_tasks
                SET status = 'failed', worker_id = ?, error = ?, updated_at = ?, completed_at = ?
                WHERE task_id = ?
                """,
                (worker_id, error[:4000], now, now, task_id),
            )
            row = self._fetch_enhancement_row(conn, task_id)
        return self._enhancement_row_to_dict(row) if row is not None else None

    def _fetch_enhancement_row(self, conn: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT
                task_id, job_id, block_id, page_index, kind, status, crop_path,
                bbox_json, metadata_json, worker_id, result_json, error, leased_at, completed_at
            FROM enhancement_tasks
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()

    def _enhancement_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["bbox"] = json.loads(item.pop("bbox_json")) if item.get("bbox_json") else None
        item["metadata"] = json.loads(item.pop("metadata_json")) if item.get("metadata_json") else {}
        item["result"] = json.loads(item.pop("result_json")) if item.get("result_json") else None
        return item

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _locked_connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self._connect() as conn:
                yield conn
