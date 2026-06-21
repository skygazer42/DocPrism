from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class PageRoute(BaseModel):
    page_index: int
    route: str
    reason: str
    text_chars: int
    block_count: int
    image_count: int
    image_area_ratio: float


class StorageStats(BaseModel):
    stored: bool
    db_path: str | None = None
    block_count: int = 0
    embedding_count: int = 0
    enhancement_task_count: int = 0


class ParseSummary(BaseModel):
    request_id: str
    file_name: str
    page_count: int
    elapsed_seconds: float
    timings: dict[str, float] = Field(default_factory=dict)
    fast_page_count: int
    vlm_page_count: int
    block_count: int
    embedding_count: int
    enhancement_task_count: int
    routes: list[PageRoute]
    fast_pages: list[dict[str, Any]]
    vlm_pages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    enhancement_tasks: list[dict[str, Any]]
    storage: StorageStats


class JobCreated(BaseModel):
    job_id: str
    status: str
    status_url: str
    blocks_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    file_name: str | None = None
    page_count: int = 0
    fast_page_count: int = 0
    vlm_page_count: int = 0
    block_count: int = 0
    embedding_count: int = 0
    enhancement_task_count: int = 0
    elapsed_seconds: float | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None


class BlocksResponse(BaseModel):
    job_id: str
    total: int
    blocks: list[dict[str, Any]]


class EnhancementsResponse(BaseModel):
    job_id: str
    total: int
    tasks: list[dict[str, Any]]


class RuntimeStatsResponse(BaseModel):
    jobs: dict[str, Any]
    pages: dict[str, Any]
    blocks: dict[str, Any]
    embeddings: dict[str, Any]
    enhancements: dict[str, Any]


class EnhancementClaimRequest(BaseModel):
    limit: int = 1
    worker_id: str = "default"
    job_id: str | None = None
    kind: str | None = None
    lease_timeout_seconds: float | None = Field(default=None, ge=0)


class EnhancementClaimResponse(BaseModel):
    total: int
    tasks: list[dict[str, Any]]


class EnhancementCompleteRequest(BaseModel):
    worker_id: str = "default"
    result: dict[str, Any] = Field(default_factory=dict)


class EnhancementFailRequest(BaseModel):
    worker_id: str = "default"
    error: str


@dataclass(frozen=True)
class PageSignal:
    page_index: int
    text_chars: int
    block_count: int
    image_count: int
    image_area_ratio: float
    has_table_hint: bool


@dataclass(frozen=True)
class BlockRecord:
    job_id: str
    page_index: int
    block_index: int
    source: str
    kind: str
    text: str
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def block_id(self) -> str:
        payload = {
            "job_id": self.job_id,
            "page_index": self.page_index,
            "block_index": self.block_index,
            "source": self.source,
            "kind": self.kind,
            "text": self.text,
            "bbox": self.bbox,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return f"{self.job_id}-p{self.page_index:04d}-b{self.block_index:04d}-{digest[:12]}"

    @property
    def embedding_text(self) -> str:
        return " ".join(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "job_id": self.job_id,
            "page_index": self.page_index,
            "block_index": self.block_index,
            "source": self.source,
            "kind": self.kind,
            "text": self.text,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EnhancementTask:
    job_id: str
    block_id: str
    page_index: int
    kind: str
    status: str
    crop_path: str
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        payload = {
            "job_id": self.job_id,
            "block_id": self.block_id,
            "kind": self.kind,
            "crop_path": self.crop_path,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return f"{self.job_id}-enh-{digest[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "block_id": self.block_id,
            "page_index": self.page_index,
            "kind": self.kind,
            "status": self.status,
            "crop_path": self.crop_path,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }
