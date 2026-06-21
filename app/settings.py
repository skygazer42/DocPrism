from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_root: Path
    work_root: Path
    db_path: Path
    mineru_vlm_base_url: str
    max_concurrent_vlm_pages: int
    embedding_provider: str
    embedding_dim: int
    embedding_batch_size: int
    max_concurrent_embedding_batches: int
    preload_embedding: bool


def load_settings() -> Settings:
    app_root = Path(os.getenv("MINERU_VLM_LAB_ROOT", "/data/mineru-vlm-lab"))
    work_root = Path(os.getenv("MINERU_VLM_LAB_WORK_ROOT", str(app_root / "work")))
    db_path = Path(os.getenv("MINERU_VLM_LAB_DB_PATH", str(app_root / "storage" / "mineru-vlm-lab.sqlite3")))
    return Settings(
        app_root=app_root,
        work_root=work_root,
        db_path=db_path,
        mineru_vlm_base_url=os.getenv("MINERU_VLM_BASE_URL", "http://127.0.0.1:18100"),
        max_concurrent_vlm_pages=_int_env("MAX_CONCURRENT_VLM_PAGES", 2),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        embedding_dim=_int_env("EMBEDDING_DIM", 32),
        embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 64),
        max_concurrent_embedding_batches=_int_env("MAX_CONCURRENT_EMBEDDING_BATCHES", 2),
        preload_embedding=_bool_env("PRELOAD_EMBEDDING", False),
    )
