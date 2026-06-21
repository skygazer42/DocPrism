from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_replica_launcher_dry_run_plans_workers_and_aggregator():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "start_vlm_router_replicas.sh"
    env = os.environ.copy()
    env.update(
        {
            "MINERU_VLM_LAB_ROOT": str(root),
            "MINERU_ENV": "/opt/mineru-env",
            "MINERU_VLM_GPUS": "4,5",
            "MINERU_VLM_REPLICAS_PER_GPU": "2",
            "MINERU_VLM_WORKER_CONCURRENCY": "1",
            "MINERU_VLM_REPLICA_BASE_PORT": "24141",
            "MINERU_VLM_ROUTER_HOST": "127.0.0.1",
            "MINERU_VLM_ROUTER_PORT": "18102",
            "MINERU_VLM_REPLICA_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    worker_lines = [line for line in lines if line.startswith("WORKER ")]
    assert worker_lines == [
        "WORKER gpu=4 replica=1 port=24141 concurrency=1",
        "WORKER gpu=4 replica=2 port=24142 concurrency=1",
        "WORKER gpu=5 replica=1 port=24143 concurrency=1",
        "WORKER gpu=5 replica=2 port=24144 concurrency=1",
    ]
    router_lines = [line for line in lines if line.startswith("ROUTER ")]
    assert len(router_lines) == 1
    assert "--local-gpus none" in router_lines[0]
    assert "--upstream-url http://127.0.0.1:24141" in router_lines[0]
    assert "--upstream-url http://127.0.0.1:24144" in router_lines[0]


def test_direct_vlm_replica_launcher_dry_run_plans_gpu_worker_pool():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "start_direct_vlm_replicas.sh"
    env = os.environ.copy()
    env.update(
        {
            "MINERU_VLM_LAB_ROOT": str(root),
            "MINERU_ENV": "/opt/vllm-env",
            "DIRECT_VLM_GPUS": "4,5",
            "DIRECT_VLM_REPLICAS_PER_GPU": "2",
            "DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION": "0.42",
            "DIRECT_VLM_MAX_MODEL_LEN": "8192",
            "DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH": "1600",
            "DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH": "1400",
            "DIRECT_VLM_WORKER_ID_PREFIX": "direct-vlm",
            "DIRECT_VLM_REPLICA_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    worker_lines = [line for line in result.stdout.splitlines() if line.startswith("DIRECT_WORKER ")]
    assert worker_lines == [
        "DIRECT_WORKER gpu=4 replica=1 worker_id=direct-vlm-gpu4-1 limit=2 gpu_memory_utilization=0.42 max_model_len=8192",
        "DIRECT_WORKER gpu=4 replica=2 worker_id=direct-vlm-gpu4-2 limit=2 gpu_memory_utilization=0.42 max_model_len=8192",
        "DIRECT_WORKER gpu=5 replica=1 worker_id=direct-vlm-gpu5-1 limit=2 gpu_memory_utilization=0.42 max_model_len=8192",
        "DIRECT_WORKER gpu=5 replica=2 worker_id=direct-vlm-gpu5-2 limit=2 gpu_memory_utilization=0.42 max_model_len=8192",
    ]
    command_lines = [line for line in result.stdout.splitlines() if line.startswith("CMD ")]
    assert len(command_lines) == 4
    assert "/opt/vllm-env/bin/python" in command_lines[0]
    assert "--kind ''" in command_lines[0]
    assert "--gpu-memory-utilization 0.42" in command_lines[0]
    assert "--max-table-image-width 1600" in command_lines[0]
    assert "--max-page-image-width 1400" in command_lines[0]


def test_router_launcher_dry_run_includes_vllm_cache_args():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "start_vlm_router.sh"
    env = os.environ.copy()
    env.update(
        {
            "MINERU_VLM_LAB_ROOT": str(root),
            "MINERU_ENV": "/opt/vllm-env",
            "MINERU_VLM_GPUS": "4,5",
            "MINERU_VLM_ROUTER_PORT": "18103",
            "MINERU_VLLM_GPU_MEMORY_UTILIZATION": "0.83",
            "MINERU_VLLM_MAX_MODEL_LEN": "8192",
            "MINERU_VLLM_KV_CACHE_MEMORY_BYTES": "17179869184",
            "MINERU_VLM_ROUTER_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "/opt/vllm-env/bin/mineru-router" in command
    assert "--port 18103" in command
    assert "--local-gpus 4,5" in command
    assert "--gpu-memory-utilization 0.83" in command
    assert "--max-model-len 8192" in command
    assert "--kv-cache-memory-bytes 17179869184" in command


def test_orchestrator_launcher_dry_run_allows_embedding_env_overrides(tmp_path):
    root = Path(__file__).resolve().parents[1]
    lab_root = tmp_path / "lab"
    lab_root.mkdir()
    (lab_root / "embedding.env").write_text(
        "\n".join(
            [
                "EMBEDDING_PROVIDER=transformers",
                "EMBEDDING_DEVICE=cuda:4",
                "PRELOAD_EMBEDDING=true",
            ]
        ),
        encoding="utf-8",
    )
    script = root / "scripts" / "start_orchestrator.sh"
    env = os.environ.copy()
    env.update(
        {
            "MINERU_VLM_LAB_ROOT": str(lab_root),
            "MINERU_ENV": "/opt/mineru-env",
            "MINERU_VLM_BASE_URL": "http://127.0.0.1:18103",
            "EMBEDDING_PROVIDER": "hash",
            "EMBEDDING_DEVICE": "cpu",
            "PRELOAD_EMBEDDING": "false",
            "MINERU_ORCHESTRATOR_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ENV MINERU_VLM_BASE_URL=http://127.0.0.1:18103" in result.stdout
    assert "ENV EMBEDDING_PROVIDER=hash" in result.stdout
    assert "ENV EMBEDDING_DEVICE=cpu" in result.stdout
    assert "ENV PRELOAD_EMBEDDING=false" in result.stdout
    assert "CMD /opt/mineru-env/bin/uvicorn app.main:app" in result.stdout


def test_orchestrator_launcher_dry_run_loads_production_env(tmp_path):
    root = Path(__file__).resolve().parents[1]
    lab_root = tmp_path / "lab"
    lab_root.mkdir()
    (lab_root / "production.env").write_text(
        "\n".join(
            [
                "MINERU_VLM_BASE_URL=http://127.0.0.1:18103",
                "NATIVE_TABLE_EXTRACTION_MODE=defer",
                "DEFERRED_TABLE_QUEUE_MODE=complex",
                "DEFERRED_TABLE_MIN_COMPLEX_BLOCKS=5",
                "ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH=640",
                "ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH=768",
            ]
        ),
        encoding="utf-8",
    )
    script = root / "scripts" / "start_orchestrator.sh"
    env = os.environ.copy()
    env.update(
        {
            "MINERU_VLM_LAB_ROOT": str(lab_root),
            "MINERU_ENV": "/opt/mineru-env",
            "MINERU_ORCHESTRATOR_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ENV MINERU_VLM_BASE_URL=http://127.0.0.1:18103" in result.stdout
    assert "ENV NATIVE_TABLE_EXTRACTION_MODE=defer" in result.stdout
    assert "ENV DEFERRED_TABLE_QUEUE_MODE=complex" in result.stdout
    assert "ENV DEFERRED_TABLE_MIN_COMPLEX_BLOCKS=5" in result.stdout
    assert "ENV ENHANCEMENT_TABLE_MAX_IMAGE_WIDTH=640" in result.stdout
    assert "ENV ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH=768" in result.stdout
