# DocPrism

<p align="center">
  <img src="assets/docprism-brand.png" width="820" alt="DocPrism 品牌图">
</p>

<p align="center">
  面向 Markdown 输出的选择性 PDF 解析服务
</p>

DocPrism 是一个自部署 PDF 解析服务：可编辑页面优先走 `PyMuPDF` 快路径，扫描页、图片页、复杂表格和低置信区域再进入异步 VLM 队列。目标不是把整篇 PDF 全量丢给 VLM，而是在延迟、成本和输出质量之间做更稳的工程取舍。

## 为什么做

- 可编辑 PDF 不该默认走 OCR / VLM。
- 多 GPU 更适合加速复杂页和增强队列，不适合浪费在所有正文页上。
- 最终输出不只是文本，还要保留 Markdown、图片资产和块级结果。

## 核心能力

- `PyMuPDF` 快路径处理可编辑正文。
- 页级路由，把扫描页和低置信页送入 VLM。
- 块级增强，专门处理复杂表格和图像区域。
- FastAPI 接口，支持同步解析和异步任务。
- SQLite 持久化 `jobs / blocks / embeddings / enhancement_tasks`。
- Markdown 导出，附带 `assets/` 图片资源。

## 快速开始

安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

启动本地服务：

```bash
export MINERU_VLM_LAB_ROOT="$PWD"
export MINERU_VLM_LAB_WORK_ROOT="$PWD/work"
export MINERU_VLM_LAB_DB_PATH="$PWD/storage/docprism.sqlite3"
export MINERU_VLM_BASE_URL="http://127.0.0.1:18100"
export EMBEDDING_PROVIDER="hash"
export EMBEDDING_DIM="32"

python -m uvicorn app.main:app --host 0.0.0.0 --port 18180
```

健康检查：

```bash
curl -s http://127.0.0.1:18180/health
curl -s http://127.0.0.1:18180/api/v1/stats
```

说明：仓库里的脚本和环境变量前缀仍沿用历史命名 `MINERU_VLM_LAB_*`。如果上游 `MINERU_VLM_BASE_URL` 不可用，可编辑 PDF 仍然能正常走快路径。

## 接口

同步解析：

```bash
curl -X POST http://127.0.0.1:18180/parse \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "persist=true" \
  -F "run_embedding=false"
```

异步解析：

```bash
curl -X POST http://127.0.0.1:18180/api/v1/jobs \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "run_embedding=false"
```

查询结果：

```bash
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>/blocks
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>/enhancements
```

导出 Markdown：

```bash
python scripts/export_job_markdown.py <job_id> \
  --output-dir exports/<job_id> \
  --title "Document Title" \
  --wait-enhancements all
```

## 部署

单机多 GPU 部署可直接复用现有模板：

```bash
export MINERU_VLM_LAB_ROOT="$PWD"
cp configs/production-6gpu-4090.env.example production.env
./scripts/download_vlm_model.sh
nohup ./scripts/start_orchestrator.sh > logs/orchestrator.log 2>&1 &
nohup ./scripts/start_direct_vlm_replicas.sh > logs/direct-vlm-replicas.log 2>&1 &
```

当前已验证的 6 卡 4090 关键参数：

```bash
DIRECT_VLM_GPUS=0,1,2,3,4,5
DIRECT_VLM_REPLICAS_PER_GPU=1
DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION=0.90
DIRECT_VLM_CLAIM_LIMIT=2
DIRECT_VLM_MAX_MODEL_LEN=8192
DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH=640
DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH=1024
NATIVE_TABLE_EXTRACTION_MODE=defer
DEFERRED_TABLE_QUEUE_MODE=complex
SCAN_OCR_MODE=auto
FAST_TEXT_CHUNK_CHARS=5000
```

## 基准结果

已提交结果文件：`reports/paper-bench-20260621.jsonl`

| 论文 | 页数 | 只等主解析 | 等待可选 VLM 增强 |
| --- | ---: | ---: | ---: |
| ResNet | 12 | 0.709s, 16.94 页/s | 2.814s, 4.26 页/s |
| ATLAS | 64 | 2.172s, 29.46 页/s | 4.475s, 14.30 页/s |
| Attention Is All You Need | 15 | 1.431s, 10.49 页/s | 3.526s, 4.25 页/s |
| U-Net | 8 | 0.410s, 19.49 页/s | 0.360s, 22.24 页/s |

结论很简单：大多数可编辑论文的吞吐主要来自 `PyMuPDF` 快路径，多 GPU 的价值主要体现在 VLM 页和增强任务，而不是所有页面一起上 VLM。

## 仓库结构

```text
app/        FastAPI 服务、路由、worker、存储、Markdown 导出
configs/    部署配置模板
docs/       架构、生产要求、部署验证
reports/    benchmark 结果
scripts/    启动、下载模型、导出、benchmark
tests/      单元与端到端测试
```

## 更多说明

- [架构说明](docs/architecture.md)
- [生产目标与验收要求](docs/production-requirements.md)
- [6 卡 4090 部署方案与验证结论](docs/deployment-plan-20260621.md)
