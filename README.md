<p align="center">
  <img src="assets/docprism-brand.png" width="460" alt="DocPrism 品牌图">
</p>

<p align="center">
  <strong>让 VLM 只处理真正困难的页面</strong>
</p>

<p align="center">
  面向 Markdown 输出的选择性 PDF 解析服务
</p>

<p align="center">
  可编辑页走 <code>PyMuPDF</code> 快路径，扫描页、复杂表格和低置信区域再进入异步 VLM 队列。
</p>

<p align="center">
  <strong>ATLAS 64 页：</strong> 2.172s 主解析返回 · 29.46 页/s · VLM 只处理少量复杂内容
</p>

> 这不是“把整篇 PDF 丢给 VLM”的方案。DocPrism 的重点是先路由，再增强，只把真正难的部分交给 VLM。

## 一眼看懂

- **更快**：大多数可编辑 PDF 不走 OCR，不走全量 VLM。
- **更省**：多 GPU 只服务复杂页和增强任务，不浪费在普通正文页上。
- **更实用**：输出不只有文本，还包括 Markdown、图片 assets、块级结果和异步任务状态。

## 核心亮点

- **选择性 VLM 路由**：可编辑页走 `PyMuPDF`，扫描页和低置信页才进入 VLM。
- **页级 + 块级增强**：整页处理扫描页，局部 crop 处理复杂表格和图像区域。
- **直接可集成**：提供 FastAPI 接口、SQLite 持久化和 Markdown 导出脚本。
- **有实测结果**：当前仓库已提交 benchmark，64 页可编辑论文主解析可到 `29.46 页/s`。

## 快速开始

30 秒试一下：

```bash
python -m pip install -r requirements.txt
export MINERU_VLM_LAB_ROOT="$PWD"
export MINERU_VLM_LAB_WORK_ROOT="$PWD/work"
export MINERU_VLM_LAB_DB_PATH="$PWD/storage/docprism.sqlite3"
export MINERU_VLM_BASE_URL="http://127.0.0.1:18100"
export EMBEDDING_PROVIDER="hash"
export EMBEDDING_DIM="32"
python -m uvicorn app.main:app --host 0.0.0.0 --port 18180
curl -X POST http://127.0.0.1:18180/parse \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "persist=true" \
  -F "run_embedding=false"
```

说明：仓库里的脚本和环境变量前缀仍沿用历史命名 `MINERU_VLM_LAB_*`。如果上游 `MINERU_VLM_BASE_URL` 不可用，可编辑 PDF 仍然能正常走快路径。

## 接口

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
