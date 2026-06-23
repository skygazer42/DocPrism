# DocPrism

<p align="center">
  <img src="assets/logo.png" width="180" alt="DocPrism logo">
</p>

<p align="center">
  A selective PDF parsing pipeline for Markdown-first output.
</p>

DocPrism 是一个面向自部署场景的 PDF 解析实验项目：优先保留可编辑 PDF 的本地快路径，只把扫描页、图片页、复杂表格和低置信区域送进异步 VLM 队列。它的目标不是“整篇 PDF 全量交给 VLM”，而是在可控成本下拿到更稳定的 Markdown、图片资产和结构化块输出。

这个 README 的组织方式参考了同类主流项目常见的开源首页结构，例如 [MinerU](https://github.com/opendatalab/MinerU)、[Marker](https://github.com/VikParuchuri/marker)、[Unstructured](https://github.com/Unstructured-IO/unstructured)，但下面只描述当前仓库已经实现和验证过的能力。

## 项目定位

DocPrism 关注的是一个更窄、也更工程化的边界：

- 对大多数可编辑论文或报告，正文和简单块走 `PyMuPDF` 快路径，避免无意义 OCR/VLM 成本。
- 对复杂表格、扫描页、图片重页、低置信区域，生成 crop 或页级任务，交给常驻的 VLM worker 异步处理。
- 对外暴露统一的 FastAPI 接口，内部保留页级路由、块级持久化、增强任务队列和 Markdown 导出能力。
- 默认适合单机实验或单机多 GPU 部署；当前存储和队列实现是 SQLite + 本地文件路径。

如果你需要的是：

- 通用的文档解析/OCR/VLM 基座：更接近 `MinerU`。
- 本地直接把 PDF 转成 Markdown/JSON 的 CLI/库：更接近 `Marker`。
- 带连接器、分块、数据摄取编排的大系统：更接近 `Unstructured`。
- 一个可控的“选择性 VLM 路由 + 异步增强 + Markdown 导出”服务层：这是 DocPrism 试图解决的问题。

## 核心特性

- `PyMuPDF` 快路径：可编辑文本密集页默认不走 VLM。
- 选择性 VLM：扫描页、图片页、复杂表格和低置信页才进入 VLM 处理。
- 两种异步增强模型：
  - 页级 VLM：处理扫描页、图片重页或低置信整页。
  - 块级增强：处理复杂表格或需要补充理解的图像块。
- Markdown-first 输出：导出 front matter、分页内容、表格 Markdown 和图片 assets。
- 统一 API：支持同步实验接口 `POST /parse` 和异步任务接口 `POST /api/v1/jobs`。
- 基准与可观测性：支持 `/health`、`/api/v1/stats`、benchmark 脚本、SQLite 持久化结果。
- 多 GPU worker 形态：支持 direct vLLM workers 常驻消费增强任务。

## 与同类项目的关系

| 项目 | 主要关注点 | DocPrism 的关系 |
| --- | --- | --- |
| [MinerU](https://github.com/opendatalab/MinerU) | 通用文档解析、OCR/VLM 能力、丰富生态 | DocPrism 复用其 VLM 能力，但主张先做页级路由，而不是整篇文档直接走完整 pipeline |
| [Marker](https://github.com/VikParuchuri/marker) | 本地 PDF to Markdown/JSON 转换 | DocPrism 增加服务化接口、异步增强队列和多 GPU worker 编排 |
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | 文档摄取、连接器、清洗与 chunking | DocPrism 更聚焦解析入口本身，不覆盖大规模连接器和 ingestion 编排 |

## 工作流

1. 上传 PDF 到 `POST /parse` 或 `POST /api/v1/jobs`。
2. `PyMuPDF` 收集每页信号：
   - 文本字符数
   - block 数
   - 图片数
   - 图片面积占比
   - 表格候选迹象
3. 路由决策：
   - `fast_pymupdf`：可编辑、文本密集页面
   - `vlm` + `async_direct_vlm` / `hybrid_ocr`：扫描页、图片页、低置信页
   - `enhancement_task`：复杂表格、需要局部增强的图像或页内区域
4. 结果落到 SQLite 与本地工作目录：
   - `jobs`
   - `pages`
   - `blocks`
   - `embeddings`
   - `enhancement_tasks`
5. `scripts/export_job_markdown.py` 根据块结果和增强结果导出 Markdown 与图片资产。

## 当前仓库真实支持的内容类型

| 内容类型 | 默认处理方式 | 是否必走 VLM |
| --- | --- | --- |
| 可编辑正文 | `PyMuPDF` 快路径 | 否 |
| 普通图片/图表资产 | 裁图保存到 `assets/`，Markdown 引用 | 否 |
| 简单可编辑表格 | `PyMuPDF` 文本/原生表格/导出重排 | 否 |
| 复杂表格/扫描表格 | crop 后进入增强队列 | 是 |
| 扫描页/图片重页 | OCR 优先，必要时页级 VLM | 视情况 |

## 项目状态

- 当前定位：实验性但可部署的单机服务。
- 已覆盖内容：API、页级路由、块存储、增强队列、Markdown 导出、基准脚本、测试。
- 已验证场景：可编辑学术论文、图片页、扫描页、复杂表格候选。
- 当前默认存储：SQLite，本地工作目录。
- 当前不做的事：连接器、分布式任务系统、对象存储抽象、多租户 SaaS 能力。

## 快速开始

### 1. 环境要求

- Python 3 环境
- Linux
- 对可编辑 PDF 的本地解析：只需安装 Python 依赖
- 对扫描页/复杂表格的真实 VLM 增强：需要可用的 MinerU VLM backend 或 direct vLLM workers
- 可选 OCR：`tesseract`

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### 3. 启动一个本地 smoke 服务

仓库里的脚本和环境变量前缀目前仍保留历史命名 `MINERU_VLM_LAB_*`，运行时请显式把根目录指到当前仓库：

```bash
export MINERU_VLM_LAB_ROOT="$PWD"
export MINERU_VLM_LAB_WORK_ROOT="$PWD/work"
export MINERU_VLM_LAB_DB_PATH="$PWD/storage/docprism.sqlite3"
export MINERU_VLM_BASE_URL="http://127.0.0.1:18100"
export EMBEDDING_PROVIDER="hash"
export EMBEDDING_DIM="32"

python -m uvicorn app.main:app --host 0.0.0.0 --port 18180
```

说明：

- 这套最小启动适合验证 API、可编辑 PDF 快路径、SQLite 落库和 Markdown 导出。
- 如果 `MINERU_VLM_BASE_URL` 不可用，扫描页或复杂 VLM 增强不会真正完成，但可编辑 PDF 仍可正常走快路径。

### 4. 健康检查

```bash
curl -s http://127.0.0.1:18180/health
curl -s http://127.0.0.1:18180/api/v1/stats
```

注意：`/health` 里如果 `mineru_vlm` 显示 `unavailable`，只代表上游 VLM backend 不可用，不代表 orchestrator 本身没有启动。

## 生产型启动方式

如果你要复现仓库里已经验证过的单机多 GPU 形态，可以直接复用现有脚本和配置模板。

### 1. 准备配置

```bash
export MINERU_VLM_LAB_ROOT="$PWD"
cp configs/production-6gpu-4090.env.example production.env
```

### 2. 下载模型

```bash
./scripts/download_vlm_model.sh
```

### 3. 启动 orchestrator

```bash
nohup ./scripts/start_orchestrator.sh > logs/orchestrator.log 2>&1 &
```

### 4. 启动 direct vLLM workers

```bash
nohup ./scripts/start_direct_vlm_replicas.sh > logs/direct-vlm-replicas.log 2>&1 &
```

### 5. 推荐的 6 卡 4090 关键参数

```bash
DIRECT_VLM_GPUS=0,1,2,3,4,5
DIRECT_VLM_REPLICAS_PER_GPU=1
DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION=0.90
DIRECT_VLM_CLAIM_LIMIT=2
DIRECT_VLM_MAX_MODEL_LEN=8192
DIRECT_VLM_MAX_IMAGE_WIDTH=512
DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH=640
DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH=1024
NATIVE_TABLE_EXTRACTION_MODE=defer
DEFERRED_TABLE_QUEUE_MODE=complex
SCAN_OCR_MODE=auto
FAST_TEXT_CHUNK_CHARS=5000
```

更多参数背景见：

- [架构说明](docs/architecture.md)
- [生产目标与验收要求](docs/production-requirements.md)
- [6 卡 4090 部署方案与验证结论](docs/deployment-plan-20260621.md)

## API

### 同步解析

```bash
curl -X POST http://127.0.0.1:18180/parse \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "persist=true" \
  -F "run_embedding=false"
```

### 异步解析

```bash
curl -X POST http://127.0.0.1:18180/api/v1/jobs \
  -F "file=@/path/to/file.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "run_embedding=false"
```

### 查询任务状态与结果

```bash
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>/blocks
curl http://127.0.0.1:18180/api/v1/jobs/<job_id>/enhancements
```

### 典型返回字段

`POST /parse` 返回的核心字段包括：

- `request_id`
- `page_count`
- `fast_page_count`
- `vlm_page_count`
- `block_count`
- `embedding_count`
- `enhancement_task_count`
- `timings`
- `routes`
- `blocks`
- `enhancement_tasks`
- `storage`

示例结构：

```json
{
  "request_id": "4f4d0b1b3baf4d0a9e7c4e83fcba8a2f",
  "page_count": 12,
  "fast_page_count": 11,
  "vlm_page_count": 1,
  "block_count": 37,
  "embedding_count": 0,
  "enhancement_task_count": 3,
  "timings": {
    "routing_seconds": 0.41,
    "vlm_seconds": 1.76,
    "embedding_seconds": 0.0,
    "storage_seconds": 0.05,
    "total_seconds": 2.31
  }
}
```

## Markdown 导出

```bash
python scripts/export_job_markdown.py <job_id> \
  --output-dir exports/<job_id> \
  --title "Document Title" \
  --wait-enhancements all
```

导出结果包含：

- `*.md`
- `assets/*.png`
- YAML front matter
- 分页后的块内容
- 可用时替换为增强后的表格 Markdown 或图像说明

Markdown 头部形态如下：

```md
---
title: "Document Title"
source_pdf: "paper.pdf"
job_id: "..."
page_count: 12
block_count: 37
enhancement_task_count: 3
---
```

## 已验证的 benchmark

当前仓库已提交：

- `reports/paper-bench-20260621.jsonl`

6 卡 RTX 4090 节点上的已记录结果如下：

| 论文 | 页数 | 只等主解析 | 等待可选 VLM 增强 |
| --- | ---: | ---: | ---: |
| ResNet | 12 | 0.709s, 16.94 页/s | 2.814s, 4.26 页/s |
| ATLAS | 64 | 2.172s, 29.46 页/s | 4.475s, 14.30 页/s |
| Attention Is All You Need | 15 | 1.431s, 10.49 页/s | 3.526s, 4.25 页/s |
| U-Net | 8 | 0.410s, 19.49 页/s | 0.360s, 22.24 页/s |

这些数字支持的结论是：

- 典型可编辑论文的吞吐主要由 `PyMuPDF` 路径和本地处理决定。
- 多 GPU 的价值主要体现在 VLM 队列，而不是加速所有可编辑正文页。
- “只等主解析”和“等待所有可选增强”是两种不同指标，前者更适合看解析吞吐，后者更适合看最终 Markdown 完整度。

## 测试

运行测试：

```bash
python -m pytest -q
```

运行 benchmark：

```bash
python scripts/bench_parse_and_enhancements.py \
  --base-url http://127.0.0.1:18180 \
  --pdf /path/to/file.pdf \
  --wait-enhancements all \
  --timeout 360 \
  --poll-interval 0.05
```

## 仓库结构

```text
app/        FastAPI 服务、页级路由、worker、存储、Markdown 导出
configs/    生产配置模板
docs/       架构、生产要求、部署验证文档
reports/    benchmark 结果
scripts/    启动、下载模型、导出、benchmark 脚本
tests/      单元测试与端到端测试
```

## 已知边界

- 当前默认队列和存储模型是 SQLite + 本地文件，适合单机实验，不是多节点生产终态。
- 当前 README 只覆盖仓库已实现内容；更早的旧命名如 `PageLens`、`mineru-vlm-lab` 仍残留在部分脚本和文档中，用于兼容现有环境变量与部署路径。
- Benchmark 目前主要覆盖可编辑论文与选择性增强场景；扫描 PDF、复杂表格密集 PDF 还需要继续补更多批量样本。
- Embedding 可以接入真实 provider，但当前提交的 benchmark 主要把它当作可开关的辅助阶段，而不是主要性能指标。

## 适合与不适合

适合：

- 自部署、可控成本的 PDF 解析服务
- 可编辑 PDF 占大头，但仍需要处理少量扫描页或复杂表格
- 希望保留 Markdown、图片资产、块级结果和增强队列

不适合：

- 直接替代成熟的通用 ingestion 平台
- 纯 CLI 一次性离线转换所有文档，不关心服务层和队列
- 多租户生产系统的最终形态

## 后续阅读

- [架构说明](docs/architecture.md)
- [生产目标与验收要求](docs/production-requirements.md)
- [6 卡 4090 部署方案与验证结论](docs/deployment-plan-20260621.md)
