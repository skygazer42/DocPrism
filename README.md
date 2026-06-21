# PageLens

<p align="center">
  <img src="assets/logo.png" width="180" alt="PageLens logo">
</p>

PageLens 是一个独立的 PDF 文档解析实验与部署项目。它不依赖 DocPilot，也不把 MinerU pipeline 当作主链路；核心目标是把可编辑 PDF、扫描页、图片页、复杂表格拆开路由，让 VLM 只处理真正需要视觉理解的部分。

当前项目用于验证：

- 可编辑论文是否可以用 PyMuPDF 快速解析。
- 复杂表格、扫描页、图片重页是否可以通过 VLM 异步增强。
- 多 GPU vLLM 常驻 worker 是否能支撑生产并发。
- Markdown 输出是否保留正文、图片、图号、表号和关键数值。

详细部署结论见：

- [6 卡 4090 部署方案与验证结论](docs/deployment-plan-20260621.md)
- [生产目标与验收要求](docs/production-requirements.md)
- [架构说明](docs/architecture.md)

## 核心结论

当前设计不是“整篇 PDF 全部丢给 VLM”。这样会慢，也浪费 GPU。

项目采用分层处理：

| 内容类型 | 默认处理方式 | 是否走 VLM |
| --- | --- | --- |
| 可编辑正文 | PyMuPDF 快路径 | 否 |
| 普通图片/图表资产 | 裁图保存到 assets，Markdown 引用 | 否 |
| 简单可编辑表格 | PyMuPDF 文本/native table/exporter 重排 | 否 |
| 复杂表格/扫描表格 | crop 后进入 enhancement queue | 是 |
| 扫描页/图片重页 | OCR 优先，必要时页级 VLM | 视情况 |

VLM 在这里主要做复杂表格、扫描页、图片页或局部图文块增强，不负责大多数可编辑 PDF 的 layout 和正文抽取。

## 6 卡实测

测试机器：6 张 RTX 4090，GPU 0-5。

Embedding 未纳入本轮性能测试。

| 论文 | 页数 | 只等主解析 | 等待可选 VLM 增强 |
| --- | ---: | ---: | ---: |
| ResNet | 12 | 0.709s, 16.94 页/s | 2.814s, 4.26 页/s |
| ATLAS | 64 | 2.172s, 29.46 页/s | 4.475s, 14.30 页/s |
| Attention Is All You Need | 15 | 1.431s, 10.49 页/s | 3.526s, 4.25 页/s |
| U-Net | 8 | 0.410s, 19.49 页/s | 0.360s, 22.24 页/s |

解释：

- ATLAS 64 页可编辑论文 parse-only 达到 29.46 页/s。
- 6 卡不能直接加速 PyMuPDF 的可编辑正文解析，因为那是 CPU/本地路径。
- 6 卡加速的是 VLM crop/page 队列。
- 当前路由优化后，典型可编辑论文只产生少量 VLM 任务，所以 GPU 常驻但不一定满载。

## 并发能力

项目支持三层并发：

1. 多文档 job 并发：`/api/v1/jobs` 会后台创建任务。
2. 页级并发：`MAX_CONCURRENT_FAST_PAGES` 和 `MAX_CONCURRENT_ASYNC_VLM_PAGE_RENDERS` 控制页面处理并发。
3. VLM worker 并发：6 卡配置下一卡一个 direct vLLM worker，每个 worker 每轮 claim 多个 enhancement task。

当前 6 卡配置核心参数：

```bash
DIRECT_VLM_GPUS=0,1,2,3,4,5
DIRECT_VLM_REPLICAS_PER_GPU=1
DIRECT_VLM_REPLICA_GPU_MEMORY_UTILIZATION=0.90
DIRECT_VLM_CLAIM_LIMIT=2
DIRECT_VLM_MAX_MODEL_LEN=8192
DIRECT_VLM_MAX_IMAGE_WIDTH=512
DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH=640
DIRECT_VLM_MAX_PAGE_IMAGE_WIDTH=1024
```

边界：

- 当前是单机 SQLite queue + 本地文件路径，适合单机 6 卡部署和实验。
- 多节点生产需要把 queue、对象存储、状态库拆出去，例如 Postgres/Redis/S3 兼容存储。

## 快速部署

在部署机上：

```bash
cd /data/mineru-vlm-lab
cp configs/production-6gpu-4090.env.example production.env
```

下载模型：

```bash
./scripts/download_vlm_model.sh
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动 orchestrator：

```bash
nohup ./scripts/start_orchestrator.sh > logs/orchestrator-6gpu.log 2>&1 &
```

启动 direct vLLM workers：

```bash
nohup ./scripts/start_direct_vlm_replicas.sh > logs/direct-vlm-replicas-6gpu.log 2>&1 &
```

健康检查：

```bash
curl -s http://127.0.0.1:18180/health
curl -s http://127.0.0.1:18180/api/v1/stats
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3,4,5
```

注意：direct worker 模式不依赖 `18100` 的 MinerU router。`/health` 里 `mineru_vlm_base_url` 显示 unavailable 时，不代表 direct vLLM workers 不可用。

## API

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
/data/mineru32-bench/env/bin/python scripts/export_job_markdown.py <job_id> \
  --output-dir exports/<job_id> \
  --title "Document Title" \
  --wait-enhancements all
```

## Markdown 输出验证

已验证 4 篇论文：

| 论文 | 图片 assets | 表格编号 | 图号编号 | 关键内容 |
| --- | ---: | --- | --- | --- |
| ResNet | 6 | Table 1-14 | Figure 1-7 | `ResNet-50`, `22.85`, `6.71` |
| ATLAS | 12 | Table 1-19 | Figure 1-11 | `ATLAS`, `13 TeV`, `95%`, `CL` |
| Attention | 6 | Table 1-4 | Figure 1-5 | `Attention Is All You Need`, `BLEU`, `Transformer` |
| U-Net | 12 | Table 1-2 | Figure 1-3 | `U-Net`, `biomedical`, `overlap-tile`, `ISBI` |

校验结论：

- Markdown 图片链接均能对应到本地 assets。
- 未发现 0 字节图片。
- 未发现异常串 `Figure 34. F`。
- U-Net 的正文内表格已经通过 exporter 重排为标准 Markdown pipe table。

## 停止服务和释放显存

停止本项目进程：

```bash
pkill -TERM -f '/data/mineru-vlm-lab/scripts/run_direct_vlm_worker.py' || true
pkill -TERM -f 'uvicorn app.main:app --host 0.0.0.0 --port 18180 --app-dir /data/mineru-vlm-lab' || true
```

如果 vLLM EngineCore 变成孤儿进程：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits -i 0,1,2,3,4,5
ps -fp <pid...>
kill -TERM <kdsoft EngineCore pid...>
```

只清理本项目的 `kdsoft` 进程，不要误杀其他用户或 root 服务。

## 测试

```bash
python -m pytest -q
```

Benchmark：

```bash
/data/mineru32-bench/env/bin/python scripts/bench_parse_and_enhancements.py \
  --base-url http://127.0.0.1:18180 \
  --pdf /path/to/file.pdf \
  --wait-enhancements all \
  --timeout 360 \
  --poll-interval 0.05
```

## 仓库内容

已提交：

- `app/`：服务、路由、VLM worker、存储、Markdown 导出。
- `scripts/`：模型下载、启动、benchmark、导出。
- `configs/`：6 卡 4090、8 卡、多 router 配置模板。
- `docs/`：部署方案、生产要求、架构说明。
- `tests/`：端到端、worker、路由、导出测试。
- `reports/paper-bench-20260621.jsonl`：本次论文 benchmark 结果。

未提交：

- 模型权重、运行日志、SQLite 数据库、PDF 样本、导出资产、work 目录。
