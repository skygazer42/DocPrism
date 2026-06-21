# PDF VLM Accelerator 部署方案与验证结论

日期：2026-06-21

目标：独立部署一个不依赖 DocPilot、不走 MinerU pipeline 主链路的 PDF 解析服务。服务只研究 VLM 能力和工程路由：可编辑 PDF 走 PyMuPDF 快路径，扫描页、复杂页、复杂表格或图片块才进入 VLM 队列。Embedding 当前不纳入性能目标。

## 1. 生产目标

业务目标来自当前讨论：

- MinerU 3.2 pipeline 单卡测试 64 页约 102 秒，约 1 页/2 秒，不满足上线。
- 生产目标是文档解析低延迟：64 页文档解析端到端需要接近 10 秒以内；若按专家说法，8 卡 L20 可达到约 26 页/秒。
- 当前机器最多可用 GPU 0-5，共 6 张 RTX 4090。当前阶段先验证“文档解析”，不计算入库和 Embedding。
- 不使用 DocPilot，不依赖 MinerU pipeline；服务独立成项目，模型权重从 ModelScope 下载。
- 生产提速方向不是“全页都吃 VLM”，而是路由优化、并发架构、服务常驻、页面级并发、异步化、只处理复杂页。

## 2. 当前架构

服务由两层组成：

1. Orchestrator
   - FastAPI 服务，默认端口 `18180`。
   - 接收 `/parse` 同步请求和 `/api/v1/jobs` 异步请求。
   - 负责页级路由、PyMuPDF 快速抽取、图片/表格 crop 生成、任务入 SQLite。

2. Direct VLM workers
   - 不走 transformers serving，不走 MinerU pipeline。
   - 使用 vLLM async engine 常驻加载 MinerU VLM 模型。
   - 每张 GPU 一个 worker，通过 enhancement queue 拉取任务。
   - 用 `gpu_memory_utilization=0.90` 和 `max_model_len=8192` 提高 KV cache 可用空间。

核心路由：

- 可编辑文字页：PyMuPDF 本地抽取，CPU 路径，极快，不占 VLM。
- 图/表资产：导出为 Markdown image link 和 assets 文件。
- 简单表格：优先使用 PyMuPDF 原文或导出阶段轻量表格重排。
- 复杂表格、扫描表格、噪声原生表格：进入 VLM crop enhancement。
- 扫描/图片重页：走 OCR 或 VLM 页级解析，避免全量文档无脑 VLM。

## 3. 推荐 6 卡 4090 配置

配置文件：

- `configs/production-6gpu-4090.env.example`
- 复制为 `production.env` 后启动。

关键参数：

```bash
MINERU_VLM_GPUS=0,1,2,3,4,5
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
DEFERRED_TABLE_MIN_COMPLEX_BLOCKS=5
DEFERRED_TABLE_MIN_COMPLEX_CHARS=700
EMBEDDING_PROVIDER=hash
PRELOAD_EMBEDDING=false
```

解释：

- `NATIVE_TABLE_EXTRACTION_MODE=defer`：生产默认不全局打开 PyMuPDF `find_tables()`，因为它会显著拖慢部分论文解析。
- `DEFERRED_TABLE_QUEUE_MODE=complex`：只把复杂表格送入 VLM，避免普通可编辑表格拖慢整体吞吐。
- `DIRECT_VLM_MAX_TABLE_IMAGE_WIDTH=640`：普通 crop 控制输入宽度，降低 VLM 延迟。
- `ENHANCEMENT_COMPLEX_TABLE_MAX_IMAGE_WIDTH=768`：复杂表格才提高宽度。
- `EMBEDDING_PROVIDER=hash`：当前 benchmark 只看解析，不把 embedding 服务波动混进结果。

## 4. 启动命令

在 110 主机：

```bash
cd /data/mineru-vlm-lab
cp configs/production-6gpu-4090.env.example production.env
```

启动 orchestrator：

```bash
nohup ./scripts/start_orchestrator.sh > logs/orchestrator-6gpu.log 2>&1 &
```

启动 6 卡 direct VLM workers：

```bash
nohup ./scripts/start_direct_vlm_replicas.sh > logs/direct-vlm-replicas-6gpu.log 2>&1 &
```

健康检查：

```bash
curl -s http://127.0.0.1:18180/health
curl -s http://127.0.0.1:18180/api/v1/stats
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3,4,5
```

注意：`/health` 里的 `mineru_vlm_base_url=http://127.0.0.1:18100` 可能显示 unavailable。当前生产研究路径使用 direct VLM workers，不依赖 18100 的 MinerU router，所以该字段不是 direct-worker 模式是否可用的判断依据。

## 5. 解析与导出命令

同步解析：

```bash
curl -X POST http://127.0.0.1:18180/parse \
  -F "file=@/path/to/input.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "persist=true" \
  -F "run_embedding=false"
```

异步解析：

```bash
curl -X POST http://127.0.0.1:18180/api/v1/jobs \
  -F "file=@/path/to/input.pdf" \
  -F "max_concurrent_vlm_pages=2" \
  -F "run_embedding=false"
```

查询：

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

## 6. 2026-06-21 实测结果

测试机器：6 张 RTX 4090，GPU 0-5。

运行方式：

- `wait=none`：只统计主解析返回，不等待可选 VLM enhancement。
- `wait=all`：等待可选 VLM enhancement 后导出，适合验证 Markdown 最终质量。
- Embedding 不纳入本次测试。

结果文件：

- `reports/paper-bench-20260621.jsonl`
- Markdown 导出目录在运行机上为 `exports/paper-bench-20260621/`，该目录是生成物，不进入 git。

| 论文 | 页数 | wait=none | wait=all | VLM enhancement |
| --- | ---: | ---: | ---: | ---: |
| ResNet | 12 | 0.709s, 16.94 页/s | 2.814s, 4.26 页/s | 3 个可选任务 |
| ATLAS | 64 | 2.172s, 29.46 页/s | 4.475s, 14.30 页/s | 4 个可选任务 |
| Attention Is All You Need | 15 | 1.431s, 10.49 页/s | 3.526s, 4.25 页/s | 2 个可选任务 |
| U-Net | 8 | 0.410s, 19.49 页/s | 0.360s, 22.24 页/s | 0 个任务 |

结论：

- ATLAS 64 页在可编辑快路径下达到 29.46 页/s，超过“26 页/s”的页速目标。
- 短文档的页/s 会被固定开销影响，不能直接按页数线性外推。
- `wait=all` 变慢是因为等待少量 VLM crop enhancement。它提高最终 Markdown 质量，但不应作为纯解析吞吐的唯一指标。
- 6 卡不能直接加速纯可编辑 PDF 的 PyMuPDF 抽取；6 卡加速的是 VLM 队列。
- 当前优化后的典型论文只产生 0-4 个 VLM 任务，所以 GPU 常驻但不满载。这是低延迟路由优化的结果，不是 GPU 没启好。

## 7. Markdown 与图片核验

已核验 4 篇论文：

| 论文 | 图片 assets | 表格编号 | 图片/图号编号 | 关键内容 |
| --- | ---: | --- | --- | --- |
| ResNet | 6 | Table 1-14 | Figure 1-7 | `ResNet-50`, `22.85`, `6.71` 均存在 |
| ATLAS | 12 | Table 1-19 | Figure 1-11 | `ATLAS`, `13 TeV`, `95%`, `CL` 均存在 |
| Attention | 6 | Table 1-4 | Figure 1-5 | `Attention Is All You Need`, `BLEU`, `Transformer` 均存在 |
| U-Net | 12 | Table 1-2 | Figure 1-3 | `U-Net`, `biomedical`, `overlap-tile`, `ISBI` 均存在 |

校验结论：

- Markdown 图片链接均能对应到本地 assets。
- 未发现 0 字节图片。
- 未发现已知异常串 `Figure 34. F`。
- U-Net 的 Table 1/2 起初只是正文文本，现已在 Markdown exporter 中补了 inline text table 重排，输出为标准 pipe table。

## 8. 关于 PyMuPDF 原生表格识别

单独测试过 `NATIVE_TABLE_EXTRACTION_MODE=auto`：

| 论文 | defer 页/s | auto 页/s | auto 效果 |
| --- | ---: | ---: | --- |
| ResNet | 19.56 | 8.13 | 可识别 14 个 native table，但明显变慢 |
| Attention | 12.61 | 9.09 | 可识别部分 native table |
| U-Net | 34.48 | 15.77 | 仍识别不到无边框文本流表格 |
| ATLAS | 30.76 | 23.16 | 速度下降 |

生产建议：

- 不全局打开 `auto`。
- 默认继续用 `defer + complex queue`。
- 对明确表格密集、结构化要求高的任务，可以单独开“高质量模式”或只对表格页启用原生表格识别。
- U-Net 这类无边框文本流表格，靠 PyMuPDF `find_tables()` 不可靠，已通过导出阶段轻量重排覆盖一类常见场景。

## 9. GPU 显存释放

停止项目进程：

```bash
pkill -TERM -f '/data/mineru-vlm-lab/scripts/run_direct_vlm_worker.py' || true
pkill -TERM -f 'uvicorn app.main:app --host 0.0.0.0 --port 18180 --app-dir /data/mineru-vlm-lab' || true
```

如果 vLLM EngineCore 变成孤儿进程，继续查并只杀 kdsoft 的 EngineCore：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits -i 0,1,2,3,4,5
ps -fp <pid...>
kill -TERM <kdsoft EngineCore pid...>
```

本次释放后状态：

```text
GPU0: 4 MiB / 24564 MiB
GPU1: 4 MiB / 24564 MiB
GPU2: 4 MiB / 24564 MiB
GPU3: 4 MiB / 24564 MiB
GPU4: 425 MiB / 24564 MiB
GPU5: 1271 MiB / 24564 MiB
```

GPU4/5 的残留来自 root 的 `/usr/local/bin/python` 进程，不属于本项目，本次未处理。

## 10. 已知边界

- 当前 benchmark 未包含 Embedding 和入库总链路，只验证文档解析与可选 VLM enhancement。
- 扫描 PDF、复杂表格密集 PDF、纯图片 PDF 还需要单独批量 benchmark。
- 当前 SQLite 适合实验和单机验证；生产要换成正式任务库/对象存储。
- 多节点时应把 enhancement queue 和对象存储从本地文件系统抽出来。
- 6 卡常驻会占满每卡约 22GB 显存，这是 vLLM KV cache 和模型热启动预期结果。

## 11. 代码提交与仓库发布

本项目应提交代码、配置模板、测试、文档和 benchmark JSONL。

不提交：

- 模型权重：`.cache/`, `.modelscope/`
- 运行日志：`logs/`
- SQLite 数据库：`storage/`, `*.sqlite3*`
- PDF 样本和导出资产：`samples/`, `exports/`, `work/`
- 本机生产环境文件：`production.env`, `embedding.env`

初始化和提交：

```bash
cd /data/mineru-vlm-lab
git init
git add .gitignore README.md app configs docs scripts tests requirements.txt requirements-dev.txt pytest.ini reports/paper-bench-20260621.jsonl mineru.json
git commit -m "Initial MinerU VLM lab deployment"
```

推送到远端仓库：

```bash
git remote add origin <repo-url>
git branch -M main
git push -u origin main
```

当前缺少 `<repo-url>` 时，只能先完成本机 git commit。拿到仓库地址后可直接执行 remote add 和 push。
