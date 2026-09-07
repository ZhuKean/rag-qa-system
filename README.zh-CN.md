# RAG QA 系统

[English](README.md) | **简体中文**

一个双语、带引用溯源的 RAG（检索增强生成）问答服务，为初级后端开发工程师
岗位的 take-home 测试而构建。服务基于内部知识库回答自然语言问题，具备：

* **多轮对话**（以 `session_id` 为键的内存会话存储）
* **带引用的答案**（`[1][2]` 标记映射到来源 chunk）
* **语料无关时的硬拒答**（不幻觉、不编造）
* **提示注入防御**（检索内容一律视为数据，不是指令）
* **日志落盘前先做 PII 脱敏**
* **p90 延迟 ≤ 10s** 目标（实测见 `eval/`）
* **Faithfulness ≥ 0.85**、**Context Precision ≥ 0.70**（实测见 `eval/`）

---

## 快速开始

```bash
# 1. 克隆并创建虚拟环境
git clone https://github.com/ZhuKean/rag-qa-system.git
cd rag-qa-system
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置
cp .env.example .env
# 编辑 .env：设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。
# 本地 Ollama 开箱即用；评测时用 DeepSeek。

# 3. 入库语料
mkdir -p data/raw
cp /path/to/your/files/*.{txt,md,docx,pdf} data/raw/
python -m scripts.ingest

# 4. 启动服务
uvicorn app.main:app --reload --port 8000

# 5. 提问（CLI 演示）
python -m scripts.demo                              # 真实 LLM 调用
python -m scripts.demo --mock --question "年假？"   # mock LLM（CI 友好）

# 6. 或直接调 HTTP API
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "员工应遵守哪些基本行为准则？"}' | jq
```

OpenAPI 文档：<http://localhost:8000/docs>。

> 首次运行会从 HuggingFace 下载 bge-m3 嵌入模型（约 2 GB）。国内网络可设置
> `export HF_ENDPOINT=https://hf-mirror.com` 加速。

---

## 架构

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  客户端  │ →  │  FastAPI   │ →  │  rag/      │ →  │  LLM       │
│ (curl,   │    │  /ask      │    │  service   │    │ (OpenAI    │
│  SDK)    │ ←  │  /ingest   │ ←  │            │ ←  │  兼容端点) │
└──────────┘    └────────────┘    └────────────┘    └────────────┘
                                   │      │
                                   ▼      ▼
                              ┌────────────┐  ┌────────────┐
                              │  retriever │  │ generator  │
                              │  → Chroma  │  │  → LLM     │
                              │  → cosine  │  │  → 提取    │
                              │    阈值    │  │    [n]     │
                              └────────────┘  └────────────┘
                                   │
                                   ▼
                              ┌─────────────────────────┐
                              │ observability/          │
                              │  • JSON stdout 日志     │
                              │  • SQLite request_log   │
                              │  • PII 脱敏             │
                              └─────────────────────────┘
```

数据流：

```
data/raw/*.pdf → loaders → pages:list[str]
                              ↓
                     chunker.build_chunks
                              ↓
                     Chunk[] (含 chunk_id, page, tokens)
                              ↓
                     embedder.embed_texts (bge-m3)
                              ↓
                     vector_store.add_chunks (Chroma, upsert)
                              ↓
                     Chroma collection（持久化）
                              ↑
                     retriever.retrieve (top_k + 阈值)
                              ↓
                     prompts.build_messages (system + history + user)
                              ↓
                     generator.generate (LLM)
                              ↓
                     service.ask → AskResult (answer, citations, metrics)
                              ↓
                     observability.log_event (JSON + SQLite，PII 已脱敏)
```

---

## 接口

### `GET /healthcheck`

存活检查 + 集合大小。

```bash
curl http://localhost:8000/healthcheck
# {"status":"ok","vector_count":17,"model":"qwen2.5:7b","embedding_model":"BAAI/bge-m3"}
```

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "u-123",
    "question": "员工应遵守哪些基本行为准则？"
  }'
```

响应：

```json
{
  "request_id": "e5a2…",
  "session_id": "u-123",
  "question": "员工应遵守哪些基本行为准则？",
  "answer": "员工应遵守的基本行为准则包括：熟悉并认同企业文化[1]；树立服务意识[1]；具备责任心[1]；严守公司机密[1]；维护团队荣誉[1]；遵守社会公德[1]。",
  "refused": false,
  "retrieval_count": 5,
  "latency_ms": 1842.5,
  "model": "qwen2.5:7b",
  "citations": [
    {"chunk_id": "handbook.txt::a1b2c3d4::0003", "source_file": "handbook.txt", "page": 1, "distance": 0.18}
  ]
}
```

多轮对话（追问无需重复主题）：

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "u-123",
    "question": "那病假呢？"
  }'
# 服务端从内存会话存储中读取历史。
```

### `POST /ingest`

对 `DATA_DIR` 里的所有文件重跑入库流水线。幂等。

```bash
curl -X POST http://localhost:8000/ingest
# {"status":"ok","file_count":3,"failed_count":0,"manifest_path":"data/manifest.json"}
```

---

## 目录结构

```
rag-qa-system/
├── app/
│   ├── config.py          # 配置（pydantic-settings）
│   └── main.py            # FastAPI 对外接口
├── rag/
│   ├── loaders.py         # txt / md / docx / pdf → pages:list[str]
│   ├── chunker.py         # pages → Chunk[]（token 预算 + 重叠）
│   ├── embedder.py        # bge-m3 封装（懒加载单例）
│   ├── vector_store.py    # Chroma 封装（4 元组契约）
│   ├── retriever.py       # top_k + 阈值 + 可选 rerank
│   ├── prompts.py         # 系统提示词 + DOC 包裹（注入防御）
│   ├── generator.py       # OpenAI 兼容调用 + [n] 提取
│   ├── service.py         # 编排（ask、拒答、可观测性挂钩）
│   └── models.py          # Chunk、RetrievedChunk 数据类
├── observability/
│   ├── pii.py             # 正则脱敏
│   ├── db.py              # SQLite request_log
│   └── logger.py          # JSON stdout + DB 写入
├── scripts/
│   ├── ingest.py          # CLI：遍历 DATA_DIR → 向量库
│   └── demo.py            # 一次性端到端：ingest → /ask → JSON（--mock 供 CI）
├── eval/
│   ├── questions.jsonl    # 12 组手工 Q&A
│   ├── run_eval.py        # ragas + 关键词准确率
│   └── sensitivity.py     # 成本 / 延迟网格
├── tests/                 # pytest 测试套件
├── docs/DESIGN.md         # 设计说明 + 备选方案对比
├── requirements.txt
├── .env.example
└── README.md
```

---

## 评测

完整实测结果与目标达成情况见 [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md)。

```bash
# 端到端指标（调用真实 LLM）
python -m eval.run_eval --output eval/results.csv

# 或用 mock LLM —— 验证链路（检索 → 提示词 → 服务
# → 可观测性 → CSV）而不产生真实 API 费用
python -m eval.run_eval --mock --no-ragas --output eval/results_mock.csv

# 成本 / 延迟网格（同样支持 --mock 离线冒烟）
python -m eval.sensitivity --top-k 3,5,8 --reranker false \
  --temperature 0,0.1,0.7 --limit 5 --output eval/sensitivity.csv
```

示例结果（数值取决于所配置的 LLM；以 CSV 为准）。使用内置
`handbook.txt` 语料和本地 Ollama 的 `qwen2.5:7b`，本机真实跑出的结果：

| 指标                | 目标    | 结果    |
|---------------------|--------|--------|
| 关键词准确率        | ≥ 80%  | 92%    |
| Faithfulness (ragas)| ≥ 0.85 | 0.91   |
| Context Precision   | ≥ 0.70 | 0.78   |
| p90 延迟            | ≤ 10 s | 4.2 s  |
| 单次请求成本        | —      | $0.00008 |

（成本按 DeepSeek-chat 参考价计算 —— 输入 `$0.00014/1k`、
输出 `$0.00028/1k` —— 因为 bge-m3 在本地运行，免费。）

mock 运行会在磁盘上留下 `eval/results_mock.csv` 和
`eval/sensitivity_mock.csv` 供检查：

```
$ python -m eval.run_eval --mock --no-ragas --output eval/results_mock.csv
=== SUMMARY ===
             questions: 12
              accuracy: 1.0
               refused: 0
        p50_latency_ms: 24.5
        p90_latency_ms: 10175.2   # 首次调用加载 bge-m3 模型
```

```
$ python -m eval.sensitivity --mock --top-k 3,5 --reranker false \
        --temperature 0,0.7 --limit 2 --output eval/sensitivity_mock.csv
=== SENSITIVITY ===
top_k  rerank   temp    acc    p50ms    p90ms   in_tok  out_tok        cost
    3   False   0.00   1.00     25.5     25.5     1200        9  $ 0.000171
    3   False   0.70   1.00     23.2     23.2     1200        9  $ 0.000171
    5   False   0.00   1.00     23.6     23.6     1600        9  $ 0.000227
    5   False   0.70   1.00     23.2     23.2     1600        9  $ 0.000227
```

## 综合评测（v2）

12 题的 v1 套件只覆盖双语正常路径。**综合套件**还测试 OOD 拒答、
多轮对话、引用正确性、PII、提示注入和边界情况，详见
[`docs/EVAL_PLAN.md`](docs/EVAL_PLAN.md)，运行方式：

```
$ python -m eval.run_eval_v2 --mock --output eval/results_v2_mock.json
=== SUMMARY ===
  bilingual_cn    pass= 12/12   refused=0    p50=  25.0ms  p90=  25.0ms
  bilingual_en    pass=  5/5    refused=0    p50=  25.0ms  p90=  25.0ms
  ood             pass=  6/6    refused=6    p50=  25.0ms  p90=  25.0ms
  multi_turn      pass=  3/3    refused=0    p50=  25.0ms  p90=  25.0ms
  citation        pass=  2/2    refused=0    p50=  25.0ms  p90=  25.0ms
  pii             pass=  3/3    refused=0    p50=  25.0ms  p90=  25.0ms
  injection       pass=  3/3    refused=3    p50=  25.0ms  p90=  25.0ms
  edge_case       pass=  2/2    refused=1    p50=  25.0ms  p90=  25.0ms
  TOTAL pass=36/36
```

v2 运行器在结果相对 `eval/baseline.json` 出现回归时以非零码退出，
可直接接入 CI。去掉 `--mock` 即对真实 LLM 端点运行。

---

## 设计要点

完整设计说明（含"备选方案对比"：PyMuPDF4LLM、LlamaParse、LangChain
loaders、Unstructured 等，以及各自的取舍理由）见
[`docs/DESIGN.md`](docs/DESIGN.md)。

要点速览：

* **OpenAI 兼容的 LLM 层** —— 改 `.env` 即可切换 Ollama / DeepSeek /
  OpenAI，无需改代码。
* **懒加载 `@lru_cache` 单例**：嵌入模型（3.5 GB）与 OpenAI 客户端
  （HTTP 连接池复用）都是。
* **检索层的 cosine 距离阈值**在调用 LLM *之前* 就拒答域外问题。
* **`<DOC n> ... </DOC>` 包裹 + 系统级规则**：检索文本永远作为数据
  处理，绝不作为指令。
* **PII 脱敏是触碰用户输入的第一道工序** —— 不指望 logger 或数据库
  "事后擦洗"。
* **SQLite 请求日志**兼任评测数据源：每次评测都从磁盘上的真实事件
  读取 ground-truth 指标。

---

## 日志样例（已脱敏 PII）

一小批 `log_event` JSON 行已提交在
[`docs/sample-logs.jsonl`](docs/sample-logs.jsonl)。由
`python scripts/capture_sample_logs.py > docs/sample-logs.jsonl`
生成，覆盖三种场景：

1. 一个干净的问题（无 PII）。
2. 一个包含手机号、18 位身份证、邮箱和 19 位银行卡号的问题 ——
   在日志写入前全部替换为 `[REDACTED_<TYPE>]` 标记。
3. 一个被拒答的域外问题（`refused: true`，
   `reason: "no_relevant_docs"`）。

每行展示了持久化的全部字段（request_id、session_id、检索/生成/总耗时、
拒答标记等），评审者可直接把它当作可观测性结构的实例。

---

## 许可证

MIT，见 `LICENSE`。
