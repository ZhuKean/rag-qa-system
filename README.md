# RAG QA System

A bilingual, citation-backed RAG (Retrieval-Augmented Generation) QA service
built as a take-home exercise for a Junior Backend Developer role. The
service answers natural-language questions over an internal knowledge base
with:

* **Multi-turn dialogue** (in-memory session store keyed by `session_id`)
* **Citation-backed answers** (`[1][2]` markers mapped to source chunks)
* **Hard refusal** when the corpus has nothing relevant (no hallucination)
* **Prompt-injection defence** (retrieved docs treated as data, not instructions)
* **PII redaction** before logs are written
* **p90 latency ≤ 10s** target (measured, see `eval/`)
* **Faithfulness ≥ 0.85**, **Context Precision ≥ 0.70** (measured, see `eval/`)

---

## Quick start

```bash
# 1. Clone and set up the venv
git clone https://github.com/ZhuKean/rag-qa-system.git
cd rag-qa-system
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. Configure
cp .env.example .env
# Edit .env: set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
# Local Ollama works out of the box; DeepSeek for evaluation.

# 3. Ingest a corpus
mkdir -p data/raw
cp /path/to/your/files/*.{txt,md,docx,pdf} data/raw/
python -m scripts.ingest

# 4. Run the service
uvicorn app.main:app --reload --port 8000

# 5. Ask a question
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "员工应遵守哪些基本行为准则？"}' | jq
```

OpenAPI docs at <http://localhost:8000/docs>.

---

## Architecture

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Client  │ →  │  FastAPI   │ →  │  rag/      │ →  │  LLM       │
│  (curl,  │    │  /ask      │    │  service   │    │  (OpenAI-  │
│   SDK)   │ ←  │  /ingest   │ ←  │            │ ←  │   compati- │
└──────────┘    └────────────┘    └────────────┘    │   ble)     │
                                   │      │       └────────────┘
                                   ▼      ▼
                              ┌────────────┐  ┌────────────┐
                              │  retriever │  │ generator  │
                              │  → Chroma  │  │  → LLM     │
                              │  → cosine  │  │  → extract │
                              │  threshold │  │    [n]     │
                              └────────────┘  └────────────┘
                                   │
                                   ▼
                              ┌─────────────────────────┐
                              │ observability/          │
                              │  • JSON stdout log      │
                              │  • SQLite request_log   │
                              │  • PII redaction        │
                              └─────────────────────────┘
```

Data flow:

```
data/raw/*.pdf → loaders → pages:list[str]
                              ↓
                     chunker.build_chunks
                              ↓
                     Chunk[] (with chunk_id, page, tokens)
                              ↓
                     embedder.embed_texts (bge-m3)
                              ↓
                     vector_store.add_chunks (Chroma, upsert)
                              ↓
                     Chroma collection (persistent)
                              ↑
                     retriever.retrieve (top_k + threshold)
                              ↓
                     prompts.build_messages (system + history + user)
                              ↓
                     generator.generate (LLM)
                              ↓
                     service.ask → AskResult (answer, citations, metrics)
                              ↓
                     observability.log_event (JSON + SQLite, PII-redacted)
```

---

## Endpoints

### `GET /healthcheck`

Liveness + collection size.

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

Response:

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

Multi-turn (follow-up without re-stating the subject):

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "u-123",
    "question": "那病假呢？"
  }'
# Server reads session history from the in-memory store.
```

### `POST /ingest`

Re-runs the ingest pipeline against everything in `DATA_DIR`. Idempotent.

```bash
curl -X POST http://localhost:8000/ingest
# {"status":"ok","file_count":3,"failed_count":0,"manifest_path":"data/manifest.json"}
```

---

## Project layout

```
rag-qa-system/
├── app/
│   ├── config.py          # Settings (pydantic-settings)
│   └── main.py            # FastAPI surface
├── rag/
│   ├── loaders.py         # txt / md / docx / pdf → pages:list[str]
│   ├── chunker.py         # pages → Chunk[] (token-budget, overlap)
│   ├── embedder.py        # bge-m3 wrapper (lazy singleton)
│   ├── vector_store.py    # Chroma wrapper (4-tuple contract)
│   ├── retriever.py       # top_k + threshold + optional rerank
│   ├── prompts.py         # system prompt + DOC wrapping (injection defence)
│   ├── generator.py       # OpenAI-compatible call + [n] extraction
│   ├── service.py         # orchestration (ask, refusal, observability hook)
│   └── models.py          # Chunk, RetrievedChunk dataclasses
├── observability/
│   ├── pii.py             # regex redaction
│   ├── db.py              # SQLite request_log
│   └── logger.py          # JSON stdout + DB writer
├── scripts/
│   └── ingest.py          # CLI: walk DATA_DIR → vector store
├── eval/
│   ├── questions.jsonl    # 12 hand-crafted Q&A pairs
│   ├── run_eval.py        # ragas + keyword accuracy
│   └── sensitivity.py     # cost / latency grid
├── tests/                 # pytest suite
├── docs/DESIGN.md         # design notes + alternatives considered
├── requirements.txt
├── .env.example
└── README.md
```

---

## Evaluation

```bash
# End-to-end metrics
python -m eval.run_eval --output eval/results.csv

# Cost / latency grid
python -m eval.sensitivity --top-k 3,5,8 --reranker false \
  --temperature 0,0.1,0.7 --limit 5 --output eval/sensitivity.csv
```

Sample results (numbers depend on the LLM you configure; the CSV is the
source of truth):

| Metric              | Target | Result |
|---------------------|--------|--------|
| Keyword accuracy    | ≥ 80%  | 92%    |
| Faithfulness (ragas)| ≥ 0.85 | 0.91   |
| Context Precision   | ≥ 0.70 | 0.78   |
| p90 latency         | ≤ 10 s | 4.2 s  |
| Cost / request      | —      | $0.00008 |

---

## Design highlights

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design notes including
"Alternatives Considered" (PyMuPDF4LLM, LlamaParse, LangChain loaders,
Unstructured, …) and the reasoning behind each pick.

Quick highlights:

* **OpenAI-compatible LLM layer** — swap Ollama / DeepSeek / OpenAI by
  editing `.env`; no code change.
* **Lazy `@lru_cache` singletons** for both the embedder (3.5 GB) and the
  OpenAI client (HTTP pool reuse).
* **Cosine distance threshold** at the retriever stage refuses
  out-of-domain queries *before* the LLM runs.
* **`<DOC n> ... </DOC>` wrapping + system-level rule** to treat
  retrieved text as data, never as instructions.
* **PII redaction is the *first* thing** that touches user-supplied
  strings — never trust the logger or DB to "scrub later".
* **SQLite request log** doubles as the eval data source: every eval run
  reads its ground-truth metrics from real on-disk events.

---

## License

MIT. See `LICENSE`.
