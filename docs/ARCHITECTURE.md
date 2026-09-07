# Architecture

The pipeline is `load -> chunk -> embed -> store -> retrieve -> prompt -> generate`,
each stage a separate module with a narrow contract so we can swap any one
piece without touching the others. Details on the **library choices** (and
what we rejected) live here; the high-level design summary is in
[`DESIGN.md`](DESIGN.md).

## Stage responsibilities

| Stage       | Module              | Contract                                  | Hot path? |
|-------------|---------------------|-------------------------------------------|-----------|
| load        | `rag.loaders`       | `list[str]` — one entry per page          | one-shot (ingest) |
| chunk       | `rag.chunker`       | `list[str]` -> `list[Chunk]`              | one-shot (ingest) |
| embed       | `rag.embedder`      | `list[str]` -> `list[vector]`             | ingest + every query |
| store       | `rag.vector_store`  | `add_chunks(chunks)` / `query(text, top_k)` | both |
| retrieve    | `rag.retriever`     | `retrieve(q, top_k) -> list[RetrievedChunk]` | every query |
| prompt      | `rag.prompts`       | `build_messages(q, docs, history)`        | every query |
| generate    | `rag.generator`     | `chat(messages) -> ChatResult`            | every query |
| observe     | `observability.*`   | `log_event(name, **fields)`               | every query |

Each stage is **independently testable** — the unit tests in `tests/` cover
one module at a time with no IO beyond the unit under test (vector store
gets a tempdir, LLM gets a mock).

## Library choices (and what we rejected)

| Concern        | Pick                              | Rejected (and why) |
|----------------|-----------------------------------|--------------------|
| Web framework  | **FastAPI + uvicorn**             | Flask (no OpenAPI), Litestar (smaller ecosystem) |
| Config         | **pydantic-settings**             | dynaconf (heavy), plain os.environ (no validation) |
| Vector store   | **Chroma (embedded)**             | Qdrant (extra service), pgvector (DB lock-in), FAISS (no metadata filter, no persistence story) |
| Embeddings     | **BAAI/bge-m3 via sentence-transformers** | OpenAI text-embedding-3 (cost + data leaves), multilingual-e5 (smaller context, English-friendlier) |
| LLM client     | **openai SDK 1.x against OpenAI-compatible endpoint** | LangChain LLM module (framework lock-in), direct HTTP (reinvent the wheel) |
| Loaders        | **pypdf + python-docx + pypdfium2 + rapidocr** | PyMuPDF4LLM (AGPL — viral for proprietary use), Unstructured (heavy deps + slow), LlamaParse (cloud-only — fails the compliance constraint), LangChain loaders (framework lock-in) |
| Chunking       | **Hand-rolled three-tier (paragraph -> sentence -> hard)** | LangChain TextSplitter (over-engineered for our needs), semantic chunkers (cost + determinism) |
| Evaluation     | **ragas**                         | DeepEval (smaller community), TruLens (too opinionated) |
| Logging        | **stdlib `logging` + JSON formatter** | loguru (non-essential dependency), structlog (extra dep for ~30 lines of formatter code) |
| Persistence    | **SQLite via stdlib**             | Postgres (single-process deploy), TinyDB (no SQL, harder to query p90) |

The loaders decision deserves more depth: PyMuPDF4LLM would have given
us better Markdown-formatted output for academic PDFs, but its AGPL
license forces the rest of the project to inherit AGPL or be sold
separately. LlamaParse would have handled tables and figures beautifully
but routes the raw text through LlamaIndex's cloud, which our compliance
review rejected. For **scanned PDFs** (no text layer), we render each
image-only page through pypdfium2 and OCR it with RapidOCR — the fallback
is automatic, gated by a per-page character-count heuristic.

## Cross-cutting choices

* **Lazy singletons via `@lru_cache`** for both the embedder (~3.5 GB) and
  the OpenAI HTTP pool. The first call pays the load cost; every later
  call is fast.
* **Feature flags** (`RERANKER_ENABLED`, `RETRIEVAL_DISTANCE_THRESHOLD`)
  live in `.env` so we can A/B test the sensitivity grid without code
  changes.
* **One direction of dependency**: `app` -> `rag` -> `observability`.
  Nothing in `rag` imports from `app`; nothing in `observability` imports
  from `rag`. The dependency graph is acyclic, which keeps the
  `eval/` and `scripts/` subpackages drop-in replaceable.

## Pipeline shape (one picture)

```
                    (offline)                 (online, every request)
                +-------------+              +-----------------------+
   docs/  --+   | loaders    |   chunks     | retriever  -> docs    |
            +-> | chunker    | ---------->  | generator  -> answer  |
                | embedder   |  vectors     | logs      -> JSON+DB  |
                | vector_st. |              +-----------------------+
                +-------------+
```
