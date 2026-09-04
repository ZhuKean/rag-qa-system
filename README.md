# RAG QA Service

Grounded, citation-backed question answering over an internal knowledge base — built as a case-study project for a Junior Backend Developer assessment.

## Features

- **RAG pipeline** — retrieval-augmented generation over a bilingual (EN / 中文) corpus, including scanned PDFs via OCR
- **Citations** — every answer is grounded in retrieved sources with file/page-level references; answers strictly rely on retrieved context
- **Multi-turn dialogue** — session-scoped conversation continuity
- **Observability** — structured request logging (SQLite) with latency, token usage, and retrieval traces for issue diagnosis
- **Safety** — PII redaction before logging, baseline prompt-injection defenses
- **Evaluation** — RAGAS metrics (Faithfulness, Context Precision), a custom accuracy benchmark, and cost/sensitivity analysis across top_k / reranker / temperature

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, auto OpenAPI docs, industry standard |
| LLM | Any OpenAI-compatible endpoint | Provider-agnostic: Ollama (local) / DeepSeek / OpenAI via one config switch |
| Embeddings | BAAI/bge-m3 | Local, free, strong on Chinese + English |
| Vector store | Chroma | Embedded, zero-ops, pip install |
| Logs & traces | SQLite | Zero-config structured request log |
| Evaluation | RAGAS | Built-in faithfulness / context_precision metrics |
| OCR | RapidOCR (onnxruntime) | Pure pip install, Apple Silicon friendly |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Optional: generate the bilingual demo corpus (incl. a scanned PDF for OCR)
python scripts/make_sample_corpus.py

# Build the vector index (downloads the embedding model on first run)
python scripts/ingest.py

uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

Detailed environment setup (PyCharm): [docs/SETUP.md](docs/SETUP.md)

## Project structure

```
app/           # FastAPI application & settings
rag/           # data pipeline: loaders (PDF/DOCX/OCR), chunker, embedder,
               # vector store (Chroma), ingestion orchestration
scripts/       # CLI entrypoints (ingest, sample-corpus generator)
eval/          # eval set + RAGAS runner                (arriving next phases)
security/      # PII redaction, injection defenses      (arriving next phases)
observability/ # structured logging                     (arriving next phases)
data/raw/      # source documents (bilingual sample corpus included)
tests/         # pytest suite (loaders, chunker, vector store, API)
docs/          # setup & design notes
```

## Configuration

All configuration lives in `.env` (see [.env.example](.env.example)): LLM provider/model, embedding model, top_k, reranker toggle, temperature, storage paths.

## License

MIT
