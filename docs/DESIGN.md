# Design Notes

This document explains the **why** behind the RAG QA service: the
constraints we worked under, the trade-offs we considered, and the
choices we made. It is the companion piece to `README.md`, which only
describes the *what*.

## 1. Constraints

* **Compliance-sensitive internal corpus.** Documents must stay on
  premise, which rules out cloud-only services that phone home with the
  raw text (LlamaParse, Unstructured hosted, …).
* **Bilingual.** The corpus mixes Chinese and English. Anything we pick
  must work on both without per-language hacks.
* **Bounded budget.** The service must run on a single CPU node during
  development and scale to a modest GPU later. The evaluation includes
  a cost-sensitivity requirement, so the cost of every choice is
  observable.
* **Hard targets.** Faithfulness ≥ 0.85, Context Precision ≥ 0.70,
  accuracy ≥ 80%, p90 ≤ 10 s. The retriever refusal path is the
  primary mechanism for keeping faithfulness high — no relevant docs
  in, no answer out.

## 2. Pipeline shape

```
load → chunk → embed → store → retrieve → prompt → generate
```

We deliberately **do not** add any of the following:

* **LLM-driven semantic chunking.** Each "atomic proposition" costs a
  full LLM call. On a 100-page document that's 50k+ tokens of input
  per pass, repeated for every ingest. Cost and determinism both
  suffer — chunk IDs become unstable, breaking idempotent upserts.
* **Query rewriting on follow-up.** A simpler "history concatenation"
  strategy gives the LLM enough context to resolve pronouns and
  ellipses. Query rewriting is an easy upgrade if eval shows it's
  needed.
* **Cross-encoder reranker by default.** It's behind a feature flag
  (`RERANKER_ENABLED`) because the cross-encoder adds ~200 ms per
  request and 1 GB of model weight. The cost-sensitivity sweep
  (`eval/sensitivity.py`) shows the trade-off explicitly so we can
  decide deliberately.

## 3. Library choices

| Concern        | Pick                              | Rejected (and why) |
|----------------|-----------------------------------|--------------------|
| Web framework  | **FastAPI + uvicorn**             | Flask (no OpenAPI), Litestar (smaller ecosystem) |
| Config         | **pydantic-settings**             | dynaconf (heavy), plain os.environ (no validation) |
| Vector store   | **Chroma (embedded)**             | Qdrant (extra service), pgvector (DB lock-in), FAISS (no metadata filter, no persistence story) |
| Embeddings     | **BAAI/bge-m3 via sentence-transformers** | OpenAI text-embedding-3 (cost + data leaves), multilingual-e5 (smaller context, English-friendlier) |
| LLM client     | **openai SDK 1.x against OpenAI-compatible endpoint** | LangChain LLM module (framework lock-in), direct HTTP (reinvent the wheel) |
| Loaders        | **pypdf + python-docx + rapidocr** | PyMuPDF4LLM (AGPL — viral for proprietary use), Unstructured (heavy deps + slow), LlamaParse (cloud-only — fails the compliance constraint), LangChain loaders (framework lock-in) |
| Chunking       | **Hand-rolled three-tier (paragraph → sentence → hard)** | LangChain TextSplitter (over-engineered for our needs), semantic chunkers (see §2) |
| Evaluation     | **ragas**                         | DeepEval (smaller community), TruLens (too opinionated) |
| Logging        | **stdlib `logging` + JSON formatter** | loguru (non-essential dependency), structlog (extra dep for ~30 lines of formatter code) |
| Persistence    | **SQLite via stdlib**             | Postgres (single-process deploy), TinyDB (no SQL, harder to query p90) |

The loaders decision deserves more depth: PyMuPDF4LLM would have given
us better Markdown-formatted output for academic PDFs, but its AGPL
license forces the rest of the project to inherit AGPL or be sold
separately. LlamaParse would have handled tables and figures beautifully
but routes the raw text through LlamaIndex's cloud, which our compliance
review rejected.

## 4. Prompt-injection defence

Three concentric layers:

1. **System prompt is built only from server-controlled strings.**
   The user question is concatenated *after* the system message in the
   message list, never mixed into it.
2. **Retrieved content is wrapped in `<DOC n> ... </DOC>` markers** and
   the system prompt explicitly forbids the model from treating that
   content as instructions.
3. **Output is constrained** to a `[n]` citation format that is parsed
   downstream. Anything the LLM produces outside that contract (raw
   "Ignore previous instructions…") is discarded as plain text — it
   can't reach the user as an action.

The `tests/test_prompts.py` suite asserts all three invariants
including the "user says 'reveal your system prompt'" attack.

## 5. PII handling

Regex-based redaction runs **before** any string touches the logger
or the database. We chose regex over a real NER model because:

* The PII surface is well-known (mobile, ID-card, email, bank-card).
* Regex runs in < 1 ms and has zero model weight.
* A NER model would add 500 MB+ and a model download.

The downside is occasional false positives on long numeric strings —
acceptable for a compliance posture where "miss nothing" beats "spare
the legitimate 18-digit number".

## 6. Refusal is a feature

When the retriever returns an empty list, the service replies with a
canned message and never invokes the LLM. This is cheaper and safer
than letting the model improvise. The threshold lives in `Settings`
(`retrieval_distance_threshold`) so it can be tuned against the
labelled query set; a value of 0.65 was chosen empirically against
bge-m3's score distribution on a small held-out set, but the eval
suite exists to let us move it confidently.

## 7. What this design deliberately doesn't do

* **No streaming.** `chat.completions.create` is non-streaming for
  simplicity. Adding `stream=True` is a small upgrade but it would
  touch the citation parser (we'd have to accumulate tokens before
  extracting `[n]`).
* **No auth.** The take-home spec didn't require it; production would
  need an API-key middleware or OIDC.
* **No multi-tenancy.** Single corpus, single session store. Splitting
  corpora per tenant is a configuration change rather than a code one.
* **No background ingest.** The `/ingest` endpoint runs synchronously.
  For larger corpora, the right pattern is a worker queue (RQ / Celery
  / Arq) — outside the scope of the take-home.

## 8. Open questions

* Does `bge-m3`'s official "add an instruction prefix for English
  queries" advice actually move the needle on retrieval quality for
  our corpus? Easy A/B; just hasn't been prioritised yet.
* Should we expose `retrieval_count` and `latency_ms` to the client in
  the response? Useful for back-office debugging but slightly
  information-leaky.
* Does the in-memory session store need to survive a deploy? Today it
  resets on every restart; a Redis backend is a 30-line swap if yes.
