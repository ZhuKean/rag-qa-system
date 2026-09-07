# Design Notes

A 400-word summary of the **why** behind the RAG QA service. Library
choices and the full library-rejection table live in
[`ARCHITECTURE.md`](ARCHITECTURE.md). Security mechanics (prompt
injection + PII) are detailed in [`SECURITY.md`](SECURITY.md). Open
questions and deliberate omissions are in
[`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md).

## 1. Constraints

Compliance-sensitive internal corpus, bilingual CN/EN, single-CPU
development with GPU headroom, hard targets: Faithfulness >= 0.85,
Context Precision >= 0.70, accuracy >= 80%, p90 <= 10 s. The
**retriever's refusal path is the primary mechanism for keeping
faithfulness high** — no relevant docs in, no answer out.

## 2. Pipeline shape

`load -> chunk -> embed -> store -> retrieve -> prompt -> generate`.
We deliberately do NOT add: LLM-driven semantic chunking (cost +
non-determinism breaks idempotent upsert), cross-encoder reranker by
default (~200 ms + 1 GB model), query rewriting on follow-up (history
concatenation is enough for our use case). Each is a feature flag,
not a missing decision — see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 3. Library choices

See [`ARCHITECTURE.md`](ARCHITECTURE.md). One-liner: open-source,
embedded, single-process, with deterministic chunk IDs that survive
re-ingest.

## 4. Prompt-injection defence

Three concentric layers, all in `rag/prompts.py`: (1) system message
is built from server-controlled strings only, user content goes
*after* it; (2) retrieved docs are wrapped in `<DOC n>...</DOC>` and
the system rules forbid treating that content as instructions; (3)
output is constrained to `[n]` citations that the generator parses
back to chunk IDs. Details in [`SECURITY.md`](SECURITY.md).

## 5. PII handling

`observability/pii.py` regex-redacts phone, ID card, email, bank
card **before** anything reaches the log line. A regression test
guards the order: ID_CARD must run before BANK_CARD so the more
specific marker wins on 18-digit all-numeric IDs. Details in
[`SECURITY.md`](SECURITY.md).

## 6. Refusal is a feature

When the retriever's threshold filter returns `[]`, the service
short-circuits to a canned refusal and **the LLM is never called**.
This trades one extra hop (an LLM call on nonsense queries) for a
large jump in faithfulness on out-of-domain traffic. The threshold
itself is the only tunable knob in `retriever.py`.

## 7. What this design deliberately doesn't do

LLM-driven semantic chunking, cross-encoder reranking by default,
query rewriting, and an external vector DB service. All are
deliberate omissions with one-line upgrade paths if evaluation
shows the cost is worth paying. See
[`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md).

## 8. Open questions

Where the eval will tell us what to add: reranker ROI on > 1k
chunks, p90 latency under DeepSeek vs Ollama, real cost per 1k
calls once we have the production LLM bill, and whether to swap
Chroma for Qdrant at ~10k chunks. See
[`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md).
