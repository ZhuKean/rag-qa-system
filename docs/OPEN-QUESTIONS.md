# Open Questions

The eval suite will answer most of these. Each item is paired with
the eval that would tell us the right answer.

## What we deliberately left out

* **LLM-driven semantic chunking.** Each "atomic proposition" costs a
  full LLM call. On a 100-page document that's 50k+ tokens of input
  per pass, repeated for every ingest. Cost and determinism both
  suffer — chunk IDs become unstable, breaking idempotent upserts.
  *Eval that would change this*: a 10% accuracy lift on the eval set
  with semantic chunking at acceptable ingest cost.

* **Cross-encoder reranker by default.** It's behind a feature flag
  (`RERANKER_ENABLED`) because the cross-encoder adds ~200 ms per
  request and 1 GB of model weight. The cost-sensitivity sweep
  (`eval/sensitivity.py`) shows the trade-off explicitly so we can
  decide deliberately.
  *Eval that would change this*: the reranker-off vs reranker-on
  rows of the sensitivity CSV, on a corpus with > 1k chunks.

* **Query rewriting on follow-up.** A simpler "history concatenation"
  strategy gives the LLM enough context to resolve pronouns and
  ellipses. Query rewriting is an easy upgrade if eval shows it's
  needed.
  *Eval that would change this*: multi-turn eval set with
  pronoun-heavy second turns ("what about THAT one?").

* **No streaming.** `chat.completions.create` is non-streaming for
  simplicity. Adding `stream=True` is a small upgrade but it would
  touch the citation parser (we'd have to accumulate tokens before
  extracting `[n]`).
  *Eval that would change this*: p90 latency under load without
  streaming.

* **No auth.** The take-home spec didn't require it; production would
  need an API-key middleware or OIDC.
  *Eval that would change this*: a deployment target.

* **No multi-tenancy.** Single corpus, single session store. Splitting
  corpora per tenant is a configuration change rather than a code one.
  *Eval that would change this*: an SLA for >1 customer.

* **No background ingest.** The `/ingest` endpoint runs synchronously.
  For larger corpora, the right pattern is a worker queue (RQ / Celery
  / Arq) — outside the scope of the take-home.
  *Eval that would change this*: ingest time at 10x current corpus.

* **No Redis session store.** Today the in-memory dict resets on every
  restart; a Redis backend is a 30-line swap.
  *Eval that would change this*: any requirement for multi-replica
  deployments.

## Smaller things worth A/B testing

* **bge-m3 English instruction prefix.** The model card says you can
  prepend `"Represent this sentence for searching relevant passages: "`
  for English queries. Easy A/B; just hasn't been prioritised yet.
* **Embedding field `include=["documents"]` vs not.** Currently
  Chroma returns the documents along with the vectors, but the
  vectors alone are enough to look up the text from our manifest.
  Saves bytes over the wire on large result sets.
* **Client-visible `retrieval_count` and `latency_ms`.** Useful for
  back-office debugging but slightly information-leaky.
