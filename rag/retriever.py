"""Retriever: query → embed → top_k → threshold-filter.

The retriever is the single place that enforces *retrieval quality* before
the generator ever runs. If the top hit's cosine distance exceeds
``retrieval_distance_threshold``, we return an empty list — which the
service layer turns into a polite "I don't have that information" reply
instead of forcing the LLM to hallucinate.

The threshold is the only tunable parameter; everything else is fixed by
the embedding model and vector store. Tuning is part of Lesson 11's eval.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from app.config import Settings, get_settings
from rag.models import RetrievedChunk
from rag import vector_store

# Optional reranker: only loaded if enabled in Settings. sentence-transformers'
# `CrossEncoder` would pull a ~1GB model, so we make it strictly opt-in and
# lazy. If the dependency or weights are missing, we degrade silently.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_reranker():
    """Lazy-load the cross-encoder. Returns None if unavailable."""
    from sentence_transformers import CrossEncoder  # heavy import, deferred

    return CrossEncoder(RERANKER_MODEL)


def _rerank(query: str, hits: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-score with a cross-encoder; reorder by descending relevance.

    The cross-encoder takes (query, document) pairs and outputs a relevance
    score in [0, 1]. We use that to sort, and convert back to a
    "cosine distance" by ``1 - score`` so downstream consumers see a
    consistent number — important for threshold logic that doesn't know
    whether a reranker ran.
    """
    reranker = _get_reranker()
    if reranker is None:
        return list(hits)

    pairs = [(query, h.chunk.text) for h in hits]
    scores = reranker.predict(pairs, show_progress_bar=False)
    reranked = [
        RetrievedChunk(chunk=h.chunk, distance=1.0 - float(s))
        for h, s in zip(hits, scores)
    ]
    reranked.sort(key=lambda r: r.distance)
    return reranked


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return relevant chunks for the query, dropping anything past the threshold.

    Empty list is a *signal*, not an error: it means the corpus has nothing
    useful to say. Callers should treat it as "refuse to answer".

    If the reranker is enabled, we over-fetch (top_k * 4 candidates), rerank,
    then return the top ``top_k``. Over-fetching is necessary because the
    reranker's value is precisely in reordering — it sees the same set
    of candidates that vector retrieval surfaced.
    """
    settings = settings or get_settings()
    k = top_k if top_k is not None else settings.top_k

    fetch_k = k * 4 if settings.reranker_enabled else k
    hits = vector_store.query(query, top_k=fetch_k, settings=settings)
    if settings.reranker_enabled:
        hits = _rerank(query, hits)
        hits = hits[:k]

    # Threshold filter — keep only hits strictly BELOW the cutoff.
    # (We use strict `<` rather than `<=` to drop hits *at* the boundary
    # when the user sets the threshold as "the maximum I'll accept".)
    return [h for h in hits if h.distance < settings.retrieval_distance_threshold]
