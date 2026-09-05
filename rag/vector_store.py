"""Chroma-backed vector store.

A thin wrapper around ``chromadb.PersistentClient`` that:

* exposes a per-process singleton collection (cached, like the embedder);
* enforces the four-tuple contract on every ``add_chunks`` call
  (ids / embeddings / documents / metadatas all the same length);
* translates Chroma's distance metric back into our ``RetrievedChunk`` domain
  type so callers never touch raw chromadb objects.

Why a wrapper rather than using chromadb directly throughout the codebase?

* The wrapper is the *only* place that knows about chromadb. Swapping to
  Qdrant / pgvector later means rewriting this file, not every caller.
* It gives us one place to enforce invariants (length matching, metadata
  schema, upsert semantics) instead of repeating them at every call site.
* It surfaces the `distance` field as a typed number, not a numpy scalar.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import chromadb

from app.config import Settings, get_settings
from rag.embedder import embed_texts
from rag.models import Chunk, RetrievedChunk

COLLECTION_NAME = "rag_qa"
_DISTANCE_METRIC = "cosine"  # smaller distance = more similar


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.api.models.Collection:
    """Return the cached collection (and lazily create the persistent client).

    `PersistentClient` writes to disk, so re-using the same `path` between
    process restarts preserves the collection. Tests can pass a different
    `chroma_dir` through the Settings cache_clear trick below.
    """
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": _DISTANCE_METRIC},
    )


def reset_collection_cache() -> None:
    """Drop the cached collection; useful in tests after swapping chroma_dir."""
    _get_collection.cache_clear()


def add_chunks(chunks: Sequence[Chunk]) -> None:
    """Insert (or upsert) chunks into the vector store.

    Chroma's `add` is upsert-by-id: re-ingesting the same `chunk.chunk_id`
    overwrites the previous entry instead of duplicating it. That makes the
    ingest pipeline idempotent — running `scripts/ingest.py` twice on the
    same corpus yields exactly one entry per chunk.
    """
    if not chunks:
        return

    collection = _get_collection()
    texts = [c.text for c in chunks]
    # Embed all texts in one forward pass — cheaper than per-chunk.
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "page": c.page if c.page is not None else -1,
                "tokens": c.tokens,
            }
            for c in chunks
        ],
    )


def query(
    query_text: str,
    *,
    top_k: int,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Return up to `top_k` RetrievedChunks ranked by ascending cosine distance.

    Empty query strings are rejected at the call site (the API layer does
    this). We do not filter by distance here — filtering is the retriever's
    job because the threshold is a tuning knob owned by the retriever.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be a non-empty string")

    settings = settings or get_settings()
    collection = _get_collection()

    [embedding] = embed_texts([query_text])  # exactly one vector

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    out: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        page_value = meta.get("page", -1)
        page: int | None = page_value if page_value >= 0 else None
        chunk = Chunk(
            chunk_id=cid,
            text=doc,
            source_file=meta["source_file"],
            chunk_index=int(meta["chunk_index"]),
            tokens=int(meta["tokens"]),
            page=page,
        )
        out.append(RetrievedChunk(chunk=chunk, distance=float(dist)))
    return out


def collection_count() -> int:
    """Total number of vectors stored; used by the /health endpoint and tests."""
    return _get_collection().count()
