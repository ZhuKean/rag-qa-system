"""Tests for the Chroma wrapper.

End-to-end: each test uses a real PersistentClient pointed at a tmp dir.
We deliberately do NOT mock Chroma — the wrapper's whole purpose is to
glue Chroma's quirky API into our dataclasses, and a mock would hide the
very bugs we want to catch.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.config import get_settings
from rag.models import Chunk, RetrievedChunk
from rag import vector_store


@pytest.fixture
def fresh_chroma(monkeypatch):
    """Point the cached Settings at a tmp chroma dir; clear collection cache.

    Also patches ``_build_embedder`` so we don't need to load the 3.5 GB
    bge-m3 model — the fake returns deterministic unit vectors of the
    correct shape for Chroma.
    """
    import numpy as np
    from unittest.mock import MagicMock, patch

    tmp = Path(tempfile.mkdtemp(prefix="rag_chroma_"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp))

    fake_model = MagicMock(name="SentenceTransformer")

    def fake_encode(texts, **kwargs):  # noqa: ARG001
        # Seed by text content so identical inputs → identical vectors.
        # We still concatenate "42" as a global salt so the seed differs
        # across tests that change the corpus.
        rows = []
        for t in texts:
            seed = (hash(("rag_chroma_fake", "42", t)) & 0xFFFFFFFF) or 1
            rng = np.random.default_rng(seed=seed)
            arr = rng.standard_normal(1024)
            arr /= np.linalg.norm(arr)
            rows.append(arr)
        return np.stack(rows, axis=0)

    fake_model.encode.side_effect = fake_encode

    get_settings.cache_clear()
    from rag.embedder import _build_embedder, get_embedder
    _build_embedder.cache_clear()
    get_embedder.cache_clear()

    with patch("rag.embedder._build_embedder", return_value=fake_model):
        yield tmp

    vector_store.reset_collection_cache()
    get_embedder.cache_clear()
    _build_embedder.cache_clear()
    get_settings.cache_clear()


def _chunk(idx: int, text: str, source: str = "a.txt", page: int | None = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{source}::{idx:04d}::hash",
        text=text,
        source_file=source,
        chunk_index=idx,
        tokens=len(text),
        page=page,
    )


def test_collection_count_starts_at_zero(fresh_chroma):
    assert vector_store.collection_count() == 0


def test_add_chunks_increments_count(fresh_chroma):
    chunks = [_chunk(i, f"text {i}") for i in range(1, 6)]
    vector_store.add_chunks(chunks)
    assert vector_store.collection_count() == 5


def test_add_chunks_is_idempotent_by_id(fresh_chroma):
    """Re-adding the same chunk_ids must not duplicate entries."""
    chunks = [_chunk(1, "hello"), _chunk(2, "world")]
    vector_store.add_chunks(chunks)
    vector_store.add_chunks(chunks)
    assert vector_store.collection_count() == 2


def test_empty_add_chunks_is_noop(fresh_chroma):
    vector_store.add_chunks([])
    assert vector_store.collection_count() == 0


def test_query_returns_results_in_ascending_distance(fresh_chroma):
    """Self-query is the strongest signal: distance should be near zero."""
    chunks = [_chunk(i, f"document number {i} about topic {i}") for i in range(1, 6)]
    vector_store.add_chunks(chunks)
    out = vector_store.query("document number 3 about topic 3", top_k=3)
    assert len(out) == 3
    distances = [r.distance for r in out]
    assert distances == sorted(distances)
    # Top hit should be the document we asked about.
    assert out[0].chunk.chunk_index == 3
    assert out[0].distance < 0.1


def test_query_top_k_caps_results(fresh_chroma):
    chunks = [_chunk(i, f"item {i}") for i in range(1, 11)]
    vector_store.add_chunks(chunks)
    out = vector_store.query("item", top_k=3)
    assert len(out) == 3


def test_query_empty_string_raises(fresh_chroma):
    with pytest.raises(ValueError):
        vector_store.query("", top_k=5)
    with pytest.raises(ValueError):
        vector_store.query("   ", top_k=5)


def test_metadata_roundtrip_preserves_page(fresh_chroma):
    chunk = _chunk(1, "alpha", source="x.pdf", page=7)
    vector_store.add_chunks([chunk])
    out = vector_store.query("alpha", top_k=1)
    assert out[0].chunk.page == 7
    assert out[0].chunk.source_file == "x.pdf"


def test_metadata_handles_missing_page(fresh_chroma):
    """txt files have no page; metadata encodes this as -1, Chunk.page = None."""
    chunk = _chunk(1, "alpha", source="x.txt", page=None)
    vector_store.add_chunks([chunk])
    out = vector_store.query("alpha", top_k=1)
    assert out[0].chunk.page is None
