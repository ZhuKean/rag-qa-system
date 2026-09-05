"""Tests for the retriever's threshold logic.

We mock the underlying vector_store.query so the test runs instantly and
without loading bge-m3.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.config import get_settings
from rag.models import Chunk, RetrievedChunk
from rag import retriever


@pytest.fixture
def fake_query(monkeypatch):
    """Patch vector_store.query to return whatever the test wants."""
    def _set(return_value):
        monkeypatch.setattr(
            "rag.retriever.vector_store.query",
            lambda q, *, top_k, settings=None: return_value,
        )
    return _set


def _hit(idx: int, distance: float, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"f::{idx:04d}::hash",
            text=text,
            source_file="f.txt",
            chunk_index=idx,
            tokens=1,
        ),
        distance=distance,
    )


def test_threshold_drops_high_distance_hits(fake_query, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.5")
    get_settings.cache_clear()
    fake_query([_hit(1, 0.1), _hit(2, 0.4), _hit(3, 0.6)])
    out = retriever.retrieve("anything")
    assert [h.chunk.chunk_index for h in out] == [1, 2]
    get_settings.cache_clear()


def test_all_outside_threshold_returns_empty(fake_query, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.3")
    get_settings.cache_clear()
    fake_query([_hit(1, 0.5), _hit(2, 0.8)])
    out = retriever.retrieve("anything")
    assert out == []
    get_settings.cache_clear()


def test_reranker_overfetches_then_truncates(fake_query, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("TOP_K", "2")
    get_settings.cache_clear()

    # Stub the reranker so we don't need a real cross-encoder.
    monkeypatch.setattr(
        "rag.retriever._rerank",
        lambda q, hits: sorted(hits, key=lambda h: h.distance)[:2],
    )

    # top_k=2 with reranker → fetch 8 candidates from vector_store.
    fake_query([_hit(i, i * 0.1) for i in range(1, 9)])
    out = retriever.retrieve("anything")
    assert len(out) == 2
    get_settings.cache_clear()
