"""Tests for rag.embedder — runs without downloading bge-m3.

We patch the private ``_build_embedder`` factory, which is the seam where
``rag.embedder`` actually creates a SentenceTransformer. The fake factory
returns a MagicMock whose ``encode()`` produces deterministic unit-length
vectors, so we can test shape / order / normalization without loading
the 3.5 GB real model.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embedder import _build_embedder, embed_texts, get_embedder

# bge-m3 hidden size — pinned so tests break loudly if the model ever swaps.
_HIDDEN_SIZE = 1024


def _fake_model() -> MagicMock:
    """Return a MagicMock that mimics SentenceTransformer.encode().

    Vectors are deterministic (seeded) and unit-length, so the
    normalization invariant is testable.
    """
    fake = MagicMock(name="SentenceTransformer")

    def fake_encode(texts, **kwargs):  # noqa: ARG001 — kwargs kept for parity
        n = len(list(texts))
        rng = np.random.default_rng(seed=42)
        arr = rng.standard_normal((n, _HIDDEN_SIZE))
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms  # already unit-length, regardless of kwargs

    fake.encode.side_effect = fake_encode
    return fake


@pytest.fixture
def fake_embedder():
    """Patch the embedder factory for the duration of one test."""
    fake = _fake_model()
    get_embedder.cache_clear()
    _build_embedder.cache_clear()
    with patch("rag.embedder._build_embedder", return_value=fake):
        yield fake
    get_embedder.cache_clear()
    _build_embedder.cache_clear()


# ---------------------------------------------------------------------------
# Shape and order
# ---------------------------------------------------------------------------

def test_embed_texts_returns_one_vector_per_input(fake_embedder) -> None:
    out = embed_texts(["hello", "world", "你好世界"])

    assert isinstance(out, list)
    assert len(out) == 3
    for vec in out:
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) == _HIDDEN_SIZE


def test_embed_texts_preserves_input_order(fake_embedder) -> None:
    texts = ["alpha", "beta", "gamma", "delta"]
    out = embed_texts(texts)

    first_components = [v[0] for v in out]
    assert len(set(first_components)) == len(texts), (
        "vectors appear identical — input order may have been lost"
    )


# ---------------------------------------------------------------------------
# Empty input short-circuit
# ---------------------------------------------------------------------------

def test_embed_texts_empty_input_short_circuits(fake_embedder) -> None:
    out = embed_texts([])
    assert out == []
    fake_embedder.encode.assert_not_called()


def test_embed_texts_empty_string_still_calls_model(fake_embedder) -> None:
    """An empty *string* is a legitimate (degenerate) input — let the model decide."""
    embed_texts([""])
    fake_embedder.encode.assert_called_once()


# ---------------------------------------------------------------------------
# L2 normalization invariant
# ---------------------------------------------------------------------------

def test_embed_texts_normalizes_by_default(fake_embedder) -> None:
    out = embed_texts(["hello", "world"])

    for vec in out:
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-5, f"vector not unit-normalized: norm={norm}"


def test_embed_texts_passes_normalize_flag(fake_embedder) -> None:
    embed_texts(["hi"])
    kwargs = fake_embedder.encode.call_args.kwargs
    assert kwargs.get("normalize_embeddings") is True


def test_embed_texts_can_disable_normalization(fake_embedder) -> None:
    embed_texts(["hi"], normalize=False)
    kwargs = fake_embedder.encode.call_args.kwargs
    assert kwargs.get("normalize_embeddings") is False


# ---------------------------------------------------------------------------
# Singleton / caching
# ---------------------------------------------------------------------------

def test_get_embedder_returns_cached_instance(fake_embedder) -> None:
    m1 = get_embedder()
    m2 = get_embedder()
    assert m1 is m2, "get_embedder must return the same cached instance"


def test_get_embedder_calls_factory_only_once(fake_embedder) -> None:
    """The heavy model must not be re-constructed across calls."""
    with patch("rag.embedder._build_embedder", return_value=fake_embedder) as factory:
        get_embedder()
        get_embedder()
        get_embedder()
    factory.assert_called_once()


# ---------------------------------------------------------------------------
# Pluggability via Settings
# ---------------------------------------------------------------------------

def test_embedder_reads_model_name_from_settings(fake_embedder, monkeypatch) -> None:
    """Swapping embedding_model in Settings must propagate to get_embedder()."""
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    fake2 = _fake_model()
    with patch("rag.embedder._build_embedder", return_value=fake2) as factory:
        get_embedder()
    factory.assert_called_once_with("BAAI/bge-small-en-v1.5")
    get_settings.cache_clear()
