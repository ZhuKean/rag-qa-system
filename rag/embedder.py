"""Text embedding via sentence-transformers (bge-m3).

Wraps the SentenceTransformer model behind an `@lru_cache`-cached singleton so
that:

  * the heavy model (~3.5 GB on disk, several hundred MB in RAM) is loaded once
    per process, not at import time;
  * FastAPI dependency injection can call ``get_embedder()`` cheaply and
    deterministically across requests.

Cosine similarity between two vectors equals their dot product when the vectors
are L2-normalised, so we always ask for ``normalize_embeddings=True`` and treat
``<a, b>`` as the similarity score.

The ``SentenceTransformer`` import is deferred to inside ``get_embedder()`` so
that simply importing this module doesn't drag in PyTorch (which is the main
reason cold-start of this service used to take several seconds even for
non-embedding requests like ``/healthcheck``).
"""
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

from app.config import get_settings

if TYPE_CHECKING:  # only for static analysers; not at runtime
    from sentence_transformers import SentenceTransformer

# A single embedding is just a Python list of floats; chromadb and most
# vector stores accept that directly.
Embedding = list[float]


@lru_cache(maxsize=1)
def _build_embedder(model_name: str) -> "SentenceTransformer":
    """Construct the actual SentenceTransformer — separated so tests can patch.

    Tests override this with a fake factory that returns a mock object,
    keeping the test runtime free of sentence-transformers (and the several
    hundred MB of native memory that loading it entails).
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_embedder() -> "SentenceTransformer":
    """Return the cached SentenceTransformer instance.

    The model name comes from Settings, so swapping ``EMBEDDING_MODEL`` in
    ``.env`` switches the loaded model on the next fresh process.

    ``maxsize=1`` is defensive: we never expect more than one model per
    process, but explicit ``maxsize`` documents that intent.

    Importing sentence-transformers here (not at module top) keeps the rest
    of the service fast at startup — important for tests and CLI scripts
    that only need loaders / chunker.
    """
    settings = get_settings()
    return _build_embedder(settings.embedding_model)


def embed_texts(
    texts: Sequence[str],
    *,
    normalize: bool = True,
    batch_size: int = 32,
) -> list[Embedding]:
    """Embed a batch of strings into fixed-size L2-normalised vectors.

    Empty input short-circuits without touching the model. The underlying
    ``encode()`` call is one forward pass per batch; pick ``batch_size`` to
    trade memory for throughput (CPU on bge-m3 ≈ 1–2 sentences/sec, so
    ~32 is usually the sweet spot before latency dominates).

    Returns a list of float vectors. Length of each vector equals the model's
    hidden size (1024 for bge-m3). The list is in the same order as ``texts``.
    """
    texts = list(texts)
    if not texts:
        return []

    model = get_embedder()
    vectors = model.encode(
        list(texts),
        normalize_embeddings=normalize,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # vectors is a numpy ndarray (n, dim); convert each row to a plain list.
    return [v.tolist() for v in vectors]
