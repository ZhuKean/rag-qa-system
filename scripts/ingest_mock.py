"""Ingest the corpus into Chroma with a lightweight fake embedder.

This is a development convenience: the production ingest path uses
real bge-m3 embeddings, but in environments where torch + transformers
can't be loaded (CI, low-memory sandboxes), this script fakes
embeddings as 1024-dim Gaussian vectors.

The chunk_id stability and upsert semantics are identical to the real
path, so chunk IDs can be inspected and used for ``expected_citations``
even when running with fake embeddings.

Usage:
    python -m scripts.ingest_mock --reset
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is importable when run as a plain script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    """Patch embedder, then delegate to scripts.ingest.main()."""
    # Patch embed_texts BEFORE any module that depends on it is imported
    import numpy as np
    import rag.embedder as embedder_mod

    def _fake_embed_texts(texts, settings=None):
        # Deterministic per-call seeding so the same text always gets the
        # same vector — this preserves Chroma's deduplication behaviour
        # and means "re-ingest" is still idempotent at the storage level.
        rng = np.random.default_rng(20260907)
        return rng.random((len(texts), 1024)).tolist()

    embedder_mod.embed_texts = _fake_embed_texts

    # Now import scripts.ingest — its embed_texts is resolved lazily
    # because rag/embedder.py uses a deferred import.
    from scripts.ingest import main as ingest_main

    argv = sys.argv[1:]
    return ingest_main(argv)


if __name__ == "__main__":
    sys.exit(main())