"""Ingest the corpus into Chroma.

Usage:
    python -m scripts.ingest                # ingest everything under DATA_DIR
    python -m scripts.ingest --reset        # wipe the collection first

What it does, in order:

1. Walk ``DATA_DIR`` for files matching the loader's accepted suffixes.
2. For each file: load → chunk → embed → upsert into Chroma.
3. Write ``DATA_DIR/../manifest.json`` with per-file stats (sha256, count, ts).

The script is idempotent: re-running it on an unchanged corpus touches zero
records (because ``chunk.chunk_id`` is content-derived and Chroma's
``add`` is upsert-by-id).

Failure semantics:
    A single failed file is logged and the run continues. The exit code is
    non-zero if any file failed, so CI / cron can detect partial ingests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_settings
from rag import vector_store
from rag.chunker import build_chunks
from rag.loaders import load_docx, load_pdf, load_text

logger = logging.getLogger("ingest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# Dispatch table: suffix -> loader.
# Kept here (not in rag.loaders) so the ingest CLI owns the policy of
# which extensions it will walk. rag.loaders stays a pure library.
LOADERS = {
    ".txt": load_text,
    ".md": load_text,
    ".docx": load_docx,
    ".pdf": load_pdf,
}


@dataclass
class FileManifest:
    source_file: str
    sha256: str
    chunk_count: int
    ingested_at: float
    elapsed_seconds: float


def _iter_files(data_dir: Path) -> list[Path]:
    """Return all files in data_dir whose suffix has a loader.

    Sorted by name for deterministic output.
    """
    return sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in LOADERS
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_one(path: Path) -> FileManifest | None:
    """Run loader → chunker → vector_store for one file. Returns the manifest.

    Returns None on loader failure (logged, not raised) so the CLI can keep
    going for the remaining files.
    """
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:  # unreachable given _iter_files filter, but defensive
        return None

    started = time.monotonic()
    try:
        pages = loader(str(path))
    except Exception as exc:
        logger.error("loader failed for %s: %s", path.name, exc)
        return None

    chunks = build_chunks(pages, source_file=path.name)
    if not chunks:
        logger.warning("%s produced 0 chunks — skipping", path.name)
        return None

    vector_store.add_chunks(chunks)
    elapsed = time.monotonic() - started
    logger.info(
        "ingested %s: %d chunks in %.2fs", path.name, len(chunks), elapsed
    )
    return FileManifest(
        source_file=path.name,
        sha256=_sha256(path),
        chunk_count=len(chunks),
        ingested_at=time.time(),
        elapsed_seconds=elapsed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the corpus into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete every record in the collection before ingesting.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        logger.error("DATA_DIR does not exist: %s", data_dir)
        return 1

    if args.reset:
        logger.info("resetting collection (--reset)")
        # Chroma exposes a delete-by-collection primitive; we use the
        # underlying client API to drop and recreate the collection.
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_dir)
        client.delete_collection(vector_store.COLLECTION_NAME)
        vector_store.reset_collection_cache()
        logger.info("collection dropped; will recreate on next add")

    files = _iter_files(data_dir)
    if not files:
        logger.warning("no ingestable files in %s", data_dir)
        return 0

    manifests: list[FileManifest] = []
    failed = 0
    for path in files:
        m = ingest_one(path)
        if m is None:
            failed += 1
        else:
            manifests.append(m)

    # Manifest lives next to the corpus, not in it (so it doesn't get
    # re-ingested on the next run).
    manifest_path = data_dir.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": time.time(),
                "corpus_dir": str(data_dir),
                "file_count": len(manifests),
                "failed_count": failed,
                "files": [asdict(m) for m in manifests],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote manifest: %s (%d files, %d failed)",
        manifest_path,
        len(manifests),
        failed,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
