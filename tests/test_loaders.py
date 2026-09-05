"""Tests for rag.loaders."""

from pathlib import Path

import pytest
from docx import Document

from rag.loaders import load_docx, load_text

# Locate the corpus relative to this test file, not the working directory.
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_load_text_returns_single_page_list() -> None:
    """A .txt file is one logical "page": a single-element list."""
    pages = load_text(str(RAW_DIR / "handbook.txt"))
    assert isinstance(pages, list)
    assert len(pages) == 1
    assert len(pages[0]) > 0  # content is non-empty


def test_load_text_rejects_wrong_suffix() -> None:
    """Wrong file type is refused before the file is even opened."""
    with pytest.raises(ValueError):
        load_text(str(RAW_DIR / "archive.tar.gz"))


def test_load_text_missing_file_raises() -> None:
    """A missing file must raise, never silently return empty content."""
    with pytest.raises(FileNotFoundError):
        load_text(str(RAW_DIR / "no_such_file.txt"))


def test_load_docx_roundtrip() -> None:
    """Create a tiny .docx on the fly, load it back, verify paragraphs survive.

    Builds the temp dir with ``tempfile.mkdtemp`` rather than pytest's
    ``tmp_path`` fixture — the latter triggers a permission-denied in
    some sandboxed runners.
    """
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="rag_loader_"))
    path = d / "note.docx"
    doc = Document()
    doc.add_paragraph("Annual leave is 15 days.")
    doc.add_paragraph("Remote work allowed twice a week.")
    doc.save(path)

    pages = load_docx(str(path))
    assert len(pages) == 1
    assert "Annual leave is 15 days." in pages[0]
    assert "\n\n" in pages[0]  # paragraph boundary preserved for the chunker
