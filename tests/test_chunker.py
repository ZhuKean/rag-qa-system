"""Tests for rag.chunker (chunk_text and build_chunks)."""

from pathlib import Path

import pytest

from rag.chunker import build_chunks, chunk_text, estimate_tokens
from rag.loaders import load_text
from rag.models import Chunk

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# ---------------------------------------------------------------------------
# estimate_tokens (smoke + the CJK bug we previously fixed)
# ---------------------------------------------------------------------------

def test_estimate_tokens_cjk_counts_one_per_char() -> None:
    """CJK characters count as ~1 token each, not collapsed 4-per-token."""
    assert estimate_tokens("你好") == 2
    # "中文测试" = 4 CJK; "abc" = 3 non-CJK → 3 // 4 = 0; total = 4.
    assert estimate_tokens("中文测试abc") == 4


def test_estimate_tokens_english_uses_4_chars_per_token() -> None:
    """English text rounds down at 4 chars per token."""
    assert estimate_tokens("abcdefgh") == 2          # exactly 8/4
    assert estimate_tokens("abcdefg") == 1           # 7 // 4


# ---------------------------------------------------------------------------
# chunk_text — parameter validation
# ---------------------------------------------------------------------------

def test_chunk_text_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", max_tokens=0)


def test_chunk_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", max_tokens=10, overlap_tokens=-1)


def test_chunk_text_rejects_overlap_ge_max() -> None:
    """overlap must be strictly less than max, otherwise chunks would never advance."""
    with pytest.raises(ValueError):
        chunk_text("hello", max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        chunk_text("hello", max_tokens=10, overlap_tokens=20)


# ---------------------------------------------------------------------------
# chunk_text — semantic invariants
# ---------------------------------------------------------------------------

def test_chunk_text_empty_input_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_short_text_becomes_single_chunk() -> None:
    """A few paragraphs that fit well under the budget stay together."""
    text = "第一段。\n\n第二段。\n\n第三段。"
    out = chunk_text(text, max_tokens=100, overlap_tokens=10)
    assert len(out) == 1
    assert "第一段。" in out[0]


def test_chunk_text_respects_max_tokens() -> None:
    """Every produced chunk must fit within the budget (with one safety carve-out)."""
    # Build a long text out of distinct paragraphs so chunking must split.
    paras = [f"这是第{i}段内容，用于测试预算上限是否被遵守。" for i in range(40)]
    text = "\n\n".join(paras)

    out = chunk_text(text, max_tokens=80, overlap_tokens=10)
    assert len(out) > 1, "expected chunking to actually split the input"
    for chunk in out:
        # Hard-split last-resort path is allowed to equal max_tokens but not exceed.
        assert estimate_tokens(chunk) <= 80, (
            f"chunk exceeds budget: {estimate_tokens(chunk)} tokens"
        )


def test_chunk_text_has_overlap_between_adjacent_chunks() -> None:
    """Adjacent chunks must share some content for retrieval continuity."""
    paras = [f"段落编号{i:02d}包含若干中文字符用于凑长度。" for i in range(30)]
    text = "\n\n".join(paras)

    out = chunk_text(text, max_tokens=80, overlap_tokens=15)
    assert len(out) >= 2

    # At least one character from chunk N must appear at the start of chunk N+1.
    overlaps_found = 0
    for prev, nxt in zip(out, out[1:]):
        tail = prev[-30:]               # last few chars of previous chunk
        if tail.strip() and tail.strip()[:5] in nxt:
            overlaps_found += 1
    assert overlaps_found >= 1, "no overlap detected between adjacent chunks"


def test_chunk_text_handles_oversized_paragraph_via_sentence_split() -> None:
    """A single paragraph with no \\n\\n but many sentences must split cleanly."""
    sents = [f"第{i}句内容很长用来撑爆预算。" for i in range(40)]
    big_paragraph = "".join(sents)     # no blank lines → triggers sentence fallback

    out = chunk_text(big_paragraph, max_tokens=40, overlap_tokens=5)
    assert len(out) > 1
    for chunk in out:
        assert estimate_tokens(chunk) <= 40


def test_chunk_text_preserves_content() -> None:
    """No content should be silently lost across the join."""
    paras = [f"段落{i}" for i in range(20)]
    text = "\n\n".join(paras)

    out = chunk_text(text, max_tokens=30, overlap_tokens=5)
    joined = "".join(out)
    for p in paras:
        assert p in joined, f"paragraph {p!r} lost during chunking"


# ---------------------------------------------------------------------------
# build_chunks — structural invariants
# ---------------------------------------------------------------------------

def test_build_chunks_short_text_single_chunk_correct_metadata() -> None:
    """One tiny page → one Chunk, with page=1, chunk_index=1, stable id."""
    out = build_chunks(["短文本。"], source_file="docs/note.txt")
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, Chunk)
    assert c.page == 1
    assert c.chunk_index == 1
    assert c.source_file == "docs/note.txt"
    assert c.tokens == estimate_tokens("短文本。")
    assert c.chunk_id.startswith("docs/note.txt::")
    assert c.chunk_id.endswith("::1")


def test_build_chunks_index_is_global_across_pages() -> None:
    """chunk_index must keep increasing across page boundaries, not reset."""
    page1 = "第一页段落一。\n\n第一页段落二。" * 5
    page2 = "第二页段落。" * 5
    out = build_chunks([page1, page2], source_file="book.pdf")

    assert len(out) >= 2
    indices = [c.chunk_index for c in out]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices), "chunk_index must be unique"
    # Page numbers progress in order, never decrease.
    pages = [c.page for c in out]
    assert pages == sorted(pages)
    assert set(pages) == {1, 2}


def test_build_chunks_id_is_deterministic_for_same_text() -> None:
    """Re-ingesting the same content must produce identical chunk_ids."""
    a = build_chunks(["稳定的文本内容。" * 3], source_file="x.txt")
    b = build_chunks(["稳定的文本内容。" * 3], source_file="x.txt")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_build_chunks_runs_against_real_handbook() -> None:
    """End-to-end: load the real handbook.txt and build chunks without error."""
    pages = load_text(str(RAW_DIR / "handbook.txt"))
    chunks = build_chunks(pages, source_file="handbook.txt")
    assert len(chunks) >= 1
    assert all(c.tokens <= 400 for c in chunks), "default max_tokens=400 violated"
    assert all(c.page == 1 for c in chunks), "txt has only one page"