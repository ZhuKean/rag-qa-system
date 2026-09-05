"""Text chunking for the RAG pipeline.

Pipeline:
    loader (per page)  →  chunk_text (per page)  →  build_chunks (wraps into Chunk)

Three tiers inside chunk_text:
    1. Paragraph split on blank lines (\n\n), greedily packed up to max_tokens.
    2. Oversized paragraph → sentence split (Chinese 。！？ and English .!?).
    3. Single oversized sentence → hard split by char budget (last-resort safety net).
"""
import hashlib
import re

from .models import Chunk


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a text.

    Heuristic: CJK characters count as ~1 token each;
    all other characters count as roughly 4 chars per token.
    """
    length = len(text)
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cjk + (length - cjk) // 4


# ---------------------------------------------------------------------------
# Helpers used by chunk_text
# ---------------------------------------------------------------------------

# Lookbehind for sentence delimiters; preserves the delimiter on the left side.
_SENTENCE_DELIMS = re.compile(r"(?<=[。！？!?])\s*")


def _split_sentences(text: str) -> list[str]:
    """Split text by Chinese and English sentence terminators.

    Keeps the delimiter attached to the preceding sentence so that, when the
    chunks are concatenated, the original punctuation survives.
    """
    parts = _SENTENCE_DELIMS.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort split: cut by character budget that is provably safe.

    Worst-case budget is 4 chars per token (all-English text). Any slice of
    `max_tokens * 4` characters therefore fits within the token budget for any
    CJK/English mixture.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]
    char_budget = max_tokens * 4
    return [text[i:i + char_budget] for i in range(0, len(text), char_budget)]


def _take_tail(text: str, max_tail_tokens: int) -> str:
    """Return a trailing portion of text up to max_tail_tokens (sentence-aware).

    Used to produce the overlap carried into the next chunk. Walks backwards
    over sentences until the budget is exhausted; falls back to an empty string
    when nothing fits.
    """
    if max_tail_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tail_tokens:
        return text

    sentences = _split_sentences(text)
    out: list[str] = []
    out_tokens = 0
    for sent in reversed(sentences):
        s_tokens = estimate_tokens(sent)
        if out_tokens + s_tokens > max_tail_tokens:
            break
        out.insert(0, sent)
        out_tokens += s_tokens
    return "".join(out) if out else ""


# ---------------------------------------------------------------------------
# chunk_text — public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[str]:
    """Split text into token-bounded chunks with sentence-aware overlap.

    Returns a list of plain-text chunks. Empty or whitespace-only input yields
    an empty list. Parameter validation:

      - max_tokens must be > 0
      - overlap_tokens must be >= 0 and strictly less than max_tokens

    The first chunk never carries an overlap (there is no preceding content);
    subsequent chunks may be prefixed by a tail of up to ``overlap_tokens``
    taken from the previous chunk.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be >= 0, got {overlap_tokens}")
    if overlap_tokens >= max_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})"
        )

    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buffer: list[str] = []          # paragraphs/sentences of the in-flight chunk
    buffer_tokens: int = 0
    overlap_tail: str = ""          # tail of the previous finalized chunk

    # ---- inner closures (capture the surrounding state) --------------------

    def _joined_tokens(*extras: str) -> int:
        """estimate_tokens of the buffer if joined with the given extras."""
        if not buffer and not extras:
            return 0
        return estimate_tokens("\n\n".join(buffer + list(extras)))

    def _finalize_buffer() -> None:
        """Close the in-flight buffer as a chunk; seed the overlap tail."""
        nonlocal buffer, buffer_tokens, overlap_tail
        if not buffer:
            return
        chunk_str = "\n\n".join(buffer).strip()
        if chunk_str:
            chunks.append(chunk_str)
            overlap_tail = _take_tail(chunk_str, overlap_tokens)
        buffer = []
        buffer_tokens = 0

    def _consume(piece: str, piece_tokens: int) -> None:
        """Append a single paragraph/sentence/hard-slice to the buffer.

        Budget is enforced on the *would-be joined* string, not the sum of
        individual piece tokens — because the "\\n\\n" separators themselves
        count as non-CJK characters and can push the total over max_tokens.
        """
        nonlocal buffer, buffer_tokens, overlap_tail

        # Defensive: a single piece that still exceeds the budget (should only
        # happen if a hard split misfired) is emitted on its own.
        if piece_tokens > max_tokens:
            _finalize_buffer()
            chunks.append(piece)
            overlap_tail = _take_tail(piece, overlap_tokens)
            return

        # Empty buffer → optionally seed with overlap, then add the piece.
        if not buffer:
            if overlap_tail:
                cand_tokens = _joined_tokens(overlap_tail, piece)
                if cand_tokens <= max_tokens:
                    buffer.append(overlap_tail)
                    buffer.append(piece)
                    buffer_tokens = cand_tokens
                else:
                    buffer.append(piece)
                    buffer_tokens = piece_tokens
                overlap_tail = ""
            else:
                buffer.append(piece)
                buffer_tokens = piece_tokens
            return

        # Non-empty buffer → check the joined size, not the sum of parts.
        cand_tokens = _joined_tokens(piece)
        if cand_tokens <= max_tokens:
            buffer.append(piece)
            buffer_tokens = cand_tokens
            return

        # Would overflow → close current buffer, then start a fresh one.
        _finalize_buffer()
        if overlap_tail:
            cand2 = _joined_tokens(overlap_tail, piece)
            if cand2 <= max_tokens:
                buffer.append(overlap_tail)
                buffer.append(piece)
                buffer_tokens = cand2
            else:
                buffer.append(piece)
                buffer_tokens = piece_tokens
            overlap_tail = ""
        else:
            buffer.append(piece)
            buffer_tokens = piece_tokens

    # ---- main loop ---------------------------------------------------------

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if para_tokens <= max_tokens:
            _consume(para, para_tokens)
            continue

        # Oversized paragraph: close what we have, then split by sentences.
        _finalize_buffer()
        sentences = _split_sentences(para)
        pieces: list[str] = []
        for sent in sentences:
            if estimate_tokens(sent) <= max_tokens:
                pieces.append(sent)
            else:
                pieces.extend(_hard_split(sent, max_tokens))
        for piece in pieces:
            _consume(piece, estimate_tokens(piece))

    _finalize_buffer()
    return chunks


# ---------------------------------------------------------------------------
# build_chunks — wrap raw strings into Chunk records
# ---------------------------------------------------------------------------

def build_chunks(pages: list[str], source_file: str) -> list[Chunk]:
    """Wrap per-page chunk strings into Chunk records.

    Each Chunk gets:
      - a deterministic chunk_id based on the text content (sha256[:8]) and a
        monotonically increasing chunk_index across the whole document;
      - the 1-based page number on which the chunk originated;
      - its estimated token count (cached for later budgeting).

    Pages are processed in order, and the global chunk_index advances across
    page boundaries so the ordering is stable across re-ingestions of the
    same file.
    """
    out: list[Chunk] = []
    chunk_index = 0
    for page_no, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text):
            chunk_index += 1
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:8]
            out.append(
                Chunk(
                    chunk_id=f"{source_file}::{digest}::{chunk_index}",
                    text=piece,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    tokens=estimate_tokens(piece),
                    page=page_no,
                )
            )
    return out