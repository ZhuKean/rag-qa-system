"""Tests for the generator — mocking the OpenAI SDK so we don't hit any network."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from rag.generator import generate
from rag.models import Chunk, RetrievedChunk
from rag.prompts import Message


def _fake_completion(reply_text: str):
    """Build a fake chat.completion object that mimics the OpenAI SDK shape."""
    fake_msg = MagicMock()
    fake_msg.content = reply_text
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]
    fake_completion.model = "fake-model"
    return fake_completion


@contextmanager
def patched_openai(completion_obj):
    """Replace ``rag.generator._get_client`` so generate() uses a fake SDK."""
    client = MagicMock()
    client.chat.completions.create.return_value = completion_obj
    patcher = patch("rag.generator._get_client", return_value=client)
    patcher.start()
    try:
        yield client
    finally:
        patcher.stop()


def _doc(idx: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"f::{idx:04d}::h",
            text=f"document {idx}",
            source_file="f.txt",
            chunk_index=idx,
            tokens=1,
        ),
        distance=0.1 * idx,
    )


def test_generate_returns_answer_text():
    with patched_openai(_fake_completion("Annual leave is 15 days per year [1].")):
        result = generate([Message("user", "q")], [_doc(1)])
    assert result.answer == "Annual leave is 15 days per year [1]."
    assert not result.refused


def test_generate_aligns_citations_to_docs():
    docs = [_doc(1), _doc(2), _doc(3)]
    with patched_openai(_fake_completion("Alpha [2]. Beta [1][3]. Gamma [9].")):
        result = generate([Message("user", "q")], docs)
    # Out-of-range [9] silently dropped; duplicates removed; first-occurrence order.
    assert [d.chunk.chunk_index for d in result.citations] == [2, 1, 3]


def test_generate_marks_refusal():
    with patched_openai(
        _fake_completion(
            "I cannot answer this question based on the available documents."
        )
    ):
        result = generate([Message("user", "q")], [_doc(1)])
    assert result.refused is True
    assert result.citations == []


def test_generate_handles_missing_message_content():
    """Some providers return None for content in edge cases."""
    obj = _fake_completion("")
    obj.choices[0].message.content = None
    with patched_openai(obj):
        result = generate([Message("user", "q")], [_doc(1)])
    assert result.answer == ""
    assert result.refused is False

