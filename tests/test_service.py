"""Tests for the service layer — mocks the retriever and generator."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from rag.generator import ChatResult
from rag.models import Chunk, RetrievedChunk
from rag.prompts import Message
from rag import service


def _doc(idx: int, distance: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"f::{idx:04d}::h",
            text=f"doc {idx}",
            source_file="f.txt",
            chunk_index=idx,
            tokens=1,
        ),
        distance=distance,
    )


def test_ask_returns_refusal_when_retriever_empty():
    with patch("rag.service.retrieve", return_value=[]):
        result = service.ask("anything out of domain")
    assert result.refused is True
    assert "cannot find" in result.answer.lower()
    assert result.citations == []
    assert result.retrieval_count == 0


def test_ask_pipes_through_to_generator():
    docs = [_doc(1), _doc(2)]
    fake_chat = ChatResult(
        answer="The policy is X [1].",
        citations=[docs[0]],
        model="fake",
        refused=False,
    )
    with (
        patch("rag.service.retrieve", return_value=docs),
        patch("rag.service.generate", return_value=fake_chat) as gen,
    ):
        result = service.ask("what is the policy?")
    assert result.answer == "The policy is X [1]."
    assert [c.chunk_id for c in result.citations] == ["f::0001::h"]
    gen.assert_called_once()


def test_ask_passes_history_to_prompts():
    """The history parameter must reach build_messages inside generate."""
    docs = [_doc(1)]
    captured = {}

    def fake_generate(messages, _docs, **_kw):
        captured["messages"] = messages
        return ChatResult(answer="ok [1]", citations=[_docs[0]], model="m")

    history = [
        Message("user", "earlier question"),
        Message("assistant", "earlier answer"),
    ]
    with (
        patch("rag.service.retrieve", return_value=docs),
        patch("rag.service.generate", side_effect=fake_generate),
    ):
        service.ask("follow up", history=history)
    roles = [m.role for m in captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_ask_redacts_pii_in_logged_question(caplog):
    """Verify the service redacts PII before logging."""
    import logging
    caplog.set_level(logging.INFO)

    with patch("rag.service.retrieve", return_value=[]):
        service.ask("call me at 13812345678 please")
    # The redacted phone should not appear in any log record.
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "13812345678" not in joined
