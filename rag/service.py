"""Service-layer orchestration.

This module glues the retrieval, prompting, and generation stages together
into a single ``ask()`` call. It owns:

* the request_id (UUID) used to correlate logs, DB rows, and API responses;
* the *refusal* path — when the retriever returns an empty list, we short
  circuit and never invoke the LLM;
* the timing measurements that the observability layer logs.

It deliberately does NOT know about FastAPI. That keeps it usable from the
CLI (for evaluation) and the HTTP layer (for production) without coupling.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

from observability.logger import log_event
from observability.pii import redact_pii
from rag.generator import ChatResult, generate
from rag.prompts import Message, build_messages
from rag.retriever import retrieve

logger = logging.getLogger("service")


@dataclass
class Citation:
    """Citation as it appears in the API response (text-light)."""

    chunk_id: str
    source_file: str
    page: int | None
    distance: float


@dataclass
class AskResult:
    """What ``ask()`` returns; also the JSON shape of the /ask response."""

    request_id: str
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    retrieval_count: int = 0
    latency_ms: float = 0.0
    model: str = ""
    # All retrieved chunk_ids in prompt order. The generator maps [n] to
    # docs[n-1], so eval tooling needs the FULL list (not just the cited
    # subset) to validate citation indices.
    retrieved_ids: list[str] = field(default_factory=list)


REFUSAL_REPLY = (
    "I cannot find this in the knowledge base. "
    "Could you rephrase, or ask about a different topic?"
)


def _to_citations(docs) -> list[Citation]:
    return [
        Citation(
            chunk_id=h.chunk.chunk_id,
            source_file=h.chunk.source_file,
            page=h.chunk.page,
            distance=h.distance,
        )
        for h in docs
    ]


def ask(
    question: str,
    *,
    history: list[Message] | None = None,
    session_id: str | None = None,
) -> AskResult:
    """End-to-end QA call.

    Pipeline:
        1. Allocate request_id and start the clock.
        2. Retrieve; if zero relevant hits, return the canned refusal.
        3. Build messages (system + history + question).
        4. Generate the reply.
        5. Log a structured event for observability.

    The refusal path is intentional: it prevents the LLM from hallucinating
    an answer when the corpus has nothing useful to say. This is cheaper
    AND safer than letting the LLM improvise.
    """
    request_id = uuid.uuid4().hex
    started = time.monotonic()

    log_event(
        "ask.start",
        request_id=request_id,
        session_id=session_id,
        question=redact_pii(question),
        history_len=len(history or []),
    )

    # Input guard: an empty (or whitespace-only) question must be refused,
    # not crash the retriever — Chroma rejects empty query strings.
    if not question or not question.strip():
        latency_ms = (time.monotonic() - started) * 1000
        log_event(
            "ask.refuse",
            request_id=request_id,
            session_id=session_id,
            reason="empty_question",
            retrieval_count=0,
            retrieval_ms=0.0,
            latency_ms=latency_ms,
        )
        return AskResult(
            request_id=request_id,
            question=question,
            answer=REFUSAL_REPLY,
            refused=True,
            retrieval_count=0,
            latency_ms=latency_ms,
            retrieved_ids=[],
        )

    t0 = time.monotonic()
    docs = retrieve(question)
    retrieval_ms = (time.monotonic() - t0) * 1000

    if not docs:
        latency_ms = (time.monotonic() - started) * 1000
        log_event(
            "ask.refuse",
            request_id=request_id,
            session_id=session_id,
            reason="no_relevant_docs",
            retrieval_count=0,
            retrieval_ms=retrieval_ms,
            latency_ms=latency_ms,
        )
        return AskResult(
            request_id=request_id,
            question=question,
            answer=REFUSAL_REPLY,
            refused=True,
            retrieval_count=0,
            latency_ms=latency_ms,
            retrieved_ids=[],
        )

    messages = build_messages(question, docs, history=history)

    t0 = time.monotonic()
    chat: ChatResult = generate(messages, docs)
    generation_ms = (time.monotonic() - t0) * 1000

    latency_ms = (time.monotonic() - started) * 1000
    log_event(
        "ask.complete",
        request_id=request_id,
        session_id=session_id,
        retrieval_count=len(docs),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        latency_ms=latency_ms,
        refused=chat.refused,
        model=chat.model,
        answer_chars=len(chat.answer),
        answer=chat.answer,
    )
    return AskResult(
        request_id=request_id,
        question=question,
        answer=chat.answer,
        citations=_to_citations(chat.citations),
        refused=chat.refused,
        retrieval_count=len(docs),
        latency_ms=latency_ms,
        model=chat.model,
        retrieved_ids=[h.chunk.chunk_id for h in docs],
    )


def to_dict(result: AskResult) -> dict:
    """JSON-friendly serialisation, including PII redaction on question/answer."""
    return {
        "request_id": result.request_id,
        "session_id": None,  # filled in by the API layer
        "question": redact_pii(result.question),
        "answer": redact_pii(result.answer),
        "refused": result.refused,
        "retrieval_count": result.retrieval_count,
        "latency_ms": round(result.latency_ms, 1),
        "model": result.model,
        "citations": [asdict(c) for c in result.citations],
    }
