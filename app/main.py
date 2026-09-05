"""FastAPI surface for the RAG service.

Endpoints:

    GET  /healthcheck    — liveness + vector-store count
    POST /ask            — primary QA call (with optional multi-turn history)
    POST /ingest         — admin trigger for the ingest CLI
    GET  /sessions/{id}  — fetch the in-memory conversation history (debug aid)

The session store is in-process and intentionally simple:
    * It survives only as long as the worker process.
    * Production deployments would back this with Redis or similar.
    * We expose it through an endpoint so the front-end / SDK can pull
      history if it ever needs to.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from observability.logger import log_event
from rag.prompts import Message
from rag.service import ask, to_dict

logger = logging.getLogger("app")


# ---- request/response models --------------------------------------------

class HistoryTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str


class AskRequest(BaseModel):
    session_id: str | None = None
    question: str
    history: list[HistoryTurn] | None = None


class AskResponse(BaseModel):
    request_id: str
    session_id: str | None
    question: str
    answer: str
    refused: bool
    retrieval_count: int
    latency_ms: float
    model: str
    citations: list[dict[str, Any]]


class IngestResponse(BaseModel):
    status: str
    file_count: int
    failed_count: int
    manifest_path: str


# ---- session store (in-process) -----------------------------------------

_session_lock = threading.Lock()
_sessions: dict[str, list[Message]] = {}


def _remember_turn(session_id: str, message: Message) -> None:
    with _session_lock:
        _sessions.setdefault(session_id, []).append(message)


def _recall_history(session_id: str) -> list[Message]:
    with _session_lock:
        return list(_sessions.get(session_id, []))


# ---- FastAPI app ---------------------------------------------------------

app = FastAPI(title="RAG QA system", version="0.2.0")


@app.get("/healthcheck")
def healthcheck() -> dict:
    """Liveness probe + vector-store size.

    Hitting Chroma on every healthcheck costs a small SQLite read; that's
    acceptable for /healthcheck which is meant to be called by a load
    balancer infrequently.
    """
    from rag.vector_store import collection_count  # local import to avoid
                                                  # loading chromadb at import
    settings = get_settings()
    return {
        "status": "ok",
        "vector_count": collection_count(),
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(400, "question must be a non-empty string")

    history: list[Message] = []
    if req.history:
        history = [Message(role=t.role, content=t.content) for t in req.history]
    elif req.session_id:
        history = _recall_history(req.session_id)

    result = ask(question=req.question, history=history, session_id=req.session_id)

    if req.session_id:
        _remember_turn(req.session_id, Message(role="user", content=req.question))
        _remember_turn(req.session_id, Message(role="assistant", content=result.answer))

    payload = to_dict(result)
    payload["session_id"] = req.session_id
    return AskResponse(**payload)


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint() -> IngestResponse:
    """Run the ingest pipeline synchronously.

    We could offload to a background task, but for simplicity we run it
    inline: typical corpora are dozens of files and finish in seconds.
    For real production, swap to a queue / worker.
    """
    from scripts.ingest import main as ingest_main

    # scripts.ingest.main parses argv from sys.argv; we override with [].
    exit_code = ingest_main([])
    settings = get_settings()
    manifest = Path(settings.data_dir).parent / "manifest.json"

    if exit_code != 0:
        raise HTTPException(500, "ingest failed; check logs")

    import json

    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    return IngestResponse(
        status="ok",
        file_count=data.get("file_count", 0),
        failed_count=data.get("failed_count", 0),
        manifest_path=str(manifest),
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """Debug aid: read back the in-memory history for a session."""
    with _session_lock:
        history = _sessions.get(session_id, [])
    return {
        "session_id": session_id,
        "turn_count": len(history),
        "history": [{"role": m.role, "content": m.content} for m in history],
    }
