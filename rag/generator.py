"""LLM generation via an OpenAI-compatible endpoint.

Supports Ollama (local), DeepSeek, OpenAI, or any service that mimics the
``/v1/chat/completions`` schema. Switching providers is a config-only
operation — change ``LLM_BASE_URL`` and ``LLM_MODEL`` in ``.env``.

The generator does four things:

1. Translate our ``Message`` dataclasses into the dict schema the SDK wants.
2. Call ``chat.completions.create`` with our temperature / model settings.
3. Extract ``[n]`` citations from the reply text.
4. Map each cited index back to the originating ``RetrievedChunk`` so the
   API can render them as sources.

The OpenAI client is wrapped in ``lru_cache`` so we reuse the underlying
HTTP connection pool across requests — instantiating ``OpenAI(...)`` is
cheap but creates a new ``httpx.Client`` each time.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from app.config import Settings, get_settings
from rag.models import RetrievedChunk
from rag.prompts import Message, citation_pattern

logger = logging.getLogger("generator")

_REFUSAL_PHRASE = "I cannot answer this question based on the available documents"


@dataclass
class ChatResult:
    """Structured reply from the LLM.

    `answer`  : the assistant's reply text, with ``[n]`` citations inline.
    `citations`: the unique RetrievedChunks referenced in `answer`,
                 in the order they first appear.
    `model`   : the actual model id that produced the reply (handy when the
                 config has a default and the SDK reports the resolved name).
    """

    answer: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    refused: bool = False


@lru_cache(maxsize=1)
def _get_client():
    """Return a cached OpenAI client wired to the configured endpoint.

    `api_key="ollama"` is a placeholder Ollama accepts but doesn't check.
    Real providers (DeepSeek / OpenAI) use the actual key from `.env`.

    Remote endpoints with a placeholder key are rejected up front: sending
    "ollama" to a hosted provider produces a confusing 401 deep inside the
    first real request.
    """
    from openai import OpenAI

    settings = get_settings()
    if settings.llm_base_url is None:
        raise RuntimeError(
            "LLM_BASE_URL is not configured. Set it in .env "
            "(e.g. http://localhost:11434/v1 for local Ollama)."
        )
    api_key = settings.llm_api_key or "missing"
    is_local = any(
        h in settings.llm_base_url
        for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
    )
    if not is_local and (api_key.lower() in {"ollama", "missing", "changeme"} or len(api_key) < 20):
        raise RuntimeError(
            f"LLM_API_KEY is a placeholder ({api_key[:4]}...) but "
            f"LLM_BASE_URL is a remote endpoint ({settings.llm_base_url}). "
            "Set the real key in .env or export LLM_API_KEY in this shell. "
            "Run `python -m scripts.preflight` for a full diagnosis."
        )
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=api_key,
    )


def _to_sdk_messages(messages: Sequence[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _extract_citations(
    answer: str, docs: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Parse ``[n]`` markers in `answer` and map them to RetrievedChunks.

    Out-of-range indices are silently dropped (the LLM occasionally
    hallucinates ``[9]`` when only 5 docs were provided). Duplicates are
    removed while preserving first-occurrence order.
    """
    seen: set[int] = set()
    ordered: list[RetrievedChunk] = []
    pattern = re.compile(citation_pattern())
    for match in pattern.finditer(answer):
        idx = int(match.group(1)) - 1  # [n] -> 0-indexed
        if 0 <= idx < len(docs) and idx not in seen:
            seen.add(idx)
            ordered.append(docs[idx])
    return ordered


def generate(
    messages: Sequence[Message],
    docs: Sequence[RetrievedChunk],
    *,
    settings: Settings | None = None,
) -> ChatResult:
    """Send the chat to the LLM and return a structured ChatResult.

    Refusal detection: if the answer contains the canonical refusal
    phrase from the system prompt, we set ``refused=True`` and return an
    empty citation list. This lets the service layer decide whether to
    surface a special response or simply echo back the refusal.
    """
    settings = settings or get_settings()
    client = _get_client()

    completion = client.chat.completions.create(
        model=settings.llm_model,
        messages=_to_sdk_messages(messages),  # type: ignore[arg-type]
        temperature=settings.temperature,
    )
    answer = (completion.choices[0].message.content or "").strip()
    model_name = getattr(completion, "model", settings.llm_model)

    refused = _REFUSAL_PHRASE in answer
    citations = [] if refused else _extract_citations(answer, docs)
    return ChatResult(answer=answer, citations=citations, model=model_name, refused=refused)
