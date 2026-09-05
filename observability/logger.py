"""Structured JSON logging.

A single function ``log_event(name, **fields)`` does three things:

  1. Builds a one-line JSON record with timestamp, level, event name,
     and the caller's fields (with PII already redacted by the caller).
  2. Writes it to stdout (so it lands in container logs / journald).
  3. Persists a corresponding row in the SQLite request log (so we can
     compute p90 latencies without grepping log files).

Why both stdout and SQLite?
    * Stdout is for humans + shipping to log aggregators (Loki, ELK).
    * SQLite is for ad-hoc queries during evaluation: "what was the p90
      generation latency over the last 100 requests?"

The two are independent; one can fail without breaking the other.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from observability import db
from observability.pii import redact_pii

# A dedicated logger so app code doesn't pick up unrelated handlers.
_logger = logging.getLogger("rag.obs")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)


# Fields whose values should never be re-redacted (they're already safe).
_SAFE_FIELDS = frozenset(
    {
        "request_id",
        "session_id",
        "event",
        "ts",
        "latency_ms",
        "retrieval_ms",
        "generation_ms",
        "retrieval_count",
        "refused",
        "model",
        "answer_chars",
        "reason",
        "history_len",
    }
)


def log_event(name: str, **fields: Any) -> None:
    """Emit one structured event.

    Two safety guarantees:
      * String fields with PII-sensitive names ('question', 'answer') are
        redacted automatically. Other string fields are passed through
        unchanged — the caller is expected to have redacted them if needed.
      * The DB write is best-effort: if it fails (disk full, schema drift)
        we still emit to stdout and don't crash the request.
    """
    record: dict[str, Any] = {"ts": time.time(), "event": name}
    record.update(fields)

    # Auto-redact known sensitive string fields.
    for k in ("question", "answer"):
        v = record.get(k)
        if isinstance(v, str):
            record[k] = redact_pii(v)

    line = json.dumps(record, ensure_ascii=False, default=str)
    _logger.info(line)

    try:
        db.write_log(
            request_id=record.get("request_id") or f"no-id-{record['ts']}",
            event=name,
            session_id=record.get("session_id"),
            question=record.get("question"),
            answer=record.get("answer"),
            refused=record.get("refused"),
            retrieval_count=record.get("retrieval_count"),
            retrieval_ms=record.get("retrieval_ms"),
            generation_ms=record.get("generation_ms"),
            latency_ms=record.get("latency_ms"),
            model=record.get("model"),
            ts=record["ts"],
        )
    except Exception as exc:  # pragma: no cover — observability must not break callers
        _logger.warning("db write failed for event %s: %s", name, exc)
