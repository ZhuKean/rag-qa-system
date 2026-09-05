"""Observability: structured logging + SQLite persistence + PII redaction.

Three things live here because they're inseparable in practice:

* ``pii.py``     — regex-based PII scrubbing. MUST run *before* a string
                   hits either the logger or the DB.
* ``logger.py``  — JSON-line stdout logger + SQLite side channel.
* ``db.py``      — schema + helpers for the request log table.

Design note: we use the stdlib ``logging`` module rather than loguru so we
keep zero non-essential dependencies. The JSON formatter is ~30 lines.
"""
