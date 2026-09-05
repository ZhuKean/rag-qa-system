"""Tests for the SQLite request log."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from app.config import get_settings
from observability import db


@pytest.fixture
def fresh_db(monkeypatch):
    """Point the cached Settings at a tmp SQLite path; clear connection cache.

    We build the tmp path ourselves rather than relying on pytest's tmp_path
    fixture, because some sandboxed environments restrict the default
    pytest tmp directory.
    """
    fd, path = tempfile.mkstemp(suffix=".db", prefix="rag_test_")
    os.close(fd)
    monkeypatch.setenv("SQLITE_PATH", path)
    get_settings.cache_clear()
    db.close_all()
    yield Path(path)
    db.close_all()
    get_settings.cache_clear()
    Path(path).unlink(missing_ok=True)


def test_writes_one_row(fresh_db):
    db.write_log(
        request_id="r1",
        event="ask.start",
        question="hello",
    )
    rows = db.query_events("ask.start")
    assert len(rows) == 1
    assert rows[0]["request_id"] == "r1"
    assert rows[0]["event"] == "ask.start"


def test_upsert_replaces_existing_row(fresh_db):
    db.write_log(request_id="r1", event="ask.start", question="a")
    db.write_log(request_id="r1", event="ask.complete", answer="b")
    rows = db.query_events()
    assert len(rows) == 1
    assert rows[0]["event"] == "ask.complete"


def test_query_filters_by_event(fresh_db):
    db.write_log(request_id="r1", event="ask.start")
    db.write_log(request_id="r2", event="ask.complete")
    starts = db.query_events("ask.start")
    assert [r["request_id"] for r in starts] == ["r1"]


def test_meta_round_trips_as_json(fresh_db):
    db.write_log(request_id="r1", event="ask.start", meta={"k": "v"})
    rows = db.query_events()
    import json
    assert json.loads(rows[0]["meta"]) == {"k": "v"}
