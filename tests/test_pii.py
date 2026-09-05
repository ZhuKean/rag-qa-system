"""Tests for PII redaction."""
from __future__ import annotations

from observability.pii import contains_pii, redact_pii


def test_redact_phone():
    out = redact_pii("call me at 13812345678")
    assert "13812345678" not in out
    assert "[REDACTED_PHONE]" in out


def test_redact_email():
    out = redact_pii("contact alice@example.com today")
    assert "alice@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_id_card():
    out = redact_pii("ID 11010119900307888X here")
    assert "11010119900307888X" not in out
    assert "[REDACTED_ID_CARD]" in out


def test_redact_id_card_all_numeric():
    # Regression: an 18-digit all-numeric PRC ID card (the common case
    # — most cards do NOT end with X) must be labelled ID_CARD, not
    # BANK_CARD, even though the 18-digit run matches both regexes.
    out = redact_pii("ID 110101199003078888 here")
    assert "110101199003078888" not in out
    assert "[REDACTED_ID_CARD]" in out
    assert "[REDACTED_BANK_CARD]" not in out


def test_redact_bank_card():
    out = redact_pii("card 6222021234567890123 end")
    assert "6222021234567890123" not in out
    assert "[REDACTED_BANK_CARD]" in out


def test_idempotent():
    once = redact_pii("phone 13812345678")
    twice = redact_pii(once)
    assert once == twice


def test_passthrough_for_safe_text():
    text = "annual leave is 15 days per year"
    assert redact_pii(text) == text
    assert contains_pii(text) is False


def test_empty_and_none():
    assert redact_pii("") == ""
    assert redact_pii(None) == ""


def test_contains_pii_true():
    assert contains_pii("13812345678") is True
    assert contains_pii("foo@bar.com") is True


def test_multiple_pii_in_one_string():
    out = redact_pii("user 13812345678 / alice@example.com / 11010119900307888X")
    assert "13812345678" not in out
    assert "alice@example.com" not in out
    assert "11010119900307888X" not in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_ID_CARD]" in out
