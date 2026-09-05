"""PII redaction — runs BEFORE anything is logged or persisted.

Why regex instead of a real NER model?
    * The corpus is internal and the PII surface area is well-known:
      phone numbers, ID-card numbers, email, bank-card numbers.
    * Regex is ~1ms per call and has zero model-loading cost.
    * A real NER model (presidio, flair) would add 500MB+ and a model
      download — overkill for the four patterns we actually need.

Order of patterns matters: longer / more specific patterns first, so the
phone-number regex doesn't swallow the first 11 digits of a bank-card
number.
"""
from __future__ import annotations

import re

# Patterns are conservative: we'd rather miss an exotic format than
# accidentally redact a benign string. Each one has a tiny comment
# explaining the intent.

_BANK_CARD = re.compile(r"\b\d{16,19}\b")
# Loose: any 16–19 digit run. False positives are acceptable because
# the alternative is leaking a real card number.

_ID_CARD = re.compile(r"\b\d{17}[\dXx]\b")
# PRC ID card: 18 chars, last may be X. Anchored on word boundaries so
# we don't redact the middle of a long number.

_PHONE = re.compile(r"\b1[3-9]\d{9}\b")
# PRC mobile: starts with 1, second digit 3–9, then 9 more digits.

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


_REPLACEMENTS = (
    # Order matters: more specific patterns must run FIRST so they get
    # the chance to claim a match before a looser pattern steals it.
    # An 18-digit all-numeric PRC ID card matches BOTH _ID_CARD (17
    # digits + 1 digit) AND _BANK_CARD (16–19 digits). If _BANK_CARD
    # ran first it would relabel the ID card as a bank card, which
    # is misleading for downstream audit even though the data itself
    # stays redacted. Id_CARD before Bank_CARD ensures the more
    # specific marker wins for the common (X-less) case.
    (_ID_CARD, "[REDACTED_ID_CARD]"),
    (_BANK_CARD, "[REDACTED_BANK_CARD]"),
    (_PHONE, "[REDACTED_PHONE]"),
    (_EMAIL, "[REDACTED_EMAIL]"),
)


def redact_pii(text: str | None) -> str:
    """Return a copy of `text` with recognised PII replaced by markers.

    None and empty strings pass through unchanged. Redaction is idempotent:
    calling it twice gives the same result as calling it once.
    """
    if not text:
        return text or ""
    for pattern, marker in _REPLACEMENTS:
        text = pattern.sub(marker, text)
    return text


def contains_pii(text: str) -> bool:
    """Quick boolean check used in tests."""
    if not text:
        return False
    return any(p.search(text) for p, _ in _REPLACEMENTS)
