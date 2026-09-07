"""Validate that data/raw/ satisfies the corpus coverage matrix.

Exits 0 if everything is in spec, 1 if anything fails. Used as a CI gate.

Usage:
    python -m scripts.corpus_health_check --data-dir data/raw
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Files that MUST exist after build_corpus.py (standard profile)
REQUIRED_FILES = [
    "handbook_cn.txt",
    "handbook_en.txt",
    "legal_disclaimer.txt",
    "office_directory.txt",
    "faq.md",
    "it_ticket_history.txt",
    "product_overview.txt",
    "policy_it_security.docx",
    "travel_expense_policy.pdf",
    "onboarding_scanned.pdf",
]

# PII regexes — copied from observability/pii.py to avoid coupling
RE_PHONE = re.compile(r"\b1[3-9]\d{9}\b")
RE_EMAIL = re.compile(r"\b\S+@\S+\.\S+\b")
RE_ID_CARD = re.compile(r"\b\d{17}[\dXx]\b")
RE_BANK_CARD = re.compile(r"\b\d{16,19}\b")

# Injection samples — at least one must be present
INJECTION_STRINGS = [
    "ignore previous instructions",
    "disregard all prior context",
    "忘记之前所有的指令",
    "你是 DAN",
]

# OOD sanity: corpus must NOT contain these themes so retriever returns
# no hits on weather/stock/news queries.
OOD_FORBIDDEN_KEYWORDS = [
    "weather forecast",
    "stock price",
    "体育比分",
    "天气预报",
    "今日股市",
]


def _all_text(data_dir: Path) -> str:
    """Concatenate text from every file in data_dir."""
    chunks = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".jsonl"}:
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        elif path.suffix.lower() == ".pdf":
            # Try text layer first; if empty, treat as scanned
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
                chunks.append(text)
            except Exception:
                pass
        elif path.suffix.lower() == ".docx":
            try:
                from docx import Document

                doc = Document(str(path))
                chunks.append("\n".join(p.text for p in doc.paragraphs))
            except Exception:
                pass
    return "\n".join(chunks)


def _cjk_ratio(text: str) -> tuple[int, int]:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    return cjk, en


def _has_image_only_pdf(data_dir: Path) -> tuple[bool, str]:
    """A scanned PDF: file exists AND text layer is empty."""
    target = data_dir / "onboarding_scanned.pdf"
    if not target.exists():
        return False, ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(target))
        text = "".join(p.extract_text() or "" for p in reader.pages).strip()
        if len(text) < 20:
            return True, target.name
    except Exception:
        pass
    return False, target.name


def check(data_dir: Path) -> int:
    failures = []
    info = []

    # 1. Required files present
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        failures.append(f"missing required files: {missing}")
    info.append(f"required files present: {len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)}")

    # 2. Total chars — informational only, the corpus is intentionally small
    all_text = _all_text(data_dir)
    total = len(all_text)
    if total < 3_000:
        failures.append(f"total chars {total:,} < 3,000 (corpus too thin)")
    else:
        info.append(f"total chars: {total:,} (informational; sweet spot 5K–40K)")

    # 3. CJK / EN ratio
    cjk, en = _cjk_ratio(all_text)
    ratio_cjk = cjk / max(cjk + en, 1)
    if ratio_cjk < 0.5:
        failures.append(f"CJK ratio {ratio_cjk:.1%} < 50% (corpus should be CN-dominant)")
    else:
        info.append(f"CJK / EN ratio: {cjk:,} / {en:,} ({ratio_cjk:.0%} / {1 - ratio_cjk:.0%})")

    # 4. PII triggers
    phones = RE_PHONE.findall(all_text)
    emails = RE_EMAIL.findall(all_text)
    id_cards = RE_ID_CARD.findall(all_text)
    bank_cards = RE_BANK_CARD.findall(all_text)
    pii_summary = (
        f"phone ×{len(phones)}, email ×{len(emails)}, "
        f"id_card ×{len(id_cards)}, bank_card ×{len(bank_cards)}"
    )
    if len(phones) < 3 or len(emails) < 2 or len(id_cards) < 1 or len(bank_cards) < 1:
        failures.append(f"PII triggers insufficient: {pii_summary}")
    else:
        info.append(f"PII triggers found: {pii_summary}")

    # 5. Prompt-injection strings
    injection_hits = [s for s in INJECTION_STRINGS if s in all_text]
    if not injection_hits:
        failures.append("no prompt-injection sample strings found in corpus")
    else:
        info.append(f"prompt-injection strings found: {len(injection_hits)}/{len(INJECTION_STRINGS)}")

    # 6. OOD sanity
    ood_hits = [s for s in OOD_FORBIDDEN_KEYWORDS if s in all_text]
    if ood_hits:
        failures.append(f"OOD leakage (forbidden keywords present): {ood_hits}")
    else:
        info.append(f"OOD sanity: no weather/stock/news keywords")

    # 7. Cross-doc anchor: faq.md must reference handbook_cn.txt
    faq_path = data_dir / "faq.md"
    if faq_path.exists():
        faq_text = faq_path.read_text(encoding="utf-8")
        # Look for §N.N style anchors or 员工手册 mentions
        has_anchor = bool(
            re.search(r"§\d+\.?\d*", faq_text)
            or "员工手册" in faq_text
        )
        if not has_anchor:
            failures.append("faq.md missing cross-doc anchors (no §N or 《员工手册》 reference)")
        else:
            info.append("faq.md contains cross-doc anchors")

    # 8. Image-only PDF (OCR trigger)
    has_scanned, name = _has_image_only_pdf(data_dir)
    if not has_scanned:
        failures.append("no image-only PDF (onboarding_scanned.pdf missing or has text layer)")
    else:
        info.append(f"image-only PDF present: {name}")

    # 9. DOCX present
    docx_files = list(data_dir.glob("*.docx"))
    if not docx_files:
        failures.append("no .docx file (DOCX loader path not tested)")
    else:
        info.append(f"DOCX present: {docx_files[0].name}")

    # Report
    print("=" * 60)
    for line in info:
        print(f"[OK]   {line}")
    for line in failures:
        print(f"[FAIL] {line}")
    print("=" * 60)

    if failures:
        print(f"\n{len(failures)} failure(s). CI gate would block.")
        return 1
    print("\ncorpus is in spec.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    sys.exit(check(Path(args.data_dir)))


if __name__ == "__main__":
    main()