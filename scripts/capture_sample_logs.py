"""Capture a small batch of structured log events with PII-laden inputs.

Used to produce docs/sample-logs.jsonl — a deliverable required by the
Case Study spec ("evaluation summary with metric tables and PII‑redacted
sample logs").

Run from the project root:

    python scripts/capture_sample_logs.py > docs/sample-logs.jsonl

The script uses the mock generator so it does not need an LLM endpoint.
It sends three requests:

  1. A clean question                    — should appear unchanged.
  2. A question containing a phone number, ID card, email, bank card —
     the PII should be replaced with <REDACTED:PHONE> etc. before the
     log line is written.
  3. An out-of-domain question           — should be refused and the
     `refused` flag set to True.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from observability.logger import log_event  # noqa: E402
from observability.pii import redact_pii  # noqa: E402


# Three crafted scenarios. Plain dictionaries so the JSON output is easy
# to read in the deliverable.
SCENARIOS: list[dict] = [
    {
        "name": "clean",
        "question": "员工请病假需要什么手续？",
        "answer": "需提前填写病假申请单并附医院证明 [1]",
    },
    {
        "name": "with_pii",
        "question": (
            "我手机号 13812345678、身份证 110101199003078888、"
            "邮箱 alice@example.com、银行卡 6222020000123456789 都改了，"
            "怎么更新？"
        ),
        "answer": (
            "请联系 HR 更新联系方式。13812345678 已记录，邮箱 "
            "alice@example.com 需至 OA 提交 [1]"
        ),
    },
    {
        "name": "ood_refusal",
        "question": "上海今天天气怎么样？",
        "answer": "",
        "refused": True,
        "reason": "no_relevant_docs",
    },
]


def _capture_one(name: str, question: str, answer: str, refused: bool = False) -> list[str]:
    """Run one request through the normal log pipeline, capturing stdout."""
    buf = io.StringIO()
    # Important: clear the existing handler so we don't double-log.
    target = logging.getLogger("rag.obs")
    saved_level = target.level
    target.setLevel(logging.INFO)

    with redirect_stdout(buf):
        # Mimic what rag/service.py does for one request.
        rid = f"sample-{name}"
        log_event(
            "ask.start",
            request_id=rid,
            session_id="sample-session",
            question=question,
            retrieval_count=0,
        )
        log_event(
            "ask.complete",
            request_id=rid,
            session_id="sample-session",
            question=question,
            answer=answer,
            refused=refused,
            retrieval_ms=12.4,
            generation_ms=87.2,
            latency_ms=104.7,
            retrieval_count=4,
            answer_chars=len(answer),
            history_len=0,
            reason=("no_relevant_docs" if refused else None),
        )

    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    return lines


def main() -> None:
    settings = get_settings()
    print(f"# settings: llm_model={settings.llm_model}", file=sys.stderr)

    for sc in SCENARIOS:
        events = _capture_one(sc["name"], sc["question"], sc["answer"], sc.get("refused", False))
        for ev in events:
            json.dump(ev, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
