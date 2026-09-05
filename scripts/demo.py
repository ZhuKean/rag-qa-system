"""End-to-end demo: run one full /ask cycle and print the response.

Usage:
    python -m scripts.demo               # real LLM (needs .env configured)
    python -m scripts.demo --mock        # mock LLM (no network/model needed)

This script is meant for quick smoke-testing on a laptop. It loads the
indexed corpus from ``data/chroma`` (run ``python -m scripts.ingest``
first), calls ``rag.service.ask`` once, and prints a friendly version
of the JSON response.

Why a separate script instead of just ``curl``?

* On a fresh dev box, you often want to confirm the pipeline works
  *before* launching the HTTP layer or even starting Ollama.
* The mock flag lets you verify retrieve + prompt + citation alignment
  in CI without paying the LLM cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.service import ask, to_dict  # noqa: E402


def _fake_completion(reply: str) -> MagicMock:
    """Build a stand-in for an OpenAI chat completion."""
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    completion.model = "mock-model"
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo one /ask cycle.")
    parser.add_argument("--mock", action="store_true",
                        help="patch the OpenAI client so no real LLM is called")
    parser.add_argument("--question", default="年假有多少天？",
                        help="the user question (Chinese or English)")
    parser.add_argument("--session", default="demo-session")
    args = parser.parse_args(argv)

    if args.mock:
        # Patch the openai client at the place generator uses it.
        reply = ("根据《员工手册》第 3.2 条，年假为 15 个自然日，须在自然年度内休完 [1]。"
                 "新入职员工按当年剩余日历月份折算。")
        with patch(
            "rag.generator._get_client",
            return_value=MagicMock(
                chat=MagicMock(
                    completions=MagicMock(
                        create=MagicMock(return_value=_fake_completion(reply))
                    )
                )
            ),
        ):
            result = ask(question=args.question, session_id=args.session)
    else:
        result = ask(question=args.question, session_id=args.session)

    print(json.dumps(to_dict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())