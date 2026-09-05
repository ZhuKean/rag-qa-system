"""Run the evaluation suite against the live service.

For every question in ``eval/questions.jsonl`` we:

    1. Call ``service.ask(question)`` (which exercises the real retriever
       and real LLM).
    2. Compute two ragas metrics (Faithfulness, Context Precision) on the
       produced answer/context pair.
    3. Compute a simple substring-based accuracy rule — does the answer
       contain every ``expected_keyword``?
    4. Aggregate everything into a CSV row and a final summary table.

Run::

    python -m eval.run_eval --output eval/results.csv

Use ``--mock`` to skip the real LLM (the mock fakes ``rag.service.generate``
so it returns a reply containing the question's expected keywords).
Useful in CI or offline smoke-tests when no LLM endpoint is reachable.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class EvalRow:
    id: int
    question: str
    answer: str
    refused: bool
    retrieval_count: int
    latency_ms: float
    keywords_hit: int
    keywords_total: int
    accuracy: bool
    faithfulness: float | None = None
    context_precision: float | None = None


def _accuracy(answer: str, keywords: list[str]) -> tuple[int, int, bool]:
    hits = sum(1 for k in keywords if k in answer)
    return hits, len(keywords), hits == len(keywords) and len(keywords) > 0


def _ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> tuple[list[float], list[float]]:
    """Compute per-row ragas scores. Returns (faithfulness, ctx_precision)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, faithfulness

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    ds = Dataset.from_dict(data)
    result = evaluate(ds, metrics=[faithfulness, context_precision])
    df = result.to_pandas()
    return (
        df["faithfulness"].astype(float).tolist(),
        df["context_precision"].astype(float).tolist(),
    )


def _load_questions(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_questions(
    items: Iterable[dict],
    rows: list[EvalRow],
    q_texts: list[str],
    a_texts: list[str],
    c_texts: list[list[str]],
    g_texts: list[str],
) -> None:
    """Exercise the real pipeline on each question and accumulate outputs.

    Re-retrieves after each ``ask`` so ragas gets the real context (without
    widening the AskResult dataclass for one-off callers).
    """
    from rag.retriever import retrieve as _retrieve
    from rag.service import ask

    for item in items:
        t0 = time.monotonic()
        result = ask(item["question"])
        elapsed = (time.monotonic() - t0) * 1000
        hits, total, acc = _accuracy(result.answer, item.get("expected_keywords", []))
        rows.append(EvalRow(
            id=item["id"],
            question=item["question"],
            answer=result.answer,
            refused=result.refused,
            retrieval_count=result.retrieval_count,
            latency_ms=elapsed,
            keywords_hit=hits,
            keywords_total=total,
            accuracy=acc,
        ))
        q_texts.append(item["question"])
        a_texts.append(result.answer)
        docs = _retrieve(item["question"])
        c_texts.append([d.chunk.text for d in docs])
        g_texts.append(" ".join(item.get("expected_keywords", [])))


def _summarise_and_write(
    rows: list[EvalRow],
    *,
    no_ragas: bool,
    q_texts: list[str],
    a_texts: list[str],
    c_texts: list[list[str]],
    g_texts: list[str],
    output_path: Path,
) -> int:
    if not no_ragas and rows:
        try:
            logger.info("computing ragas metrics on %d rows", len(rows))
            faith, ctx_p = _ragas_metrics(q_texts, a_texts, c_texts, g_texts)
            for r, f, p in zip(rows, faith, ctx_p):
                r.faithfulness = f
                r.context_precision = p
        except Exception as exc:
            logger.warning("ragas evaluation failed: %s", exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    n = len(rows)
    acc_count = sum(1 for r in rows if r.accuracy)
    refused_count = sum(1 for r in rows if r.refused)
    latencies = [r.latency_ms for r in rows]
    p50 = statistics.median(latencies) if latencies else 0.0
    p90 = (
        statistics.quantiles(latencies, n=10)[8]
        if len(latencies) >= 10
        else max(latencies, default=0.0)
    )
    summary = {
        "questions": n,
        "accuracy": acc_count / n if n else 0.0,
        "refused": refused_count,
        "p50_latency_ms": round(p50, 1),
        "p90_latency_ms": round(p90, 1),
        "mean_faithfulness": (
            round(
                statistics.mean(r.faithfulness for r in rows if r.faithfulness is not None),
                3,
            )
            if any(r.faithfulness is not None for r in rows)
            else None
        ),
        "mean_context_precision": (
            round(
                statistics.mean(
                    r.context_precision for r in rows if r.context_precision is not None
                ),
                3,
            )
            if any(r.context_precision is not None for r in rows)
            else None
        ),
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")
    print(f"\nresults written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suite.")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--output", default="eval/results.csv")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip ragas (keyword accuracy only).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N questions.")
    parser.add_argument("--mock", action="store_true",
                        help="Patch rag.service.generate so no LLM is called.")
    args = parser.parse_args(argv)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("questions file not found: %s", questions_path)
        return 1

    raw = _load_questions(questions_path)
    if args.limit:
        raw = raw[: args.limit]

    rows: list[EvalRow] = []
    q_texts: list[str] = []
    a_texts: list[str] = []
    c_texts: list[list[str]] = []
    g_texts: list[str] = []

    if args.mock:
        # Patch rag.service.generate so the LLM step is bypassed. The fake
        # returns a reply containing the question's expected keywords, so
        # the accuracy metric exercises the full service plumbing without
        # needing a reachable LLM endpoint.
        from unittest.mock import patch
        from rag.generator import ChatResult

        def _fake_generate(messages, docs, **_kwargs):  # noqa: ANN001
            question = next(
                (m.content for m in reversed(messages) if m.role == "user"),
                "",
            )
            keywords = _mock_keywords.get(question, [])
            return ChatResult(
                answer=" ".join(keywords) + " [1]",
                citations=list(docs),
                refused=False,
                model="mock-model",
            )

        _mock_keywords: dict[str, list[str]] = {
            item["question"]: item.get("expected_keywords", [])
            for item in raw
        }
        with patch("rag.service.generate", side_effect=_fake_generate):
            _run_questions(raw, rows, q_texts, a_texts, c_texts, g_texts)
    else:
        _run_questions(raw, rows, q_texts, a_texts, c_texts, g_texts)

    return _summarise_and_write(
        rows,
        no_ragas=args.no_ragas,
        q_texts=q_texts,
        a_texts=a_texts,
        c_texts=c_texts,
        g_texts=g_texts,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    sys.exit(main())