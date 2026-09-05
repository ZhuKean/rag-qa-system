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

NB: this script touches the real LLM. Make sure ``.env`` points at a
reachable endpoint (Ollama locally, or DeepSeek for evaluation).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ragas imports are deferred to keep the lightweight tests fast.
# We only need them inside main().
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suite.")
    parser.add_argument(
        "--questions",
        default="eval/questions.jsonl",
        help="Path to the JSONL question file.",
    )
    parser.add_argument(
        "--output",
        default="eval/results.csv",
        help="Where to write the per-question CSV.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip ragas (use only the keyword accuracy metric).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N questions (for smoke tests).",
    )
    args = parser.parse_args(argv)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("questions file not found: %s", questions_path)
        return 1

    from rag.service import ask  # deferred: pulls in the full service

    raw = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        raw = raw[: args.limit]

    rows: list[EvalRow] = []
    q_texts: list[str] = []
    a_texts: list[str] = []
    c_texts: list[list[str]] = []
    g_texts: list[str] = []

    for item in raw:
        t0 = time.monotonic()
        result = ask(item["question"])
        elapsed = (time.monotonic() - t0) * 1000
        hits, total, acc = _accuracy(result.answer, item.get("expected_keywords", []))
        rows.append(
            EvalRow(
                id=item["id"],
                question=item["question"],
                answer=result.answer,
                refused=result.refused,
                retrieval_count=result.retrieval_count,
                latency_ms=elapsed,
                keywords_hit=hits,
                keywords_total=total,
                accuracy=acc,
            )
        )
        q_texts.append(item["question"])
        a_texts.append(result.answer)
        # We don't have the raw docs in AskResult — re-retrieve here
        # so ragas gets real context. Costs an extra embed + query but
        # keeps the API contract small.
        from rag.retriever import retrieve

        docs = retrieve(item["question"])
        c_texts.append([d.chunk.text for d in docs])
        g_texts.append(" ".join(item.get("expected_keywords", [])))

    if not args.no_ragas and rows:
        try:
            logger.info("computing ragas metrics on %d rows", len(rows))
            faith, ctx_p = _ragas_metrics(q_texts, a_texts, c_texts, g_texts)
            for r, f, p in zip(rows, faith, ctx_p):
                r.faithfulness = f
                r.context_precision = p
        except Exception as exc:
            logger.warning("ragas evaluation failed: %s", exc)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    # Summary
    n = len(rows)
    acc_count = sum(1 for r in rows if r.accuracy)
    refused_count = sum(1 for r in rows if r.refused)
    latencies = [r.latency_ms for r in rows]
    p50 = statistics.median(latencies) if latencies else 0
    p90 = (
        statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else max(latencies, default=0)
    )
    summary = {
        "questions": n,
        "accuracy": acc_count / n if n else 0,
        "refused": refused_count,
        "p50_latency_ms": round(p50, 1),
        "p90_latency_ms": round(p90, 1),
        "mean_faithfulness": (
            round(statistics.mean(r.faithfulness for r in rows if r.faithfulness is not None), 3)
            if any(r.faithfulness is not None for r in rows)
            else None
        ),
        "mean_context_precision": (
            round(statistics.mean(r.context_precision for r in rows if r.context_precision is not None), 3)
            if any(r.context_precision is not None for r in rows)
            else None
        ),
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
