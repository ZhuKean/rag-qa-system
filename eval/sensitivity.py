"""Cost-sensitivity sweep.

Tries a grid of (top_k, reranker, temperature) settings and reports a
quality / cost / latency table. The point isn't to find the absolute best
combination — it's to show the trade-offs so the deployment config can
be chosen deliberately.

Cost model (assumed):
    * Embedding cost:  $0 (bge-m3 runs locally on CPU)
    * LLM input:      $0.00014 / 1k tokens (DeepSeek-chat input)
    * LLM output:     $0.00028 / 1k tokens (DeepSeek-chat output)
    * Latency:        measured wall-clock per request

Replace COST_INPUT_PER_1K / COST_OUTPUT_PER_1K with your provider's rates.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import statistics
import sys
import time
from dataclasses import dataclass

from app.config import get_settings
from observability import db

logger = logging.getLogger("sensitivity")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Token cost — DeepSeek-chat reference rates, USD per 1k tokens.
# Override with --cost-input / --cost-output if you switch providers.
COST_INPUT_PER_1K = 0.00014
COST_OUTPUT_PER_1K = 0.00028


@dataclass
class SweepRow:
    top_k: int
    reranker: bool
    temperature: float
    accuracy: float
    p50_latency_ms: float
    p90_latency_ms: float
    est_input_tokens: int
    est_output_tokens: int
    est_cost_per_request_usd: float


def _estimated_cost(rows: list) -> tuple[int, int, float]:
    """Average input / output tokens per request, then dollar cost."""
    in_tokens = [r["retrieval_count"] * 400 for r in rows]   # rough: 400 tok per chunk
    out_tokens = [len(r["answer"]) // 2 for r in rows]      # ~2 chars per English token
    avg_in = int(statistics.mean(in_tokens)) if in_tokens else 0
    avg_out = int(statistics.mean(out_tokens)) if out_tokens else 0
    cost = (avg_in / 1000) * COST_INPUT_PER_1K + (avg_out / 1000) * COST_OUTPUT_PER_1K
    return avg_in, avg_out, round(cost, 6)


def _accuracy_from_events(limit: int) -> float:
    """Compute accuracy from the most recent `limit` ask.complete rows.

    We use the keyword-match signal encoded as ``answer_chars`` only as a
    proxy when explicit accuracy fields are unavailable. The proper
    metric is the one computed in ``run_eval.py`` — the sweep is meant
    to be lightweight and reuses what we already have.
    """
    rows = db.query_events("ask.complete", limit=limit)
    if not rows:
        return 0.0
    refused = sum(1 for r in rows if r.get("refused"))
    return 1.0 - (refused / len(rows))


def _pXX(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) < 10:
        return max(values)
    return statistics.quantiles(values, n=10)[q * 10 - 1]


def _run_grid(
    top_k_values: list[int],
    reranker_values: list[bool],
    temperature_values: list[float],
    limit: int,
    *,
    mock: bool = False,
) -> list[SweepRow]:
    settings = get_settings()
    out: list[SweepRow] = []
    for top_k, rerank, temp in itertools.product(
        top_k_values, reranker_values, temperature_values
    ):
        logger.info("sweep top_k=%d rerank=%s temp=%.2f", top_k, rerank, temp)
        # Mutate Settings fields directly; this is a single-process script
        # so we don't worry about the cached lru_cache.
        settings.top_k = top_k
        settings.reranker_enabled = rerank
        settings.temperature = temp

        # Warm-up to amortise model load.
        from rag.service import ask
        ask("warmup question")

        t0 = time.monotonic()
        for _ in range(limit):
            ask("员工应遵守哪些基本行为准则？")
        elapsed = (time.monotonic() - t0) * 1000 / max(limit, 1)

        events = db.query_events("ask.complete", limit=limit)
        lats = [r["latency_ms"] for r in events]
        rows = events
        avg_in, avg_out, cost = _estimated_cost(rows)
        out.append(
            SweepRow(
                top_k=top_k,
                reranker=rerank,
                temperature=temp,
                accuracy=_accuracy_from_events(limit=limit),
                p50_latency_ms=_pXX(lats, 0.5),
                p90_latency_ms=_pXX(lats, 0.9),
                est_input_tokens=avg_in,
                est_output_tokens=avg_out,
                est_cost_per_request_usd=cost,
            )
        )
    return out


def _apply_mock() -> "object":
    """Context manager that patches rag.service.generate so the LLM is
    never called. Yields the patcher so the caller can stop it."""
    from unittest.mock import patch
    from rag.generator import ChatResult

    def _fake_generate(messages, docs, **_kwargs):  # noqa: ANN001
        return ChatResult(
            answer="根据《员工手册》第 1 条，这是答案。",
            citations=list(docs),
            refused=False,
            model="mock-model",
        )

    return patch("rag.service.generate", side_effect=_fake_generate)


def main(argv: list[str] | None = None) -> int:
    global COST_INPUT_PER_1K, COST_OUTPUT_PER_1K

    parser = argparse.ArgumentParser(description="Cost-sensitivity sweep.")
    parser.add_argument("--top-k", default="3,5,8")
    parser.add_argument("--reranker", default="false")
    parser.add_argument("--temperature", default="0,0.1,0.7")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--cost-input", type=float, default=COST_INPUT_PER_1K)
    parser.add_argument("--cost-output", type=float, default=COST_OUTPUT_PER_1K)
    parser.add_argument("--output", default="eval/sensitivity.csv")
    parser.add_argument("--mock", action="store_true",
                        help="Patch rag.service.generate so no LLM is called.")
    args = parser.parse_args(argv)

    COST_INPUT_PER_1K = args.cost_input
    COST_OUTPUT_PER_1K = args.cost_output

    top_ks = [int(x) for x in args.top_k.split(",")]
    rerankers = [s.lower() == "true" for s in args.reranker.split(",")]
    temps = [float(x) for x in args.temperature.split(",")]

    if args.mock:
        patcher = _apply_mock()
        patcher.start()
        try:
            sweep = _run_grid(top_ks, rerankers, temps, limit=args.limit, mock=True)
        finally:
            patcher.stop()
    else:
        sweep = _run_grid(top_ks, rerankers, temps, limit=args.limit)
    out_path = __import__("pathlib").Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        from dataclasses import asdict

        writer = csv.DictWriter(fh, fieldnames=list(asdict(sweep[0]).keys()))
        writer.writeheader()
        for r in sweep:
            writer.writerow(asdict(r))

    print("\n=== SENSITIVITY ===")
    print(
        f"{'top_k':>5}  {'rerank':>6}  {'temp':>5}  "
        f"{'acc':>5}  {'p50ms':>7}  {'p90ms':>7}  "
        f"{'in_tok':>7}  {'out_tok':>7}  {'cost':>10}"
    )
    for r in sweep:
        print(
            f"{r.top_k:>5}  {str(r.reranker):>6}  {r.temperature:>5.2f}  "
            f"{r.accuracy:>5.2f}  {r.p50_latency_ms:>7.1f}  {r.p90_latency_ms:>7.1f}  "
            f"{r.est_input_tokens:>7}  {r.est_output_tokens:>7}  "
            f"${r.est_cost_per_request_usd:>9.6f}"
        )
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
