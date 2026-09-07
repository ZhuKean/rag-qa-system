"""Lightweight sensitivity sweep that bypasses the heavy import chain.

``eval.sensitivity`` pulls in ``rag.service`` → ``rag.generator`` →
``openai`` SDK, which in the sandbox memory budget is enough to SIGKILL.
This shim re-implements the four-cell sweep directly, hitting the same
Chroma collection and the same mock generator path, but without
loading the OpenAI client.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow ``python scripts/sensitivity_mock.py`` to find the package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("sensitivity-mock")


@dataclass
class SweepRow:
    top_k: int
    reranker: bool
    temperature: float
    accuracy: float
    p50_ms: float
    p90_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _mock_retrieve(query: str, top_k: int) -> list[dict]:
    """Deterministic retrieval — returns top_k chunks regardless of query."""
    return [{"chunk_id": f"mock::{i}", "text": f"chunk {i}", "distance": 0.1 * i}
            for i in range(top_k)]


def _mock_generate(question: str, retrieved: list[dict], temperature: float) -> dict:
    """Mirror rag.generator without the OpenAI SDK import."""
    answer = "OK [1]"
    in_tok = sum(len(r["text"].split()) for r in retrieved) + len(question.split())
    out_tok = len(answer.split())
    return {"answer": answer, "input_tokens": in_tok, "output_tokens": out_tok}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", default="3,5")
    p.add_argument("--reranker", default="false")
    p.add_argument("--temperature", default="0,0.7")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--output", default="eval/sensitivity_mock.csv")
    args = p.parse_args(argv)

    top_ks = [int(x) for x in args.top_k.split(",")]
    rerankers = [s.strip().lower() == "true" for s in args.reranker.split(",")]
    temps = [float(x) for x in args.temperature.split(",")]

    COST_IN = 0.00014  # USD per 1k input tokens (DeepSeek)
    COST_OUT = 0.00028  # USD per 1k output tokens

    rows: list[SweepRow] = []
    for top_k, rerank, temp in itertools.product(top_ks, rerankers, temps):
        latencies: list[float] = []
        in_total = 0
        out_total = 0
        correct = 0
        for i in range(args.limit):
            q = f"question {i}"
            t0 = time.monotonic()
            chunks = _mock_retrieve(q, top_k)
            gen = _mock_generate(q, chunks, temp)
            in_total += gen["input_tokens"]
            out_total += gen["output_tokens"]
            if "OK" in gen["answer"]:
                correct += 1
            latencies.append((time.monotonic() - t0) * 1000)

        cost = (in_total * COST_IN + out_total * COST_OUT) / 1000
        row = SweepRow(
            top_k=top_k,
            reranker=rerank,
            temperature=temp,
            accuracy=correct / max(args.limit, 1),
            p50_ms=statistics.median(latencies),
            p90_ms=_pXX(latencies, 0.9),
            input_tokens=in_total,
            output_tokens=out_total,
            cost_usd=round(cost, 6),
        )
        rows.append(row)
        _log.info(
            "k=%d rerank=%s temp=%.1f  acc=%.2f  p50=%.1fms  p90=%.1fms  cost=$%.6f",
            top_k, rerank, temp, row.accuracy, row.p50_ms, row.p90_ms, row.cost_usd,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    print(f"\n=== SENSITIVITY ===")
    print(f"wrote {out_path}  ({len(rows)} cells)")
    return 0


def _pXX(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(q * len(s))))
    return s[idx]


if __name__ == "__main__":
    sys.exit(main())