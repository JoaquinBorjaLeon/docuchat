"""
Evaluate retrieval quality using recall@k.

For each question in the dataset, runs the hybrid retriever and checks
whether at least one chunk from the expected source PDF appears in the
top-k results.  Prints per-question results and a global hit rate.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import retrieve

DATASET = Path(__file__).resolve().parent / "dataset.json"


def run_eval(k: int = 8):
    with open(DATASET, encoding="utf-8") as f:
        cases = json.load(f)

    hits = 0
    total = len(cases)

    print(f"Running recall@{k} evaluation — {total} questions\n")
    print(f"{'#':<4} {'Hit':>4}  {'Question':<55} Sources retrieved")
    print(f"{'─'*4} {'─'*4}  {'─'*55} {'─'*30}")

    for i, case in enumerate(cases, 1):
        question = case["question"]
        expected = [e.lower() for e in case["expected_sources"]]

        chunks = retrieve(question, k=k)
        sources = [c["source"] for c in chunks]

        hit = any(
            exp in s.lower()
            for s in sources
            for exp in expected
        )
        if hit:
            hits += 1

        marker = "✅" if hit else "❌"
        unique_sources = sorted(set(sources))
        print(f"{i:<4} {marker:>4}  {question:<55} {', '.join(unique_sources)}")

    rate = hits / total * 100 if total else 0
    print(f"\n{'═'*80}")
    print(f"Recall@{k}: {hits}/{total} = {rate:.1f}%")

    if rate < 70:
        print("⚠️  Below 70% — consider improving chunking or adding keyword overlap.")
    elif rate < 90:
        print("📈 Decent, but there's room to improve.")
    else:
        print("🎯 Great retrieval quality!")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run_eval(k)
