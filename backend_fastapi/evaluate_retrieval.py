"""Run a deterministic retrieval quality gate from labeled JSON cases."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_fastapi.ops import evaluate_retrieval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not cases:
        print("Retrieval evaluation requires at least one labeled case", file=sys.stderr)
        return 2

    metric_totals = {"recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0}
    for case in cases:
        metrics = evaluate_retrieval(
            case["retrieved_ids"],
            case["relevant_ids"],
            k=int(case.get("k", 10)),
        )
        for name in metric_totals:
            metric_totals[name] += metrics[name]

    averages = {name: value / len(cases) for name, value in metric_totals.items()}
    print(json.dumps(averages, sort_keys=True))

    thresholds = {
        "recall_at_k": float(os.getenv("MIN_RECALL_AT_K", "0.0")),
        "mrr_at_k": float(os.getenv("MIN_MRR_AT_K", "0.0")),
        "ndcg_at_k": float(os.getenv("MIN_NDCG_AT_K", "0.0")),
    }
    failures = [
        f"{name}={averages[name]:.4f} < {threshold:.4f}"
        for name, threshold in thresholds.items()
        if averages[name] < threshold
    ]
    if failures:
        print("Retrieval quality gate failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
