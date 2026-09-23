"""Compare reference and current embedding samples for deployment drift."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_fastapi.ops import embedding_drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--min-centroid-cosine", type=float, default=0.95)
    parser.add_argument("--max-norm-delta", type=float, default=0.10)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    metrics = embedding_drift(reference, current)
    print(json.dumps(metrics, sort_keys=True))

    if metrics["dimension_match"] == 0.0:
        print("Embedding drift gate failed: dimensions do not match", file=sys.stderr)
        return 1
    if metrics["centroid_cosine"] < args.min_centroid_cosine:
        print("Embedding drift gate failed: centroid cosine is below threshold", file=sys.stderr)
        return 1
    if metrics["mean_norm_delta"] > args.max_norm_delta:
        print("Embedding drift gate failed: norm delta is above threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
