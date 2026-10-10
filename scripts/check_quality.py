"""Fail when the tracked model quality metric is below the release baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline", type=float, default=0.70)
    args = parser.parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"metrics file not found: {args.metrics}; run training first")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    score = float(metrics["f1_macro"])
    if score < args.baseline:
        raise SystemExit(
            f"quality gate failed: f1_macro={score:.4f} < baseline={args.baseline:.4f}"
        )
    print(f"quality gate passed: f1_macro={score:.4f} >= baseline={args.baseline:.4f}")


if __name__ == "__main__":
    main()
