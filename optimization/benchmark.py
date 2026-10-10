"""Compare FP32 and INT8 ONNX model size, latency, and labeled accuracy."""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from arabic_sentiment.model import SentimentPredictor


def read_rows(dataset_path: Path, max_rows: int) -> pd.DataFrame:
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if dataset_path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(dataset_path)
    else:
        rows = pd.read_csv(dataset_path)
    if "text" not in rows.columns:
        raise ValueError("Dataset must include a text column")
    return rows.dropna(subset=["text"]).head(max_rows).copy()


def evaluate_model(
    model_path: Path,
    rows: pd.DataFrame,
    predictor_factory: Callable[[Path], SentimentPredictor] = SentimentPredictor,
) -> dict[str, float | int | None]:
    predictor = predictor_factory(model_path)
    latencies = []
    predictions = []
    for text in rows["text"]:
        started = time.perf_counter()
        label, _, _ = predictor.predict(str(text))
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append(label)

    accuracy = None
    if "label" in rows.columns:
        valid = rows["label"].notna()
        if valid.any():
            accuracy = float(
                sum(
                    str(expected).lower() == actual
                    for expected, actual in zip(rows.loc[valid, "label"], predictions)
                )
                / int(valid.sum())
            )

    return {
        "rows": len(rows),
        "accuracy": accuracy,
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": float(sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1]),
    }


def benchmark_models(
    fp32_path: Path,
    int8_path: Path,
    dataset_path: Path,
    max_rows: int = 1000,
    predictor_factory: Callable[[Path], SentimentPredictor] = SentimentPredictor,
) -> dict[str, object]:
    for path in (fp32_path, int8_path):
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
    rows = read_rows(dataset_path, max_rows)
    if rows.empty:
        raise ValueError("Dataset contains no review text")

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    return {
        "dataset": str(dataset_path),
        "fp32_model": str(fp32_path),
        "int8_model": str(int8_path),
        "fp32_size_bytes": fp32_size,
        "int8_size_bytes": int8_size,
        "size_reduction_percent": (1 - int8_size / fp32_size) * 100,
        "fp32": evaluate_model(fp32_path, rows, predictor_factory),
        "int8": evaluate_model(int8_path, rows, predictor_factory),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--int8", type=Path, default=Path("models/model_int8.onnx"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/reviews_sample.csv"))
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/optimization/benchmark.json")
    )
    args = parser.parse_args()
    results = benchmark_models(args.fp32, args.int8, args.data, args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()