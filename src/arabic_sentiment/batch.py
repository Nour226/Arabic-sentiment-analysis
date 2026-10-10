"""Score CSV or Parquet review data and write Parquet predictions."""

import argparse
from pathlib import Path

import pandas as pd

from arabic_sentiment.model import SentimentPredictor
from monitoring.batch_report import write_batch_drift_report


def score_batch(
    input_path: Path,
    output_path: Path,
    max_rows: int = 1000,
    model_path: Path | None = None,
) -> pd.DataFrame:
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if input_path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(input_path)
    else:
        rows = pd.read_csv(input_path)
    if "text" not in rows.columns:
        raise ValueError("Input dataset must contain a 'text' column")

    rows = rows.dropna(subset=["text"]).head(max_rows).copy()
    if rows.empty:
        raise ValueError("Input dataset contains no non-empty review text")

    predictor = SentimentPredictor(model_path) if model_path else SentimentPredictor()
    predictions = [predictor.predict(str(text)) for text in rows["text"]]
    rows["predicted_label"] = [prediction[0] for prediction in predictions]
    rows["confidence"] = [prediction[1] for prediction in predictions]
    for label in ("negative", "neutral", "positive"):
        rows[f"probability_{label}"] = [
            prediction[2][label] for prediction in predictions
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output_path, index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/reviews_sample.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/batch/predictions.parquet")
    )
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference", type=Path, help="Reference CSV for the post-scoring drift snapshot")
    parser.add_argument(
        "--drift-report",
        type=Path,
        default=Path("reports/drift_latest.json"),
    )
    args = parser.parse_args()
    result = score_batch(args.data, args.output, args.max_rows, args.model)
    if args.reference:
        reference = pd.read_csv(args.reference)
        write_batch_drift_report(reference, result, args.drift_report)
    print(f"Scored {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
