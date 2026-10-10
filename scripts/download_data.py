from pathlib import Path

import kagglehub
import pandas as pd


def fetch_and_prepare_dataset():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_dir = Path(
        kagglehub.dataset_download("abdallaellaithy/330k-arabic-sentiment-reviews")
    )
    csv_file = next(download_dir.rglob("*.csv"), None)
    if csv_file is None:
        raise FileNotFoundError(f"No CSV found in downloaded dataset: {download_dir}")

    df = pd.read_csv(csv_file)
    df = df.rename(columns={"content": "text"})
    if "text" not in df or "label" not in df:
        raise ValueError("Dataset CSV must contain text/content and label columns")

    labels = df["label"]
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    df["label"] = labels.astype("string").str.lower().map(
        {"1": "positive", "0": "negative", "positive": "positive", "negative": "negative", "neutral": "neutral"}
    )
    numeric_mask = numeric_labels.notna()
    df.loc[numeric_mask, "label"] = numeric_labels[numeric_mask].map(
        {1: "positive", 0: "negative"}
    )
    df = df.dropna(subset=["text", "label"])
    if df.empty:
        raise ValueError("No rows with supported text and sentiment labels were found")

    sample_size = min(20_000, len(df))
    output_path = raw_dir / "reviews_sample.csv"
    df.sample(n=sample_size, random_state=42)[["text", "label"]].to_csv(
        output_path, index=False
    )
    print(f"Saved {sample_size:,} rows to {output_path}")

if __name__ == "__main__":
    fetch_and_prepare_dataset()