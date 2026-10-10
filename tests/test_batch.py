from pathlib import Path

import pandas as pd
import pytest

from arabic_sentiment import batch


class FakePredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, text):
        return "positive", 0.8, {
            "negative": 0.1,
            "neutral": 0.1,
            "positive": 0.8,
        }


def test_score_batch_limits_rows_and_writes_predictions(tmp_path, monkeypatch):
    input_path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {"text": ["review one", "review two", None], "source": [1, 2, 3]}
    ).to_csv(input_path, index=False)
    captured = {}

    def fake_to_parquet(frame, path, index):
        captured["frame"] = frame.copy()
        captured["path"] = Path(path)
        captured["index"] = index

    monkeypatch.setattr(batch, "SentimentPredictor", FakePredictor)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    output_path = tmp_path / "out" / "predictions.parquet"

    result = batch.score_batch(input_path, output_path, max_rows=1)

    assert len(result) == 1
    assert result.iloc[0]["predicted_label"] == "positive"
    assert result.iloc[0]["probability_positive"] == 0.8
    assert captured["path"] == output_path
    assert captured["index"] is False


def test_score_batch_requires_text_column(tmp_path):
    input_path = tmp_path / "invalid.csv"
    pd.DataFrame({"review": ["text"]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="text.*column"):
        batch.score_batch(input_path, tmp_path / "out.parquet")


def test_score_batch_rejects_empty_data_and_invalid_limit(tmp_path):
    input_path = tmp_path / "empty.csv"
    pd.DataFrame({"text": [None]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="no non-empty"):
        batch.score_batch(input_path, tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="max_rows must be positive"):
        batch.score_batch(input_path, tmp_path / "out.parquet", max_rows=0)


def test_batch_drift_report_is_written(tmp_path):
    input_path = tmp_path / "reviews.csv"
    input_path.write_text("text,label\nshort,positive\nlonger,negative\n", encoding="utf-8")
    report_path = tmp_path / "drift.json"
    result = batch.score_batch(input_path, tmp_path / "out.parquet")
    batch.write_batch_drift_report(
        result.assign(confidence=0.8), result, report_path
    )
    assert report_path.exists()
