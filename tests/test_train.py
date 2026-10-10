import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from arabic_sentiment import train


def test_compute_metrics_returns_accuracy_and_macro_f1():
    evaluation = SimpleNamespace(
        predictions=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4], [0, 4, 0]]),
        label_ids=np.array([0, 1, 2, 1]),
    )

    metrics = train.compute_metrics(evaluation)

    assert metrics == {"accuracy": 1.0, "f1_macro": 1.0}


def test_compute_metrics_accepts_tuple_predictions():
    evaluation = SimpleNamespace(
        predictions=(np.array([[0, 3, 0], [3, 0, 0]]), np.array([0, 0])),
        label_ids=np.array([1, 0]),
    )

    metrics = train.compute_metrics(evaluation)

    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_main_forwards_cli_options(monkeypatch, tmp_path):
    captured = {}

    def fake_train(*args):
        captured["args"] = args

    monkeypatch.setattr(train, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(tmp_path / "sample.csv"),
            "--output",
            str(tmp_path / "model"),
            "--model",
            "local/model",
            "--max-samples",
            "12",
            "--epochs",
            "1",
            "--metrics-file",
            str(tmp_path / "metrics.json"),
        ],
    )

    train.main()

    assert captured["args"] == (
        tmp_path / "sample.csv",
        tmp_path / "model",
        "local/model",
        12,
        1,
        tmp_path / "metrics.json",
    )


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def filter(self, predicate):
        return FakeDataset([row for row in self.rows if predicate(row)])

    def shuffle(self, seed):
        return self

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])

    def map(self, function, batched=False, remove_columns=None):
        if batched:
            function({"text": [row["text"] for row in self.rows]})
            return self
        updated = []
        for row in self.rows:
            mapped = {**row, **function(row)}
            for column in remove_columns or []:
                mapped.pop(column, None)
            updated.append(mapped)
        return FakeDataset(updated)

    def cast_column(self, name, class_label):
        return self

    def train_test_split(self, test_size, seed):
        split_at = max(1, int(len(self.rows) * (1 - test_size)))
        return FakeDatasetDict(
            {
                "train": FakeDataset(self.rows[:split_at]),
                "test": FakeDataset(self.rows[split_at:]),
            }
        )


class FakeDatasetDict(dict):
    def map(self, function, batched=False):
        return FakeDatasetDict(
            {name: dataset.map(function, batched=batched) for name, dataset in self.items()}
        )


def test_train_runs_and_writes_metrics(monkeypatch, tmp_path):
    rows = [
        {"text": f"review {index}", "label": "positive" if index % 2 else "negative"}
        for index in range(10)
    ]
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: None,
        start_run=nullcontext,
        log_metrics=lambda metrics: logged_metrics.update(metrics),
    )
    logged_metrics = {}
    fake_datasets = SimpleNamespace(
        ClassLabel=lambda names: names,
        load_dataset=lambda *args, **kwargs: FakeDatasetDict(
            {"train": FakeDataset(rows)}
        ),
    )

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts]}

        def save_pretrained(self, path):
            (path / "tokenizer.saved").write_text("saved", encoding="utf-8")

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self):
            return None

        def evaluate(self):
            return {
                "eval_accuracy": 0.75,
                "eval_f1_macro": 0.7,
                "eval_loss": 0.5,
            }

        def save_model(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.saved").write_text("saved", encoding="utf-8")

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda _: FakeTokenizer()
        ),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: object()
        ),
        Trainer=FakeTrainer,
        TrainingArguments=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    metrics_path = tmp_path / "reports" / "metrics.json"
    output_path = tmp_path / "model"

    train.train(
        tmp_path / "sample.csv",
        output_path,
        "fake/model",
        max_samples=8,
        epochs=1,
        metrics_path=metrics_path,
    )

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "accuracy": 0.75,
        "f1_macro": 0.7,
        "eval_loss": 0.5,
    }
    assert logged_metrics["f1_macro"] == 0.7
    assert (output_path / "model.saved").exists()
    assert (output_path / "tokenizer.saved").exists()


def test_train_requires_at_least_two_valid_rows(monkeypatch, tmp_path):
    fake_mlflow = SimpleNamespace()
    fake_datasets = SimpleNamespace(
        ClassLabel=lambda names: names,
        load_dataset=lambda *args, **kwargs: FakeDatasetDict(
            {"train": FakeDataset([])}
        ),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(ValueError, match="at least two"):
        train.train(tmp_path / "sample.csv", tmp_path / "model", "fake/model")