"""Fine-tune AraBERT on a CSV dataset using CPU or GPU.

Expected CSV columns: text,label. Labels must be negative, neutral, or positive.
Example: python -m src.train --data data/reviews.csv --output models/arabic-sentiment
"""

import argparse
import json
from pathlib import Path
from typing import Any

from arabic_sentiment.config import settings

LABELS = ["negative", "neutral", "positive"]


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(eval_prediction.label_ids, predictions)),
        "f1_macro": float(
            f1_score(
                eval_prediction.label_ids,
                predictions,
                labels=range(len(LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def train(
    data_path: Path,
    output_path: Path,
    model_name: str,
    max_samples: int | None = None,
    epochs: int = 3,
    metrics_path: Path = Path("reports/training-metrics.json"),
) -> None:
    import mlflow
    from datasets import ClassLabel, load_dataset

    dataset = load_dataset("csv", data_files=str(data_path))["train"]
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    dataset = dataset.filter(lambda row: isinstance(row["text"], str) and row["text"].strip())
    if max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    if len(dataset) < 2:
        raise ValueError("Training requires at least two valid labeled rows")
    dataset = dataset.map(
        lambda row: {"labels": label_to_id[row["label"]]},
        remove_columns=["label"],
    )
    dataset = dataset.cast_column("labels", ClassLabel(names=LABELS))
    split = dataset.train_test_split(test_size=0.2, seed=42)

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = split.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label_to_id
    )

    arguments = TrainingArguments(
        output_dir="artifacts/training",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to=["mlflow"],
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        trainer.train()
        evaluation = trainer.evaluate()
        final_metrics = {
            "accuracy": float(evaluation["eval_accuracy"]),
            "f1_macro": float(evaluation["eval_f1_macro"]),
            "eval_loss": float(evaluation["eval_loss"]),
        }
        mlflow.log_metrics(final_metrics)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fine-tuned AraBERT to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT on labeled Arabic reviews")
    parser.add_argument("--data", type=Path, default=Path("data/raw/reviews_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/arabic-sentiment"))
    parser.add_argument("--model", default="aubmindlab/bert-base-arabertv02-twitter")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--metrics-file", type=Path, default=Path("reports/training-metrics.json"))
    args = parser.parse_args()
    train(args.data, args.output, args.model, args.max_samples, args.epochs, args.metrics_file)


if __name__ == "__main__":
    main()