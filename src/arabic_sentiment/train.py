"""Fine-tune AraBERT on a CSV dataset using CPU or GPU.

Expected CSV columns: text,label. Labels must be negative, neutral, or positive.
Example: python -m src.train --data data/reviews.csv --output models/arabic-sentiment
"""

import argparse
from pathlib import Path

LABELS = ["negative", "neutral", "positive"]

def train(data_path: Path, output_path: Path, model_name: str) -> None:
    from datasets import ClassLabel, load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    dataset = load_dataset("csv", data_files=str(data_path))["train"]
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    dataset = dataset.filter(lambda row: isinstance(row["text"], str) and row["text"].strip())
    dataset = dataset.map(lambda row: {"labels": label_to_id[row["label"]]})
    dataset = dataset.cast_column("labels", ClassLabel(names=LABELS))
    split = dataset.train_test_split(test_size=0.2, seed=42)

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
        num_train_epochs=3,
        weight_decay=0.01,
        no_cuda=True,
        report_to="none",
        load_best_model_at_end=True,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized["train"], eval_dataset=tokenized["test"])
    trainer.train()
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved fine-tuned AraBERT to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT on labeled Arabic reviews")
    parser.add_argument("--data", type=Path, default=Path("data/reviews.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/arabic-sentiment"))
    parser.add_argument("--model", default="aubmindlab/bert-base-arabertv02-twitter")
    args = parser.parse_args()
    train(args.data, args.output, args.model)


if __name__ == "__main__":
    main()