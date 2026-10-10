"""CLI for local monitoring checks."""

from __future__ import annotations

import argparse

from .drift import detect_drift


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for data drift and export metrics.")
    parser.add_argument("--reference", type=str, required=True, help="Reference sample file path")
    parser.add_argument("--current", type=str, required=True, help="Current sample file path")
    args = parser.parse_args()

    import csv

    def load_values(path: str) -> list[float]:
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "value" in fieldnames:
                return [float(row["value"]) for row in reader if row.get("value") not in (None, "")]
            numeric_values: list[float] = []
            for row in reader:
                for field in fieldnames:
                    if field in {"text", "label"}:
                        continue
                    try:
                        numeric_values.append(float(row[field]))
                    except (KeyError, TypeError, ValueError):
                        continue
            if numeric_values:
                return numeric_values

        raise ValueError(
            f"No numeric 'value' or feature column found in {path!r}; "
            "provide a CSV with numeric monitoring values"
        )

    try:
        reference = load_values(args.reference)
        current = load_values(args.current)
    except (FileNotFoundError, ValueError, TypeError) as exc:  # pragma: no cover - CLI guard
        raise SystemExit(str(exc)) from exc

    summary = detect_drift(reference, current)
    print(summary)


if __name__ == "__main__":
    main()
