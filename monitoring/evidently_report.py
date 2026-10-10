"""Generate the version-pinned Evidently drift report for a scored batch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_report(reference_path: Path, current_path: Path, output_path: Path) -> None:
    """Write a human-readable Data Drift and Data Quality report."""
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataQualityPreset

    def read_frame(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    reference = read_frame(reference_path)
    current = read_frame(current_path)
    report = Report(
        metrics=[DataDriftPreset(), DataQualityPreset()],
    )
    snapshot = report.run(current_data=current, reference_data=reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/evidently_latest.html"))
    args = parser.parse_args()
    generate_report(args.reference, args.current, args.output)


if __name__ == "__main__":
    main()
