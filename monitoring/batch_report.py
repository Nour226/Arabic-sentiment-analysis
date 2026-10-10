"""Create a lightweight drift snapshot for every scored batch."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .drift import detect_drift


def write_batch_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> dict[str, object]:
    """Compare text length and confidence, then persist a JSON snapshot."""
    if "text" not in reference or "text" not in current:
        raise ValueError("Both monitoring frames must contain a 'text' column")
    if "confidence" not in current:
        raise ValueError("The scored batch must contain a 'confidence' column")

    reference_lengths = reference["text"].astype(str).str.len().tolist()
    current_lengths = current["text"].astype(str).str.len().tolist()
    confidence = current["confidence"].astype(float).tolist()
    report = {
        "features": {
            "text_length": detect_drift(reference_lengths, current_lengths),
            "confidence_score": {
                "mean": float(sum(confidence) / len(confidence)) if confidence else 0.0,
                "sample_count": len(confidence),
            },
        },
        "sample_count": len(current),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
