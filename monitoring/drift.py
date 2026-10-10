"""Distribution-drift helpers for production monitoring."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("input values must not be empty")
    return array


def compute_psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """Compute the Population Stability Index (PSI) for two numeric distributions."""
    ref = _as_array(reference)
    cur = _as_array(current)
    if bins < 2:
        raise ValueError("bins must be at least 2")

    if np.allclose(ref, cur):
        return 0.0

    lower = min(float(np.min(ref)), float(np.min(cur)))
    upper = max(float(np.max(ref)), float(np.max(cur)))
    if np.isclose(lower, upper):
        spread = max(abs(lower), 1.0)
        lower -= spread
        upper += spread

    edges = np.linspace(lower, upper, bins + 1)
    expected, _ = np.histogram(ref, bins=edges)
    actual, _ = np.histogram(cur, bins=edges)
    expected_pct = np.clip(expected / expected.sum(), 1e-6, None)
    actual_pct = np.clip(actual / actual.sum(), 1e-6, None)

    score = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(score)


def ks_statistic(reference: Sequence[float], current: Sequence[float]) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic."""
    ref = _as_array(reference)
    cur = _as_array(current)

    ref_sorted = np.sort(ref)
    cur_sorted = np.sort(cur)
    values = np.unique(np.concatenate((ref_sorted, cur_sorted)))

    ref_cdf = np.searchsorted(ref_sorted, values, side="right") / ref_sorted.size
    cur_cdf = np.searchsorted(cur_sorted, values, side="right") / cur_sorted.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def detect_drift(
    reference: Sequence[float],
    current: Sequence[float],
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.1,
) -> dict[str, float | bool | str]:
    """Return a concise summary of whether the current data drifts from the reference."""
    psi_score = compute_psi(reference, current)
    ks_score = ks_statistic(reference, current)
    significant = psi_score > psi_threshold or ks_score > ks_threshold
    return {
        "psi": float(psi_score),
        "ks_statistic": float(ks_score),
        "significant": bool(significant),
        "status": "drifted" if significant else "stable",
        "psi_threshold": float(psi_threshold),
        "ks_threshold": float(ks_threshold),
    }
