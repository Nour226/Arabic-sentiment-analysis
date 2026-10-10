"""Compatibility wrapper for monitoring utilities."""

from monitoring import (
    PrometheusMetricsExporter,
    compute_psi,
    detect_drift,
    ks_statistic,
)

__all__ = [
    "PrometheusMetricsExporter",
    "compute_psi",
    "detect_drift",
    "ks_statistic",
]
