import pytest

from monitoring import (
    PrometheusMetricsExporter,
    compute_psi,
    detect_drift,
    ks_statistic,
)


def test_compute_psi_identical_distribution_is_zero():
    reference = [0.1, 0.2, 0.3, 0.4, 0.5]

    assert compute_psi(reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_compute_psi_detects_drift():
    reference = [0.1, 0.2, 0.3, 0.4, 0.5]
    current = [0.01, 0.01, 0.01, 0.01, 0.96]

    assert compute_psi(reference, current) > 0.2


def test_ks_statistic_detects_distribution_shift():
    reference = [0.0, 0.1, 0.2, 0.3]
    current = [0.8, 0.9, 1.0, 1.1]

    assert ks_statistic(reference, current) > 0.5


def test_detect_drift_returns_summary_for_merged_inputs():
    summary = detect_drift([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0])

    assert summary["psi"] > 0.0
    assert summary["ks_statistic"] > 0.5
    assert "significant" in summary


def test_prometheus_exporter_renders_metrics():
    exporter = PrometheusMetricsExporter(namespace="arabic_sentiment")
    exporter.record_http_request("predict", 150.0)
    exporter.record_drift("label_shift", 0.8)

    payload = exporter.render()
    assert "arabic_sentiment_http_request_ms" in payload
    assert "arabic_sentiment_drift_score" in payload
    assert "150" in payload
