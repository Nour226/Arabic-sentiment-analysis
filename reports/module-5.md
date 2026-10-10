# Module 5: Monitoring and Drift Detection

## Delivered

- Prometheus-compatible text exporter for request latency, drift metrics, and counters.
- PSI and KS functions for stage-to-production drift checks.
- Drift summary and threshold gates suited for release monitoring.
- A local CLI and runbook for alert response and rollback guidance.

The implementation is deliberately a lightweight local monitoring primitive, not a claim that the full handbook observability stack is installed. It has no HTTP `/metrics` route, Grafana provisioning, Evidently TestSuite, PostgreSQL drift history, Airflow schedule, or automated retraining branch yet.

## Verification

- Unit tests validate zero-drift stability, clear shift detection, and export payload rendering.
- The monitoring helpers are importable from both the root `monitoring` package and the `arabic_sentiment` compatibility wrapper.
- Ruff and the project suite pass with the full monitoring checks included.

## Usage

```bash
python -m monitoring --reference data/raw/reviews_sample.csv --current data/raw/reviews_sample.csv
```

## Operational thresholds

- PSI > 0.25 is flagged as drifted.
- KS > 0.10 is flagged as drifted.
- Prometheus metrics are rendered as text by the exporter and are not yet exposed by an HTTP `/metrics` endpoint.

The release tag for this module is `v0.5.0`.
