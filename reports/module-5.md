# Module 5: Monitoring and Drift Detection

## Delivered

- Prometheus-compatible text exporter for request latency, drift metrics, and counters.
- PSI and KS functions for stage-to-production drift checks.
- Drift summary and threshold gates suited for release monitoring.
- A local CLI and runbook for alert response and rollback guidance.

The monitoring package exposes an HTTP `/metrics` endpoint through the FastAPI service and provides repository-provisioned Prometheus and Grafana configuration.

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
- Prometheus metrics are rendered as text by the exporter and exposed at `/metrics`.

The release tag for this module is `v0.5.0`.
