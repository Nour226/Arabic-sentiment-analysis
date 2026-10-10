# Monitoring runbook

## Overview

This runbook covers the monitoring checks shipped in Module 5: a Prometheus-style metrics exporter and a PSI/KS drift detector used to compare a stable reference distribution with a current batch.

## Alerts

- Trigger a checkpoint review when PSI > 0.25.
- Trigger a rollback review when KS > 0.10.
- Alert on a sustained spike in `predict` latency above 300 ms p95.

## Response checklist

1. Confirm the current release and compare the last healthy reference distribution.
2. Inspect the Prometheus metrics payload and check whether the drift score crosses the threshold.
3. Re-run the drift report against the latest batch and validate whether the issue is a genuine model/data shift.
4. If the signal is confirmed, freeze the deployment and rollback to the last stable version.
5. Log the incident and record the remediation in the release notes.

## Local validation

```bash
python -m monitoring --reference data/raw/reviews_sample.csv --current data/raw/reviews_sample.csv
```

The command returns an explicit drift summary and prints the status: `stable` or `drifted`.
