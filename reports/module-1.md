# Module 1: Packaging and API Baseline

## Delivered

- Installable `src/arabic_sentiment` package with project metadata in `pyproject.toml`.
- FastAPI health and prediction endpoints with Pydantic schemas.
- Docker image baseline and API contract tests.

## Verification

- Health, prediction, and invalid-input behavior are covered by automated tests.
- The current test suite uses a fake predictor, so API tests require no model download.

## Maturity self-assessment

This repository is at the reproducible package/service baseline: the code is installable, tested, containerized, and exposes a documented prediction contract. To reach the next maturity level, it needs a real trained model artifact plus the tracking, registry, data-lineage, and deployment evidence delivered in Modules 2 and 3.

## Release

- Baseline release tag: `v0.1.0`.
