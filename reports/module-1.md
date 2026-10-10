# Module 1: Packaging and API Baseline

## Delivered

- Installable `src/arabic_sentiment` package with project metadata in `pyproject.toml`.
- FastAPI health and prediction endpoints with Pydantic schemas.
- Docker image baseline and API contract tests.

## Verification

- Health, prediction, and invalid-input behavior are covered by automated tests.
- The current test suite uses a fake predictor, so API tests require no model download.

## Release

- Baseline release tag: `v0.1.0`.
