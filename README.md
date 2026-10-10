# Arabic Sentiment Analysis MLOps

An end-to-end learning project for Arabic sentiment classification with AraBERT, FastAPI, ONNX Runtime, MLflow, DVC, and automated quality checks.

## Current Status

- Module 1 (`v0.1.0`): installable Python package and FastAPI baseline.
- Module 2 (`v0.2.0`): experiment tracking, MinIO artifact storage, DVC dataset lineage, and CI quality gates.
- Modules 3–5: serving/load testing, optimization, and monitoring are documented as they are completed.

## Requirements

- Python 3.11.
- Docker Desktop with Docker Compose v2 for MLflow and MinIO.
- Git.
- Internet access for the initial Kaggle and Hugging Face downloads.

On Windows, activate the project venv with `venv\\Scripts\\activate`. Install the project and extras:

```cmd
python -m pip install -e ".[dev,ml,data]"
```

## Dataset and DVC

The downloader fetches the Kaggle dataset and produces a deterministic sample of up to 20,000 rows at `data/raw/reviews_sample.csv`:

```cmd
python scripts/download_data.py
```

The sample is represented in Git by `data/raw/reviews_sample.csv.dvc`; the CSV itself is ignored by Git. The 330k-row source CSV is also ignored. After configuring and populating a DVC remote, use `dvc pull` on another machine; until then, run the downloader locally.

## Experiment Tracking and Training

Start MLflow and MinIO:

```cmd
docker compose -f docker/docker-compose.yml up -d
```

- MLflow UI: http://localhost:5000
- MinIO console: http://localhost:9001
- Local development MinIO credentials default to `minioadmin` / `minioadmin123`; override them with `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` environment variables.

Train on the sample and log loss, accuracy, and macro-F1 to MLflow:

```cmd
python -m arabic_sentiment.train --data data/raw/reviews_sample.csv
```

For a short CPU smoke run, use `--max-samples 16 --epochs 1 --output artifacts/smoke-model`. The default training run uses 20,000 rows and 3 epochs and can take a long time without a GPU. The DVC pipeline is configured in `pipelines/dvc.yaml` and `pipelines/params.yaml`:

```cmd
dvc repro pipelines/dvc.yaml:train
```

## API

Start the local API after installing the package:

```cmd
uvicorn arabic_sentiment.api.main:app --reload --port 8000
```

The API exposes `/health` and `/predict`. Until an ONNX model is exported, the predictor uses its explicit development fallback. The API container is opt-in in Compose with `--profile api`.

## Quality Checks

```cmd
ruff check src tests
pytest -v --cov=src/arabic_sentiment --cov-fail-under=80
```

GitHub Actions runs lint, tests with the 80% coverage gate, DVC graph validation, Compose validation, and an API Docker build.

## Repository Layout

- `src/arabic_sentiment/`: package, training, API, and model export/inference.
- `data/`: local datasets; tracked by DVC where applicable, not stored as Git blobs.
- `pipelines/`: DVC pipeline and parameters.
- `docker/`: API, MLflow, MinIO, and Compose configuration.
- `reports/`: module implementation and validation notes.
- `tests/`: fast deterministic unit tests; tests do not download pretrained models.

## Releases

Releases are tagged after the corresponding module branch is reviewed and merged to `main`: `v0.1.0` through `v0.5.0`.