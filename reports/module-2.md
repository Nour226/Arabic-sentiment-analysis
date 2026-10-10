# Module 2: Experiment Tracking, DVC, and CI

## Delivered

- MLflow tracking server with a SQLite backend and MinIO artifact storage.
- Kaggle dataset preparation producing a reproducible sample of up to 20,000 rows.
- DVC pointer for `data/raw/reviews_sample.csv`; the dataset bytes remain outside Git.
- A DVC training stage configured in `pipelines/dvc.yaml` and `pipelines/params.yaml`.
- AraBERT training computes accuracy and macro-F1, logs them to MLflow, and writes a DVC metrics JSON.
- CI runs Ruff, pytest with an 80% coverage floor, DVC graph validation, Compose config validation, and an API image build.

## Verification

- MLflow and MinIO started locally; both mapped web endpoints returned HTTP 200.
- The DVC sample pointer is stable; `dvc dag` and `dvc repro --dry pipelines/dvc.yaml:train` resolve the training stage and its declared outputs.
- Ruff passed and pytest passed 20 tests at 98.6% coverage.
- The short training command is provided for local validation; use the full dataset and configured hyperparameters for the tracked experiment comparison.

## Reproduction

```cmd
venv\Scripts\python.exe -m pip install -e ".[dev,ml,data]"
docker compose -f docker/docker-compose.yml up -d
venv\Scripts\dvc.exe repro pipelines/dvc.yaml:train
```

Configure a shared DVC remote before collaboration, then run `dvc push` so another machine can fetch the sample with `dvc pull`.
