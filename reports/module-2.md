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
- The DVC sample pointer is unchanged; `dvc dag` and `dvc repro --dry pipelines/dvc.yaml:train` resolve the expected stage. The training stage's model/metrics outputs remain pending a full completed run.
- Ruff passed and pytest passed 20 tests at 98.6% coverage.
- Completed small training runs were recorded in MLflow. The 16-row smoke run is diagnostic only and is not evidence of model quality. Larger CPU training jobs were still active during this run and are not reported as complete.

## Reproduction

```cmd
venv\Scripts\python.exe -m pip install -e ".[dev,ml,data]"
docker compose -f docker/docker-compose.yml up -d
venv\Scripts\dvc.exe repro pipelines/dvc.yaml:train
```

The sample pointer and metadata are committed locally. No DVC remote is configured, so another machine cannot fetch the sample until a DVC remote is configured and populated.
