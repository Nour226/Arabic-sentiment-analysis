import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Arabic Sentiment Analysis"
    VERSION: str = "0.1.0"
    MODEL_NAME: str = "aubmindlab/bert-base-arabertv02-twitter"
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "models/arabic-sentiment"))
    ONNX_MODEL_PATH: Path = Path(os.getenv("ONNX_MODEL_PATH", "models/model.onnx"))
    MAX_SEQ_LENGTH: int = 128
    BATCH_SIZE: int = 32

    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = "arabic-sentiment-analysis"

settings = Settings()