import os 
from pydantic_settings import BaseSettings 

class Settings(BaseSettings): 
    PROJECT_NAME: str = "Arabic Sentiment Analysis" 
    MODEL_NAME: str = "aubmindlab/bert-base-arabertv02" 
    # Modèle AraBERT officiel
    MAX_SEQ_LENGTH: int = 128 
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )
    MODEL_VERSION: str = "1.0.0"

settings = Settings()