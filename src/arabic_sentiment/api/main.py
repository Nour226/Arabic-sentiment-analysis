import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status

from arabic_sentiment.api.schemas import (
    HealthResponse,
    SentimentRequest,
    SentimentResponse,
)
from arabic_sentiment.config import settings
from arabic_sentiment.logging_conf import logger
from arabic_sentiment.model import SentimentPredictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("starting_api_and_loading_model")
    predictor = SentimentPredictor()
    yield
    logger.info("shutting_down_api")

app = FastAPI(
    title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan
)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
    )

@app.post("/predict", response_model=SentimentResponse)
def predict(payload: SentimentRequest, request: Request):
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty",
        )
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not ready",
        )
    try:
        label, confidence, probs = predictor.predict(payload.text)
        logger.info(
            "prediction_served",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", "N/A"),
                "label": label,
                "confidence": confidence,
            },
        )
        return SentimentResponse(
            text=payload.text,
            label=label,
            confidence=confidence,
            probabilities=probs,
            version=settings.VERSION,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("prediction_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal inference error")

def run():
    uvicorn.run("arabic_sentiment.api.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    run()