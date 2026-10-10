import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse

from arabic_sentiment.api.schemas import (
    BatchSentimentRequest,
    HealthResponse,
    MetadataResponse,
    SentimentRequest,
    SentimentResponse,
)
from arabic_sentiment.config import settings
from arabic_sentiment.logging_conf import logger
from arabic_sentiment.model import SentimentPredictor
from monitoring.prometheus_exporter import PrometheusMetricsExporter

predictor = None
metrics = PrometheusMetricsExporter()

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
    started = time.perf_counter()
    correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    response: Response = await call_next(request)
    metrics.record_http_request(
        request.url.path,
        (time.perf_counter() - started) * 1000,
    )
    metrics.increment_counter(f"http_{response.status_code}")
    response.headers["X-Request-ID"] = correlation_id
    return response

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> PlainTextResponse:
    """Expose request and drift metrics in Prometheus text format."""
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

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
            model_version=settings.VERSION,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("prediction_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal inference error")


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    """Expose the model identity needed by deployment and audit tooling."""
    backend = (
        "onnxruntime"
        if predictor is not None and getattr(predictor, "use_onnx", False)
        else "development-fallback"
    )
    return MetadataResponse(
        project=settings.PROJECT_NAME,
        model_name=settings.MODEL_NAME,
        model_version=settings.VERSION,
        backend=backend,
    )


@app.post("/predict/batch", response_model=list[SentimentResponse])
def predict_batch(payload: BatchSentimentRequest, request: Request):
    """Score a small synchronous batch using the already-loaded predictor."""
    if any(not text.strip() for text in payload.texts):
        raise HTTPException(status_code=422, detail="Texts cannot be empty")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference model is not ready")
    results = []
    for text in payload.texts:
        try:
            label, confidence, probs = predictor.predict(text)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("batch_prediction_failed", extra={"error": str(exc)})
            raise HTTPException(status_code=500, detail="Internal inference error") from exc
        results.append(
            SentimentResponse(
                text=text,
                label=label,
                confidence=confidence,
                probabilities=probs,
                model_version=settings.VERSION,
            )
        )
    logger.info(
        "batch_prediction_served",
        extra={"correlation_id": getattr(request.state, "correlation_id", "N/A"), "count": len(results)},
    )
    return results

def run():
    uvicorn.run("arabic_sentiment.api.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    run()
