import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from arabic_sentiment.api.schemas import PredictionRequest, PredictionResponse
from arabic_sentiment.model import SentimentPredictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    # Chargement du modèle au démarrage de l'application
    predictor = SentimentPredictor()
    yield

app = FastAPI(
    title="Arabic Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Point d'entrée de santé utilisé par Docker et Kubernetes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(payload: PredictionRequest):
    """Effectue une prédiction de sentiment sur du texte en arabe."""
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Le texte ne peut pas être vide")
    result = predictor.predict_one(payload.text)
    return result