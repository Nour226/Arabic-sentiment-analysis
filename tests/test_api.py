import pytest
from fastapi.testclient import TestClient

from arabic_sentiment.api import main


class FakePredictor:
    def predict(self, text):
        return "positive", 0.8, {
            "negative": 0.1,
            "neutral": 0.1,
            "positive": 0.8,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "predictor", FakePredictor())
    return TestClient(main.app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_success(client):
    payload = {"text": "المنتج رائع جداً"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["positive", "neutral", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_version"]
    assert response.headers["X-Request-ID"]


def test_metadata_and_batch_endpoints(client):
    assert client.get("/metadata").status_code == 200
    response = client.post("/predict/batch", json={"texts": ["first", "second"]})
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_predict_validation_rejection(client):
    # Rejects empty text with 422
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_correlation_id_is_preserved(client):
    response = client.get("/health", headers={"X-Request-ID": "request-123"})

    assert response.headers["X-Request-ID"] == "request-123"


def test_predict_returns_unavailable_without_model(client, monkeypatch):
    monkeypatch.setattr(main, "predictor", None)

    response = client.post("/predict", json={"text": "review"})

    assert response.status_code == 503


def test_predict_maps_inference_failure_to_500(client, monkeypatch):
    class FailingPredictor:
        def predict(self, text):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(main, "predictor", FailingPredictor())

    response = client.post("/predict", json={"text": "review"})

    assert response.status_code == 500


def test_lifespan_loads_predictor(monkeypatch):
    monkeypatch.setattr(main, "predictor", None)
    monkeypatch.setattr(main, "SentimentPredictor", FakePredictor)

    with TestClient(main.app) as test_client:
        response = test_client.get("/health")

    assert response.json()["model_loaded"] is True


def test_run_starts_uvicorn(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))

    main.run()

    assert captured["port"] == 8000
    assert captured["host"] == "0.0.0.0"
