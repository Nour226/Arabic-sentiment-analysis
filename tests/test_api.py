from fastapi.testclient import TestClient

from src.arabic_sentiment.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_metadata_endpoint() -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["labels"] == ["negative", "neutral", "positive"]


def test_empty_text_is_rejected() -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_prediction_contract() -> None:
    response = client.post("/predict", json={"text": "هذا المنتج ممتاز ورائع"})
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        data = response.json()
        assert data["label"] in {"negative", "neutral", "positive"}
        assert 0 <= data["confidence"] <= 1
        assert set(data["probabilities"]) == {"negative", "neutral", "positive"}