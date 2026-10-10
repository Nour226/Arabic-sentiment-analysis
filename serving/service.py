"""BentoML service with adaptive request batching for sentiment inference."""

from __future__ import annotations

import bentoml

from arabic_sentiment.model import SentimentPredictor


@bentoml.service(
    name="arabic-sentiment",
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class ArabicSentimentService:
    def __init__(self) -> None:
        self.predictor = SentimentPredictor()

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=100)
    def predict(self, texts: list[str]) -> list[dict[str, object]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Each request must contain non-empty text")
        results = []
        for text in texts:
            label, confidence, probabilities = self.predictor.predict(text)
            results.append(
                {
                    "text": text,
                    "label": label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            )
        return results