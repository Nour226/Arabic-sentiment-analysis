"""Locust workload for the FastAPI or BentoML sentiment endpoint."""

import os

from locust import HttpUser, between, task


class SentimentLoadUser(HttpUser):
    wait_time = between(0.2, 1.0)
    use_bentoml = os.getenv("LOCUST_BENTOML", "0").lower() in {"1", "true", "yes"}

    @task
    def predict(self) -> None:
        text = "المنتج رائع والتوصيل سريع"
        payload = {"texts": [text]} if self.use_bentoml else {"text": text}
        with self.client.post(
            "/predict", json=payload, name="POST /predict", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected HTTP {response.status_code}")