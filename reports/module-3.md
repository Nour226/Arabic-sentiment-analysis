# Module 3: Production Serving and Load Testing

## Delivered

- BentoML service with adaptive batching enabled, a maximum batch size of 32, and a 100 ms batching latency bound.
- Offline CSV/Parquet batch scorer that writes prediction labels, confidence, and class probabilities to Parquet.
- Locust workload for FastAPI and BentoML request formats.
- Optional Nginx canary profile that routes 5% of client addresses to a separate API instance.

## Verification

- BentoML service imported and served successfully; two-item batch request returned HTTP 200.
- Nginx canary entry point returned HTTP 200 from `/health` with stable and canary containers running.
- Batch scorer wrote 1,000 rows to Parquet with all expected prediction columns.
- `ruff check src tests serving loadtest` passed; focused batch/serving tests passed.
- FastAPI Locust run: 30 seconds, 10 users, spawn rate 2; 428 requests, 0 failures, 51.97 ms average, 237.69 ms max, 14.91 requests/sec.
- Metrics above are from the development fallback predictor because no exported `models/model.onnx` artifact was present. They are not representative of a trained model.

## Reproduction

```cmd
python -m pip install -e ".[serving,loadtest]"
python -m arabic_sentiment.batch --data data/raw/reviews_sample.csv --max-rows 1000 --output artifacts/batch/predictions.parquet
locust -f loadtest/locustfile.py --headless -u 10 -r 2 --run-time 30s --host http://localhost:8000
```

The canary stack is opt-in with the Compose `canary` profile and listens on port 8080.