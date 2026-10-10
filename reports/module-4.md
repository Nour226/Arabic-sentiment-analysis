# Module 4: ONNX INT8 Optimization and Benchmarking

## Delivered

- ONNX Runtime dynamic signed-INT8 quantization with per-channel weights.
- A benchmark harness comparing FP32 and INT8 model size, mean/p50/p95 latency, and labeled accuracy.
- Machine-readable benchmark JSON output under ignored `artifacts/optimization/`.
- A dedicated `optimization` package marker so the scripts and tests import consistently in CI and locally.
- The ONNX optimization dependency declared separately from development dependencies.

## Verification

- Quantization CLI logic is covered by unit tests.
- A real ONNX Runtime integration test quantized a tiny MatMul graph and verified inference output.
- The full project suite is gated at 80% coverage; Ruff covers source, tests, serving, loadtest, and optimization.
- The scripts fail clearly when the expected FP32 ONNX model or dataset is absent.
- The benchmark command records the model-level accuracy, latency, and size table when the exported FP32 and INT8 artifacts are supplied.

## Reproduction

After a completed training run and ONNX export:

```cmd
python -m pip install -e ".[optimization]"
python optimization/quantize_onnx.py --input models/model.onnx --output models/model_int8.onnx
python optimization/benchmark.py --fp32 models/model.onnx --int8 models/model_int8.onnx --data data/raw/reviews_sample.csv --max-rows 1000
```

Keep model binaries in the configured model registry or artifact store and commit the generated benchmark report. The release tag for this module is `v0.4.0`.
