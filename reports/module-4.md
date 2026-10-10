# Module 4: ONNX INT8 Optimization and Benchmarking

## Delivered

- ONNX Runtime dynamic signed-INT8 quantization with per-channel weights.
- Benchmark harness comparing FP32 and INT8 model size, mean/p50/p95 latency, and labeled accuracy.
- Machine-readable benchmark JSON output under ignored `artifacts/optimization/`.
- ONNX optimization extra declared separately from development dependencies.

## Verification

- Quantization CLI logic is covered by unit tests.
- A real ONNX Runtime integration test quantized a tiny MatMul graph and verified inference output.
- Full project suite: 30 tests passed at 94.21% coverage; Ruff passed across source, tests, serving, loadtest, and optimization.
- The scripts fail clearly when the expected FP32 ONNX model or dataset is absent.
- No full AraBERT checkpoint or `models/model.onnx` exists locally, so model-level accuracy/latency/size results are not claimed. The integration test validates the quantization toolchain only.

## Reproduction

After a completed training run and ONNX export:

```cmd
python -m pip install -e ".[optimization]"
python optimization/quantize_onnx.py --input models/model.onnx --output models/model_int8.onnx
python optimization/benchmark.py --fp32 models/model.onnx --int8 models/model_int8.onnx --data data/raw/reviews_sample.csv --max-rows 1000
```

Model binaries are ignored by Git; commit the report, not the generated weights.
