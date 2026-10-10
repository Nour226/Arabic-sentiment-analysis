import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from optimization import benchmark, quantize_onnx


def test_quantize_model_calls_dynamic_int8_and_returns_sizes(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    target = tmp_path / "model_int8.onnx"
    source.write_bytes(b"float-model")
    calls = {}

    def fake_quantize_dynamic(**kwargs):
        calls.update(kwargs)
        Path(kwargs["model_output"]).write_bytes(b"int8")

    fake_quantization = SimpleNamespace(
        QuantType=SimpleNamespace(QInt8="signed-int8"),
        quantize_dynamic=fake_quantize_dynamic,
    )
    monkeypatch.setitem(
        sys.modules, "onnxruntime.quantization", fake_quantization
    )

    original_size, quantized_size = quantize_onnx.quantize_model(source, target)

    assert original_size == len(b"float-model")
    assert quantized_size == len(b"int8")
    assert calls["weight_type"] == "signed-int8"
    assert calls["per_channel"] is True


def test_quantize_model_requires_existing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="FP32 ONNX model not found"):
        quantize_onnx.quantize_model(tmp_path / "missing.onnx", tmp_path / "out.onnx")


class FakePredictor:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, text):
        return "positive", 0.9, {}


def test_benchmark_models_reports_accuracy_latency_and_size(tmp_path):
    fp32_path = tmp_path / "model.onnx"
    int8_path = tmp_path / "model_int8.onnx"
    fp32_path.write_bytes(b"float32")
    int8_path.write_bytes(b"int8")
    data_path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {"text": ["first", "second"], "label": ["positive", "negative"]}
    ).to_csv(data_path, index=False)

    result = benchmark.benchmark_models(
        fp32_path,
        int8_path,
        data_path,
        predictor_factory=FakePredictor,
    )

    assert result["size_reduction_percent"] == pytest.approx((1 - 4 / 7) * 100)
    assert result["fp32"]["accuracy"] == 0.5
    assert result["int8"]["rows"] == 2
    assert result["int8"]["mean_latency_ms"] >= 0


def test_read_rows_rejects_invalid_limits_and_schema(tmp_path):
    data_path = tmp_path / "reviews.csv"
    pd.DataFrame({"review": ["text"]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="max_rows must be positive"):
        benchmark.read_rows(data_path, 0)
    with pytest.raises(ValueError, match="text column"):
        benchmark.read_rows(data_path, 1)


def test_quantize_model_with_onnx_runtime(tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    weight = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "quantization-test",
        [input_info],
        [output_info],
        [weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    source = tmp_path / "float.onnx"
    target = tmp_path / "int8.onnx"
    onnx.save(model, source)

    quantize_onnx.quantize_model(source, target)
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": np.array([[1, 2, 3, 4]], dtype=np.float32)})

    assert target.is_file()
    assert np.allclose(result[0], [[1, 2, 3, 4]], atol=0.1)