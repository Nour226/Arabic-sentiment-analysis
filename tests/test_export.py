import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arabic_sentiment import export


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


class FakeTokenizer:
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": FakeTensor([[1] * 8]),
            "attention_mask": FakeTensor([[1] * 8]),
        }


class FakeModel:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=FakeTensor([[0.2, 0.3, 0.5]]))


def install_model_fakes(monkeypatch, export_function=None):
    fake_torch = SimpleNamespace(
        onnx=SimpleNamespace(export=export_function), no_grad=nullcontext
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: FakeTokenizer()),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=lambda _: FakeModel()
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_export_creates_destination_and_exports_model(tmp_path, monkeypatch):
    calls = {}

    def fake_export(model, inputs, output_path, **kwargs):
        calls.update(kwargs)
        assert isinstance(inputs, tuple)
        Path(output_path).write_bytes(b"fake onnx")

    install_model_fakes(monkeypatch, fake_export)
    output_path = tmp_path / "nested" / "model.onnx"

    export.export_to_onnx(tmp_path / "checkpoint", output_path)

    assert output_path.read_bytes() == b"fake onnx"
    assert calls["opset_version"] == 14
    assert calls["output_names"] == ["logits"]


def test_verify_parity_accepts_matching_outputs(monkeypatch, tmp_path):
    import onnxruntime

    install_model_fakes(monkeypatch)
    monkeypatch.setattr(
        onnxruntime,
        "InferenceSession",
        lambda _: SimpleNamespace(
            run=lambda *_: [np.array([[0.2, 0.3, 0.5]], dtype=np.float32)]
        ),
    )

    export.verify_parity(tmp_path / "checkpoint", tmp_path / "model.onnx")


def test_verify_parity_rejects_mismatched_outputs(monkeypatch, tmp_path):
    import onnxruntime

    install_model_fakes(monkeypatch)
    monkeypatch.setattr(
        onnxruntime,
        "InferenceSession",
        lambda _: SimpleNamespace(run=lambda *_: [np.array([[9.0, 0.0, 0.0]])]),
    )

    with pytest.raises(AssertionError, match="Parity check failed"):
        export.verify_parity(tmp_path / "checkpoint", tmp_path / "model.onnx")