import sys
from types import SimpleNamespace

import numpy as np
import pytest

from arabic_sentiment.model import SentimentPredictor


class FakeTokenizer:
	def __call__(self, text, **kwargs):
		return {
			"input_ids": np.array([[1, 2]], dtype=np.int32),
			"attention_mask": np.array([[1, 1]], dtype=np.int32),
			"token_type_ids": np.array([[0, 0]], dtype=np.int32),
		}


def install_tokenizer(monkeypatch):
	monkeypatch.setitem(
		sys.modules,
		"transformers",
		SimpleNamespace(
			AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: FakeTokenizer())
		),
	)


def test_fallback_predictor_returns_valid_probabilities(tmp_path, monkeypatch):
	install_tokenizer(monkeypatch)
	predictor = SentimentPredictor(tmp_path / "missing.onnx")

	label, confidence, probabilities = predictor.predict("نص عربي")

	assert label == "positive"
	assert confidence == 0.8
	assert sum(probabilities.values()) == pytest.approx(1.0)


def test_predict_rejects_blank_text(tmp_path, monkeypatch):
	install_tokenizer(monkeypatch)
	predictor = SentimentPredictor(tmp_path / "missing.onnx")

	with pytest.raises(ValueError, match="cannot be empty"):
		predictor.predict("  ")


def test_onnx_predictor_runs_session_and_converts_inputs(tmp_path, monkeypatch):
	import onnxruntime

	model_path = tmp_path / "model.onnx"
	model_path.touch()
	install_tokenizer(monkeypatch)
	captured = {}

	class FakeSession:
		def run(self, _, inputs):
			captured.update(inputs)
			return [np.array([[0.0, 0.0, 4.0]])]

	monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_: FakeSession())
	predictor = SentimentPredictor(model_path)

	label, confidence, probabilities = predictor.predict("نص عربي")

	assert label == "positive"
	assert confidence > 0.9
	assert set(captured) == {"input_ids", "attention_mask", "token_type_ids"}
	assert captured["input_ids"].dtype == np.int64
	assert probabilities["positive"] == confidence


def test_onnx_predictor_handles_outputs_without_token_type_ids(tmp_path, monkeypatch):
	import onnxruntime

	model_path = tmp_path / "model.onnx"
	model_path.touch()
	monkeypatch.setitem(
		sys.modules,
		"transformers",
		SimpleNamespace(
			AutoTokenizer=SimpleNamespace(
				from_pretrained=lambda _: lambda *args, **kwargs: {
					"input_ids": np.array([[1]], dtype=np.int32),
					"attention_mask": np.array([[1]], dtype=np.int32),
				}
			)
		),
	)

	class FakeSession:
		def run(self, _, inputs):
			assert set(inputs) == {"input_ids", "attention_mask"}
			return [np.array([[3.0, 0.0, 0.0]])]

	monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_: FakeSession())
	predictor = SentimentPredictor(model_path)

	assert predictor.predict("test")[0] == "negative"
