import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_service_module(monkeypatch):
    api_options = {}
    service_options = {}

    def decorator(options):
        def apply(target):
            return target

        return apply

    fake_bentoml = SimpleNamespace(
        service=lambda **kwargs: (service_options.update(kwargs) or decorator(kwargs)),
        api=lambda **kwargs: (api_options.update(kwargs) or decorator(kwargs)),
    )

    class FakePredictor:
        def predict(self, text):
            return "positive", 0.8, {"positive": 0.8}

    monkeypatch.setitem(sys.modules, "bentoml", fake_bentoml)
    monkeypatch.setitem(
        sys.modules,
        "arabic_sentiment.model",
        SimpleNamespace(SentimentPredictor=FakePredictor),
    )
    service_path = Path(__file__).parents[1] / "serving" / "service.py"
    spec = importlib.util.spec_from_file_location("test_bento_service", service_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, api_options, service_options


def test_bento_service_enables_bounded_adaptive_batching(monkeypatch):
    module, api_options, service_options = load_service_module(monkeypatch)
    service = module.ArabicSentimentService()

    predictions = service.predict(["first", "second"])

    assert api_options == {
        "batchable": True,
        "max_batch_size": 32,
        "max_latency_ms": 100,
    }
    assert service_options["name"] == "arabic-sentiment"
    assert [item["label"] for item in predictions] == ["positive", "positive"]


def test_bento_service_rejects_blank_text(monkeypatch):
    module, _, _ = load_service_module(monkeypatch)
    service = module.ArabicSentimentService()

    with pytest.raises(ValueError, match="non-empty"):
        service.predict([" "])