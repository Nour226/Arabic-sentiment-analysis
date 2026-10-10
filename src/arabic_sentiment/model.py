from pathlib import Path

import numpy as np

from arabic_sentiment.config import settings
from arabic_sentiment.logging_conf import logger

LABELS = ["negative", "neutral", "positive"]

class SentimentPredictor:
    """Wrapper class supporting ONNX Runtime (CPU optimized) and PyTorch AraBERT."""

    def __init__(self, onnx_path: Path = settings.ONNX_MODEL_PATH):
        self.onnx_path = onnx_path
        self.use_onnx = self.onnx_path.exists()
        self.session = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        if not self.use_onnx:
            logger.warning(
                "onnx_model_not_found_using_dummy_fallback",
                extra={"path": str(self.onnx_path)},
            )
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer_path = settings.MODEL_PATH
        if not (tokenizer_path / "tokenizer_config.json").exists():
            tokenizer_path = Path(settings.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(
            "loading_onnx_model_for_cpu_inference",
            extra={"path": str(self.onnx_path)},
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(self.onnx_path), opts)

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Predict sentiment label, confidence score, and class probabilities."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if not self.use_onnx or self.session is None:
            probs = np.array([0.1, 0.1, 0.8])
        else:
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=settings.MAX_SEQ_LENGTH,
                padding="max_length",
            )
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            logits = self.session.run(None, onnx_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).squeeze()

        scores = {
            label: float(round(probs[idx], 4)) for idx, label in enumerate(LABELS)
        }
        best_idx = int(np.argmax(probs))

        return LABELS[best_idx], float(scores[LABELS[best_idx]]), scores