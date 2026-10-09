import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from arabic_sentiment.config import settings

class SentimentPredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path or settings.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        ).to(self.device)
        self.model.eval() # Passer en mode évaluation
        self.labels = ["negative", "neutral", "positive"]

    def predict_one(self, text: str) -> dict:
        """Prédit le sentiment d'un texte unique en arabe."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_SEQ_LENGTH,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        best_idx = int(np.argmax(probs))

        return {
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "model_version": settings.MODEL_VERSION
        }