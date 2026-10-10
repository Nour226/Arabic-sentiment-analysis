
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=2, description="Arabic product review text")
    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "المنتج ممتاز وسريع التوصيل"}]
        }
    }

class SentimentResponse(BaseModel):
    text: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool