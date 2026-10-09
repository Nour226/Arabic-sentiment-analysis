from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=2, description="Texte en arabe à analyser")
    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "المنتج رائع جدا وجد ممتاز"}]
        }
    }

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str