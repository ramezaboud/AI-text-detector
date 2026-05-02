from pydantic import BaseModel, Field

class DetectionRequest(BaseModel):
    text: str = Field(..., description="The text to analyze.", min_length=1)

class DetectionResponse(BaseModel):
    verdict: str = Field(..., description="'real' or 'ai'")
    confidenceScore: float = Field(..., description="Confidence score from 0 to 100")

class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    model_loaded: bool = Field(default=False)
