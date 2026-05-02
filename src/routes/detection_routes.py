from fastapi import APIRouter
from src.models.schemas.detection import DetectionRequest, DetectionResponse, HealthResponse
from src.controllers.detection_controller import detect_text
from src.models.ml.detector import detector

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify server and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=detector.model is not None
    )

@router.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """Detect if the provided text is AI-generated or Human-written."""
    return detect_text(request)
