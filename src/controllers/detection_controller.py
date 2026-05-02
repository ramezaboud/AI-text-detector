from fastapi import HTTPException
import langdetect

from src.helpers.config import settings
from src.models.ml.detector import detector
from src.models.schemas.detection import DetectionRequest, DetectionResponse

# Languages supported by our model
SUPPORTED_LANGUAGES = {"ar", "en"}

def detect_text(request: DetectionRequest) -> DetectionResponse:
    # 0. Clean newlines / extra whitespace
    text = request.text.replace("\n", " ").replace("\r", " ").strip()
    
    # 1. Word count validation
    words = text.split()
    word_count = len(words)
    
    if word_count < settings.MIN_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text is too short. Please provide at least {settings.MIN_WORDS} words (got {word_count})."
        )
        
    # 2. Language Detection — reject unsupported languages
    try:
        lang = langdetect.detect(text)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not detect the language of the provided text.")
    
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang}'. This model only supports Arabic (ar) and English (en)."
        )
        
    # 3. Model Inference — use chunking for long texts
    try:
        if word_count > settings.CHUNK_SIZE:
            prediction = detector.predict_chunked(text)
        else:
            prediction = detector.predict(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
        
    return DetectionResponse(
        verdict=prediction["verdict"],
        confidenceScore=prediction["confidenceScore"],
    )
