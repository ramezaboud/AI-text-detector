import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.helpers.config import settings
from src.models.ml.detector import detector
from src.routes.detection_routes import router as detection_router

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Text Detector API",
    description="API for detecting whether a given Arabic or English text is AI-generated or human-written based on XLM-RoBERTa.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(detection_router, prefix="/api", tags=["Detection"])

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up... Loading ML model.")
    detector.load()
    logger.info("Startup complete.")

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)

# text_detector\Scripts\activate