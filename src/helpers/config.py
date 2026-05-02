import torch
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    ARABIC_DATA_PATH: str = "data/ar/final_arabic_ready.csv"
    ENGLISH_DATA_PATH: str = "data/en/final_english_ready.csv"
    PIPELINE_DIR: str = "pipeline/best_model"
    LOGS_DIR: str = "logs"

    # API
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # Model
    # Currently trained model is xlm-roberta-base, saved in pipeline/best_model
    MODEL_NAME: str = "xlm-roberta-base"

    # Training
    BATCH_SIZE: int = 8
    MAX_EPOCHS: int = 5
    LR: float = 2e-5
    SAMPLE_SIZE: int = 100000
    RANDOM_SEED: int = 42

    # Text — ثوابت مش بتتغير
    MIN_WORDS:      int = 20
    MAX_TOKENS:     int = 512
    CHUNK_SIZE:     int = 400   # عدد الكلمات في كل chunk
    CHUNK_OVERLAP:  int = 100   # كلمات متداخلة بين كل chunk والتاني
    STRIDE:         int = 128

    # Auto-detected
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"


settings = Settings()