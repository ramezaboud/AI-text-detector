import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

from src.helpers.config import settings

logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = settings.DEVICE
        self.model_dir = settings.PIPELINE_DIR

    def load(self):
        """Loads the model and tokenizer from the pipeline directory into memory."""
        logger.info(f"Loading tokenizer and model from {self.model_dir}...")
        try:
            # Load tokenizer from the original huggingface base model because the local 
            # tokenizer.json is incompatible across environments and sentencepiece.bpe.model is missing.
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            
            # Use dataparallel if multiple GPUs available, similarly to training
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def predict(self, text: str) -> dict:
        """Runs inference on a given text and returns prediction and confidence."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call load() first.")

        inputs = self.tokenizer(
            text,
            max_length=settings.MAX_TOKENS,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        confidence = probs.max().item()
        prediction_idx = probs.argmax(dim=-1).item()
        
        # Determine verdict based on training (0: human, 1: AI)
        is_ai = bool(prediction_idx == 1)
        verdict = "ai" if is_ai else "real"
        confidence_score = round(confidence * 100, 2)
        
        return {
            "verdict": verdict,
            "confidenceScore": confidence_score,
        }

    def predict_chunked(self, text: str) -> dict:
        """Splits long text into overlapping chunks, runs inference on each, and aggregates results."""
        words = text.split()
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        step = chunk_size - overlap

        # Build chunks with overlap
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            # Skip chunks that are too short (less than MIN_WORDS)
            if len(chunk.split()) >= settings.MIN_WORDS:
                chunks.append(chunk)

        if not chunks:
            return self.predict(text)

        # Run inference on each chunk
        results = [self.predict(chunk) for chunk in chunks]

        # Aggregate: majority vote for verdict, average for confidence
        ai_count = sum(1 for r in results if r["verdict"] == "ai")
        avg_confidence = sum(r["confidenceScore"] for r in results) / len(results)

        final_verdict = "ai" if ai_count > len(results) / 2 else "real"

        return {
            "verdict": final_verdict,
            "confidenceScore": round(avg_confidence, 2),
            "chunksAnalyzed": len(chunks),
        }

# Global detector instance to be loaded at startup
detector = AIDetector()
