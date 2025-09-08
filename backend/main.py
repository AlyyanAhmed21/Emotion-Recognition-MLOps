from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
import cv2
import numpy as np
import base64

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.EmotionRecognition.pipeline.prediction import HFPredictor

# Define a Pydantic model for the incoming base64 image
class ImagePayload(BaseModel):
    image: str # e.g., "data:image/jpeg;base64,..."

# --- INITIALIZE APP AND PREDICTOR ---
app = FastAPI(title="Emotion Recognition API")
predictor = HFPredictor()
print("[BACKEND INFO] Hugging Face Predictor initialized successfully.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- THE ONLY ENDPOINT WE NEED ---
@app.post("/predict/frame")
async def predict_frame(payload: ImagePayload):
    """Receives a single base64 encoded frame and returns the annotated version."""
    try:
        # Decode the base64 string
        img_data = base64.b64decode(payload.image.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use the simple, single-frame predictor
            annotated_frame, probabilities = predictor.predict_single_frame(frame_rgb)

            if annotated_frame is not None:
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "annotated_image": f"data:image/jpeg;base64,{img_str}",
                    "probabilities": probabilities or {}
                }
        return {"error": "Invalid frame received"}, 400
    except Exception as e:
        print(f"[BACKEND ERROR] /predict/frame: {e}")
        return {"error": "Server failed to process frame"}, 500