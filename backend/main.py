from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import cv2
import numpy as np
import base64
import json

# This is crucial for allowing the backend to import our Predictor class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.EmotionRecognition.pipeline.prediction import Predictor

# --- CONFIGURATION ---
MODEL_PATH = "artifacts/training/model.keras"
CLASSES = CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- INITIALIZE APP AND PREDICTOR ---
app = FastAPI(title="Emotion Recognition API")
predictor = Predictor(model_path=MODEL_PATH, classes=CLASSES)
print("[BACKEND INFO] Predictor initialized successfully.")

# Allow Cross-Origin Resource Sharing (CORS) for our React frontend (running on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # The origin of our React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Emotion Recognition API is running."}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Endpoint for single image prediction."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # We need a new, simplified predictor method for this
    annotated_frame, probabilities = predictor.predict_single_frame(frame_rgb)
    
    # Encode the output image to send back via JSON
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')

    return {"annotated_image": f"data:image/jpeg;base64,{img_str}", "probabilities": probabilities}


@app.websocket("/predict/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live, real-time prediction."""
    await websocket.accept()
    print("[BACKEND INFO] WebSocket connection established.")
    try:
        while True:
            data = await websocket.receive_text()
            # The data from the frontend will be a base64 encoded image string
            img_data = base64.b64decode(data.split(',')[1])
            
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use the predictor's live function with smoothing
                annotated_frame, probabilities, _ = predictor.predict_live(frame_rgb, [])

                if probabilities:
                    # Encode the annotated frame to send back
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    
                    response = {
                        "annotated_image": f"data:image/jpeg;base64,{img_str}",
                        "probabilities": probabilities
                    }
                    await websocket.send_json(response)
    except WebSocketDisconnect:
        print("[BACKEND INFO] WebSocket connection closed.")
    except Exception as e:
        print(f"[BACKEND ERROR] WebSocket Error: {e}")

# --- HELPER FUNCTION FOR PREDICTOR ---
# You will need to add the 'predict_single_frame' method to your Predictor class.
# I will provide this in the next step.