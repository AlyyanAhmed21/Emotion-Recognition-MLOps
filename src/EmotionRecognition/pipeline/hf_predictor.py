from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
from collections import deque, Counter
from PIL import Image

# --- The correct, pure PyTorch face detector ---
from facenet_pytorch import MTCNN

LOCAL_MODEL_PATH = "sota_model"

class HFPredictor:
    def __init__(self, smoothing_window=10, confidence_threshold=0.3, face_confidence_threshold=0.95):
        print(f"[PREDICTOR INFO] Loading model from local path: {LOCAL_MODEL_PATH}...")
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.face_confidence_threshold = face_confidence_threshold

        self.classes = list(self.model.config.id2label.values())
        self.confidence_threshold = confidence_threshold
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"
        print("[PREDICTOR INFO] Predictor initialized successfully.")
        
        if self.device == 'cuda':
            print("[PREDICTOR INFO] CUDA GPU detected. Running in high-performance mode.")
            self.model.to(self.device)
        else:
            print("[PREDICTOR INFO] No CUDA GPU detected. Running on CPU.")

    def process_frame(self, frame):
        """
        Processes a single frame: detects faces, predicts emotions, and draws annotations.
        """
        if frame is None: return frame, {}

        annotated_frame = frame.copy()
        all_probabilities = {}

        # The facenet-pytorch detector expects a PIL Image
        pil_frame = Image.fromarray(frame)
        
        # --- THIS IS THE FIX ---
        # The method is called .detect(), not .detect_faces()
        boxes, probs = self.face_detector.detect(pil_frame)
        # --- END FIX ---

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < self.face_confidence_threshold:
                    continue

                x1, y1, x2, y2 = [int(coord) for coord in box]
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0: continue
                
                face_roi = frame[y1:y2, x1:x2]
            
                if face_roi.size > 0:
                    pil_image = Image.fromarray(face_roi)
                    inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                    
                    predictions = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                    pred_index = np.argmax(predictions)
                    confidence = predictions[pred_index]

                    if confidence > self.confidence_threshold:
                        self.recent_predictions.append(pred_index)
                    if self.recent_predictions:
                        most_common_pred = Counter(self.recent_predictions).most_common(1)[0][0]
                        self.stable_prediction = self.classes[most_common_pred]
                    
                    GREEN = (0, 255, 0); BLACK = (0, 0, 0); FONT = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"{self.stable_prediction} ({confidence*100:.1f}%)"
                    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, 0.8, 2)
                    
                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 10), (x1 + text_width + 10, y1), GREEN, cv2.FILLED)
                    cv2.putText(annotated_frame, text, (x1 + 5, y1 - 5), FONT, 0.8, BLACK, 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), GREEN, 3)
                    
                    all_probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
        
        return annotated_frame, all_probabilities