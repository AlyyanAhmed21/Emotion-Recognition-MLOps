from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
from mtcnn import MTCNN
from collections import deque, Counter
from PIL import Image

LOCAL_MODEL_PATH = "sota_model"

class HFPredictor:
    def __init__(self, smoothing_window=10, confidence_threshold=0.3):
        print(f"[PREDICTOR INFO] Loading model from local path: {LOCAL_MODEL_PATH}...")
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
        self.face_detector = MTCNN()
        self.classes = list(self.model.config.id2label.values())
        self.confidence_threshold = confidence_threshold
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"
        print("[PREDICTOR INFO] Predictor initialized successfully.")


    def process_frame(self, frame):
        """
        Processes a single frame: flips it for a mirror effect, detects faces, 
        predicts emotions, and draws professional annotations.
        """
        if frame is None: return frame, {}

        # --- MIRROR FIX: Flip the frame FIRST! ---
        # This ensures detection and drawing happen in the same coordinate space the user sees.
        frame = cv2.flip(frame, 1)
        annotated_frame = frame.copy()
        # --- END FIX ---
        
        all_probabilities = {}
        faces = self.face_detector.detect_faces(frame)
        
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+height, x:x+width]
            
            if face_roi.size > 0:
                pil_image = Image.fromarray(face_roi)
                inputs = self.processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predictions = probs[0].numpy()
                pred_index = np.argmax(predictions)
                
                # Use temporal smoothing for the displayed label
                confidence = predictions[pred_index]
                if confidence > self.confidence_threshold:
                    self.recent_predictions.append(pred_index)
                if self.recent_predictions:
                    most_common_pred = Counter(self.recent_predictions).most_common(1)[0][0]
                    self.stable_prediction = self.classes[most_common_pred]
                
                # --- PROFESSIONAL DRAWING LOGIC ---
                GREEN = (0, 255, 0)
                BLACK = (0, 0, 0)
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{self.stable_prediction} ({confidence*100:.1f}%)"
                
                (text_width, text_height), baseline = cv2.getTextSize(text, FONT, 0.8, 2)
                
                cv2.rectangle(annotated_frame, (x, y - text_height - baseline - 10), (x + text_width + 10, y), GREEN, cv2.FILLED)
                cv2.putText(annotated_frame, text, (x + 5, y - 5), FONT, 0.8, BLACK, 2)
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), GREEN, 3)
                
                all_probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
        
        return annotated_frame, all_probabilities