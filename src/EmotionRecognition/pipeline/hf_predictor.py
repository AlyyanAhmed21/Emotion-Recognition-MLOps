from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
# --- MODIFIED: Import MTCNN from the new PyTorch-native library ---
from facenet_pytorch import MTCNN 
from collections import deque, Counter
from PIL import Image

LOCAL_MODEL_PATH = "sota_model"

class HFPredictor:
    def __init__(self, smoothing_window=10, confidence_threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PREDICTOR INFO] Using device: {self.device}")
        
        print(f"[PREDICTOR INFO] Loading model from local path: {LOCAL_MODEL_PATH}...")
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)

        self.model.to(self.device)
        self.model.eval()

        # --- MODIFIED: Initialize the facenet-pytorch MTCNN ---
        # This is much faster as it runs the face detection on the GPU as well.
        # keep_all=True ensures we detect all faces, not just the most probable one.
        self.face_detector = MTCNN(keep_all=True, device=self.device)
        
        self.classes = list(self.model.config.id2label.values())
        self.confidence_threshold = confidence_threshold
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"
        print("[PREDICTOR INFO] Predictor initialized successfully.")

    def process_frame(self, frame):
        """
        Processes a single frame. This function is now used for ALL predictions
        (live, image, and video) to ensure consistency.
        """
        if frame is None: return frame, {}

        annotated_frame = frame.copy()
        all_probabilities = {}

        # --- MODIFIED: The face detection logic is now different ---
        # The new detector returns boxes directly. We check if any were found.
        boxes, _ = self.face_detector.detect(frame)
        
        if boxes is not None:
            for box in boxes:
                # The box format is [x1, y1, x2, y2]. We convert it to (x, y, w, h).
                x, y, x2, y2 = [int(coord) for coord in box]
                width = x2 - x
                height = y2 - y
                
                x, y = max(0, x), max(0, y)
                face_roi = frame[y:y+height, x:x+width]
                
                if face_roi.size > 0:
                    pil_image = Image.fromarray(face_roi)
                    inputs = self.processor(images=pil_image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    predictions = probs[0].cpu().numpy()
                    
                    pred_index = np.argmax(predictions)
                    confidence = predictions[pred_index]

                    if len(self.recent_predictions) > 0:
                        if confidence > self.confidence_threshold:
                            self.recent_predictions.append(pred_index)
                        if not self.recent_predictions:
                            display_emotion = "---"
                        else:
                            most_common_pred = Counter(self.recent_predictions).most_common(1)[0][0]
                            display_emotion = self.classes[most_common_pred]
                    else:
                        display_emotion = self.classes[pred_index]
                    
                    if len(self.recent_predictions) == 0:
                        self.recent_predictions.clear()

                    text = f"{display_emotion} ({confidence*100:.1f}%)"
                    
                    GREEN = (0, 255, 0); BLACK = (0, 0, 0); FONT = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, 0.8, 2)
                    
                    cv2.rectangle(annotated_frame, (x, y - text_height - baseline - 10), (x + text_width + 10, y), GREEN, cv2.FILLED)
                    cv2.putText(annotated_frame, text, (x + 5, y - 5), FONT, 0.8, BLACK, 2)
                    cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), GREEN, 3)
                    
                    all_probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
        
        return annotated_frame, all_probabilities