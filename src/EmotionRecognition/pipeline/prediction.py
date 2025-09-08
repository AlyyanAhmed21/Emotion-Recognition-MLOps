from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
from mtcnn import MTCNN
from collections import deque, Counter
from datetime import datetime
from PIL import Image

# Define the path to your local model directory
LOCAL_MODEL_PATH = "sota_model"

class HFPredictor:
    def __init__(self, smoothing_window=10, confidence_threshold=0.3):
        """
        Initializes the Hugging Face predictor by loading the SOTA model from a local directory.
        """
        print(f"[HF PREDICTOR INFO] Loading model from local path: {LOCAL_MODEL_PATH}...")
        
        # Load the processor and model from the local folder
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
        
        self.face_detector = MTCNN()
        # Get the class names directly from the model's configuration file (config.json)
        self.classes = list(self.model.config.id2label.values())
        
        # Set parameters for smoothing
        self.confidence_threshold = confidence_threshold
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"
        print("[HF PREDIDCTOR INFO] Predictor initialized successfully.")
        print(f"[HF PREDICTOR INFO] Model classes: {self.classes}")

    def predict_single_frame(self, frame):
        if frame is None: return None, None
        faces = self.face_detector.detect_faces(frame)
        annotated_frame = frame.copy() # Starts as a copy of the original
        probabilities = None
        
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+height, x:x+width]
            
            if face_roi.size > 0:
                # ... (prediction logic is the same)
                
                emotion = self.classes[pred_index]
                confidence = predictions[pred_index]
                
                # --- NEW DRAWING LOGIC ---
                # Define colors and fonts
                GREEN = (0, 255, 0)
                WHITE = (255, 255, 255)
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                
                text = f"{emotion} ({confidence*100:.1f}%)"
                
                # Get the size of the text to draw a background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(text, FONT, 1, 2)
                
                # Draw the text background rectangle
                cv2.rectangle(annotated_frame, 
                              (x, y - text_height - baseline - 10), 
                              (x + text_width + 10, y - baseline + 5), 
                              GREEN, 
                              cv2.FILLED)

                # Draw the text on top of the background
                cv2.putText(annotated_frame, text, (x + 5, y - baseline - 5), FONT, 1, WHITE, 2)

                # Draw a thicker bounding box
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), GREEN, 3)
                # --- END NEW DRAWING LOGIC ---

                probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
                
        return annotated_frame, probabilities

    def predict_live(self, frame, current_log=[]):
        """
        Processes a single frame from the live video stream with temporal smoothing.
        """
        if frame is None:
            return None, None, current_log

        faces = self.face_detector.detect_faces(frame)
        annotated_frame = frame.copy()
        probabilities = None

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
                confidence = predictions[pred_index]

                if confidence > self.confidence_threshold:
                    self.recent_predictions.append(pred_index)
                
                if self.recent_predictions:
                    most_common_pred = Counter(self.recent_predictions).most_common(1)[0][0]
                    self.stable_prediction = self.classes[most_common_pred]

                text = f"{self.stable_prediction} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                
                probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
        
        if probabilities:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (timestamp, self.stable_prediction)
            if not current_log or current_log[-1][1] != self.stable_prediction:
                current_log.append(log_entry)

        return annotated_frame, probabilities, current_log
    
    def process_websocket_frame(self, frame):
        if frame is None:
            return None, None

        faces = self.face_detector.detect_faces(frame)
        annotated_frame = frame.copy()
        probabilities = None

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

                emotion = self.classes[pred_index]
                confidence = predictions[pred_index]

                text = f"{emotion} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x, y),
                            (x+width, y+height), (0, 255, 0), 2)

                probabilities = {
                    self.classes[i]: float(predictions[i])
                    for i in range(len(self.classes))
                }

        # Always return something, even if no faces found
        if probabilities is None:
            probabilities = {}

        return annotated_frame, probabilities