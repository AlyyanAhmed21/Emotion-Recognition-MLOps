from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN 
from collections import deque, Counter
from PIL import Image

LOCAL_MODEL_PATH = "sota_model"

class HFPredictor:
    def __init__(self, smoothing_window=15, box_smoothing_factor=0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PREDICTOR INFO] Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

        # --- THIS IS THE FIX (Part 1): Add a confidence threshold to the detector ---
        # We will only consider detections where MTCNN is at least 95% confident it's a face.
        self.face_detector = MTCNN(keep_all=True, device=self.device, thresholds=[0.7, 0.8, 0.95])
        
        self.classes = list(self.model.config.id2label.values())
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.smoothing_factor = box_smoothing_factor
        self.last_known_box = None

        self.COLOR_MAP = {
            "angry": (0, 0, 255), "disgust": (0, 100, 0), "fear": (130, 0, 75),
            "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (0, 165, 255),
            "neutral": (255, 255, 255)
        }
        print("[PREDICTOR INFO] Predictor initialized successfully.")

    def reset_smoother(self):
        self.recent_predictions.clear()
        self.last_known_box = None
        print("[INFO] Prediction smoother and box tracker have been reset.")

    def _draw_annotations(self, frame, box, emotion):
        # This function is correct and remains unchanged
        x, y, x2, y2 = [int(coord) for coord in box]
        text = emotion.capitalize()
        color = self.COLOR_MAP.get(emotion, (255, 255, 255))
        # ... (rest of the drawing logic is the same)
        BLACK = (0, 0, 0); FONT = cv2.FONT_HERSHEY_DUPLEX; FONT_SCALE = 0.6; THICKNESS = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        text_y = y - 10
        cv2.rectangle(frame, (x, text_y - text_height - baseline), (x + text_width + 10, text_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, text, (x + 5, text_y), FONT, FONT_SCALE, BLACK, THICKNESS, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
        return frame

    def _predict_emotions(self, frame, box):
        # This function is correct and remains unchanged
        x, y, x2, y2 = [int(coord) for coord in box]
        face_roi = frame[y:y2, x:x2]
        if face_roi.size > 0:
            pil_image = Image.fromarray(face_roi)
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = probs[0].cpu().numpy()
            pred_index = np.argmax(predictions)
            emotion = self.classes[pred_index]
            all_probs_dict = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
            return emotion, all_probs_dict
        return None, {}

    def predict_static(self, frame):
        """
        Processes a single frame for videos or image uploads. No smoothing is applied.
        """
        if frame is None: return frame, {}
        annotated_frame = frame.copy()
        
        # --- THIS IS THE FIX (Part 2): MTCNN returns boxes and confidence probabilities ---
        boxes, probs = self.face_detector.detect(annotated_frame)
        
        probabilities = {}
        # We check if boxes is not None AND if any of the detected boxes have a high probability
        if boxes is not None:
            # Loop through each detected face and its confidence
            for box, prob in zip(boxes, probs):
                # Only process faces where the detector is >95% confident
                if prob > 0.95:
                    emotion, current_probs = self._predict_emotions(annotated_frame, box)
                    if emotion:
                        annotated_frame = self._draw_annotations(annotated_frame, box, emotion)
                        probabilities = current_probs # Panel will show the last valid face
            return annotated_frame, probabilities

        # If no high-confidence faces are found, we just return the original frame
        return annotated_frame, {}

    def predict_smoothed(self, frame):
        """
        Processes frames for the live feed, using smoothing for a single person.
        """
        # This function is designed for a single person and remains correct for that use case.
        if frame is None: return frame, {}
        annotated_frame = frame.copy()
        
        boxes_raw, probs = self.face_detector.detect(annotated_frame)
        
        # We'll only track if there's at least one high-confidence detection
        high_conf_boxes = []
        if boxes_raw is not None:
            for box, prob in zip(boxes_raw, probs):
                if prob > 0.95:
                    high_conf_boxes.append(box)

        if high_conf_boxes:
            detected_box = high_conf_boxes[0] # Track the first high-confidence face
            if self.last_known_box is None:
                self.last_known_box = detected_box
            else:
                self.last_known_box = (self.smoothing_factor * detected_box + (1 - self.smoothing_factor) * self.last_known_box)
        
        probabilities = {}
        if self.last_known_box is not None:
            emotion, probabilities = self._predict_emotions(annotated_frame, self.last_known_box)
            if emotion:
                self.recent_predictions.append(emotion)
            if self.recent_predictions:
                display_emotion = Counter(self.recent_predictions).most_common(1)[0][0]
                annotated_frame = self._draw_annotations(annotated_frame, self.last_known_box, display_emotion)

        return annotated_frame, probabilities