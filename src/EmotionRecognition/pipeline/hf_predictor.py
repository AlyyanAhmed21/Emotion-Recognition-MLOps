from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN 
from collections import deque, Counter
from PIL import Image

LOCAL_MODEL_PATH = "sota_model"

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class HFPredictor:
    def __init__(self, smoothing_window=10, box_smoothing_factor=0.3, confirmation_frames=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PREDICTOR INFO] Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.classes = list(self.model.config.id2label.values())
        
        # State for multi-face IOU Tracker (for video processing)
        self.tracked_faces = {}
        self.next_face_id = 0
        self.smoothing_window = smoothing_window
        self.box_smoothing_factor = box_smoothing_factor
        self.confirmation_frames = confirmation_frames

        # --- THIS IS THE FIX: Restore the missing variables ---
        self.live_feed_smoother = deque(maxlen=smoothing_window)
        self.live_feed_box_smoother = None

        self.COLOR_MAP = {
            "angry": (0, 0, 255), "disgust": (0, 100, 0), "fear": (130, 0, 75),
            "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (0, 165, 255),
            "neutral": (255, 255, 255)
        }
        print("[PREDICTOR INFO] Predictor initialized successfully.")

    def reset_tracker(self):
        self.tracked_faces.clear()
        self.next_face_id = 0
        self.live_feed_smoother.clear()
        self.live_feed_box_smoother = None
        print("[INFO] All trackers have been reset.")

    def _draw_annotations(self, frame, box, emotion):
        x, y, x2, y2 = [int(coord) for coord in box]
        text = emotion.capitalize()
        color = self.COLOR_MAP.get(emotion, (255, 255, 255))
        BLACK = (0, 0, 0); FONT = cv2.FONT_HERSHEY_DUPLEX; FONT_SCALE = 0.6; THICKNESS = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        text_y = y - 10
        cv2.rectangle(frame, (x, text_y - text_height - baseline), (x + text_width + 10, text_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, text, (x + 5, text_y), FONT, FONT_SCALE, BLACK, THICKNESS, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
        return frame

    def _predict_emotion_for_box(self, frame, box):
        x, y, x2, y2 = [int(coord) for coord in box]
        face_roi = frame[y:y2, x:x2]
        if face_roi.size > 0:
            pil_image = Image.fromarray(face_roi)
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits.float(), dim=-1) # Use .float() for stability
            pred_index = torch.argmax(probs, dim=1).item()
            emotion = self.classes[pred_index]
            all_probs_dict = {self.classes[i]: float(probs[0][i]) for i in range(len(self.classes))}
            return emotion, all_probs_dict
        return None, {}

    def _get_high_confidence_detections(self, frame, conf_threshold=0.95):
        boxes, probs = self.face_detector.detect(frame)
        if boxes is None: return []
        return [box for box, prob in zip(boxes, probs) if prob > conf_threshold]

    # --- THIS IS THE FINAL, HIGH-FPS LIVE PREDICTOR ---
    def predict_live_frame(self, frame):
        if frame is None: return frame, {}
        
        # --- 1. Resize the frame to a smaller, efficient size (The FPS Fix) ---
        h, w, _ = frame.shape
        scale = 640 / w if w > 0 else 1
        new_h, new_w = int(h * scale), int(w * scale)
        small_frame = cv2.resize(frame, (new_w, new_h))
        
        annotated_frame = small_frame.copy()
        probabilities = {}
        
        # --- 2. Use the simple, robust single-person tracker from before ---
        boxes_raw, probs = self.face_detector.detect(annotated_frame)
        
        high_conf_boxes = []
        if boxes_raw is not None:
            for box, prob in zip(boxes_raw, probs):
                if prob > 0.7:
                    high_conf_boxes.append(box)

        if high_conf_boxes:
            detected_box = high_conf_boxes[0]
            
            if self.live_feed_box_smoother is None:
                self.live_feed_box_smoother = detected_box
            else:
                self.live_feed_box_smoother = (
                    self.box_smoothing_factor * detected_box +
                    (1 - self.box_smoothing_factor) * self.live_feed_box_smoother
                )
            
            emotion, probabilities = self._predict_emotion_for_box(annotated_frame, self.live_feed_box_smoother)
            
            if emotion:
                self.live_feed_smoother.append(emotion)
            
            if self.live_feed_smoother:
                display_emotion = Counter(self.live_feed_smoother).most_common(1)[0][0]
                annotated_frame = self._draw_annotations(annotated_frame, self.live_feed_box_smoother, display_emotion)
        else:
            self.live_feed_box_smoother = None
            
        # --- 3. Return the processed small frame ---
        return annotated_frame, probabilities

    def predict_video_frame(self, frame):
        if frame is None: return frame
        annotated_frame = frame.copy()
        current_detections = self._get_high_confidence_detections(frame, conf_threshold=0.95)
        unmatched_detections = list(range(len(current_detections)))
        for face_id, face_data in self.tracked_faces.items():
            best_match_iou, best_match_idx = 0, -1
            for i in unmatched_detections:
                iou = calculate_iou(face_data['box'], current_detections[i])
                if iou > best_match_iou: best_match_iou, best_match_idx = iou, i
            if best_match_iou > 0.4:
                detected_box = current_detections[best_match_idx]
                face_data['box'] = (self.box_smoothing_factor * detected_box + (1 - self.box_smoothing_factor) * face_data['box'])
                face_data['frames_unseen'] = 0
                emotion, _ = self._predict_emotion_for_box(annotated_frame, face_data['box'])
                if face_data['history']:
                    potential_emotion = Counter(face_data['history']).most_common(1)[0][0]
                    if emotion == potential_emotion: face_data['confirmation_counter'] += 1
                    else: face_data['confirmation_counter'] = 0
                if face_data['confirmation_counter'] >= self.confirmation_frames:
                    face_data['stable_emotion'] = emotion
                if emotion: face_data['history'].append(emotion)
                unmatched_detections.remove(best_match_idx)
            else:
                face_data['frames_unseen'] += 1
        for i in unmatched_detections:
            new_id = self.next_face_id; self.next_face_id += 1
            emotion, _ = self._predict_emotion_for_box(annotated_frame, current_detections[i])
            if emotion:
                self.tracked_faces[new_id] = {
                    'box': current_detections[i], 'history': deque([emotion] * 5, maxlen=self.smoothing_window),
                    'frames_unseen': 0, 'stable_emotion': emotion, 'confirmation_counter': self.confirmation_frames
                }
        faces_to_remove = [face_id for face_id, face_data in self.tracked_faces.items() if face_data['frames_unseen'] > 10]
        for face_id in faces_to_remove: del self.tracked_faces[face_id]
        for face_id, face_data in self.tracked_faces.items():
            annotated_frame = self._draw_annotations(annotated_frame, face_data['box'], face_data['stable_emotion'])
        return annotated_frame

    def predict_static_with_probs(self, frame):
        if frame is None: return frame, {}
        annotated_frame = frame.copy()
        current_detections = self._get_high_confidence_detections(frame, conf_threshold=0.95)
        probabilities = {}
        if current_detections:
            for box in current_detections:
                emotion, current_probs = self._predict_emotion_for_box(annotated_frame, box)
                if emotion:
                    annotated_frame = self._draw_annotations(annotated_frame, box, emotion)
                    probabilities = current_probs
            return annotated_frame, probabilities
        h, w, _ = annotated_frame.shape
        error_text = "No face detected!"; font = cv2.FONT_HERSHEY_DUPLEX; font_scale = 1.0; thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(error_text, font, font_scale, thickness)
        text_x = (w - text_width) // 2; text_y = (h + text_height) // 2
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (text_x - 15, text_y - text_height - 15), (text_x + text_width + 15, text_y + baseline + 15), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        cv2.putText(annotated_frame, error_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        return annotated_frame, {}