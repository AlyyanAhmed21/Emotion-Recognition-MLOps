from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN 
from collections import deque, Counter
from PIL import Image

LOCAL_MODEL_PATH = "sota_model"

# --- HELPER FUNCTION for the IOU Tracker ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class HFPredictor:
    def __init__(self, smoothing_window=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PREDICTOR INFO] Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
        self.model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

        self.face_detector = MTCNN(keep_all=True, device=self.device, thresholds=[0.7, 0.8, 0.95])
        self.classes = list(self.model.config.id2label.values())
        
        # State for multi-face IOU Tracker
        self.tracked_faces = {}
        self.next_face_id = 0
        self.smoothing_window = smoothing_window
        
        # State for single-face live feed smoother
        self.live_feed_smoother = deque(maxlen=smoothing_window)
        self.live_feed_box_smoother = None
        self.box_smoothing_factor = 0.4


        self.COLOR_MAP = {
            "angry": (0, 0, 255), "disgust": (0, 100, 0), "fear": (130, 0, 75),
            "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (0, 165, 255),
            "neutral": (255, 255, 255)
        }
        print("[PREDICTOR INFO] Predictor initialized successfully.")

    def reset_tracker(self):
        """Resets the state of the multi-face tracker and live feed smoother."""
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
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_index = torch.argmax(probs, dim=1).item()
            emotion = self.classes[pred_index]
            all_probs_dict = {self.classes[i]: float(probs[0][i]) for i in range(len(self.classes))}
            return emotion, all_probs_dict
        return None, {}

    def predict_video_frame(self, frame):
        if frame is None: return frame
        annotated_frame = frame.copy()
        
        boxes, probs = self.face_detector.detect(annotated_frame)
        current_detections = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob > 0.95:
                    current_detections.append(box)

        unmatched_detections = list(range(len(current_detections)))
        matched_track_ids = []

        for face_id, face_data in self.tracked_faces.items():
            best_match_iou, best_match_idx = 0, -1
            for i in unmatched_detections:
                iou = calculate_iou(face_data['box'], current_detections[i])
                if iou > best_match_iou:
                    best_match_iou, best_match_idx = iou, i
            
            if best_match_iou > 0.4:
                face_data['box'] = current_detections[best_match_idx]
                face_data['frames_unseen'] = 0
                emotion, _ = self._predict_emotion_for_box(annotated_frame, face_data['box'])
                if emotion: face_data['history'].append(emotion)
                unmatched_detections.remove(best_match_idx)
                matched_track_ids.append(face_id)
            else:
                face_data['frames_unseen'] += 1

        for i in unmatched_detections:
            new_id = self.next_face_id
            self.next_face_id += 1
            emotion, _ = self._predict_emotion_for_box(annotated_frame, current_detections[i])
            if emotion:
                self.tracked_faces[new_id] = {
                    'box': current_detections[i],
                    'history': deque([emotion] * 5, maxlen=self.smoothing_window),
                    'frames_unseen': 0
                }

        faces_to_remove = [face_id for face_id, face_data in self.tracked_faces.items() if face_data['frames_unseen'] > 10]
        for face_id in faces_to_remove: del self.tracked_faces[face_id]

        for face_id, face_data in self.tracked_faces.items():
            if face_data['history']:
                stable_emotion = Counter(face_data['history']).most_common(1)[0][0]
                annotated_frame = self._draw_annotations(annotated_frame, face_data['box'], stable_emotion)
        return annotated_frame

    def predict_smoothed(self, frame):
        if frame is None: return frame, {}
        annotated_frame = frame.copy()
        boxes_raw, probs = self.face_detector.detect(annotated_frame)
        
        high_conf_boxes = []
        if boxes_raw is not None:
            for box, prob in zip(boxes_raw, probs):
                if prob > 0.95: high_conf_boxes.append(box)

        if high_conf_boxes:
            detected_box = high_conf_boxes[0]
            if self.live_feed_box_smoother is None: self.live_feed_box_smoother = detected_box
            else: self.live_feed_box_smoother = (self.box_smoothing_factor * detected_box + (1 - self.box_smoothing_factor) * self.live_feed_box_smoother)
        
        probabilities = {}
        if self.live_feed_box_smoother is not None:
            emotion, probabilities = self._predict_emotion_for_box(annotated_frame, self.live_feed_box_smoother)
            if emotion: self.live_feed_smoother.append(emotion)
            if self.live_feed_smoother:
                display_emotion = Counter(self.live_feed_smoother).most_common(1)[0][0]
                annotated_frame = self._draw_annotations(annotated_frame, self.live_feed_box_smoother, display_emotion)

        return annotated_frame, probabilities