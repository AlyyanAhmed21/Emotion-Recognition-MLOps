import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
from mtcnn import MTCNN
from collections import deque, Counter
from datetime import datetime

class Predictor:
    def __init__(self, model_path, classes, input_image_size=(224, 224), smoothing_window=15, confidence_threshold=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.face_detector = MTCNN()
        self.classes = classes
        self.input_image_size = input_image_size
        self.confidence_threshold = confidence_threshold
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"

    def _process_frame(self, frame):
        """Helper function to find faces and predict emotion in a single frame."""
        faces = self.face_detector.detect_faces(frame)
        annotated_frame = frame.copy()
        
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+height, x:x+width]
            
            if face_roi.size > 0:
                face_resized = cv2.resize(face_roi, self.input_image_size, interpolation=cv2.INTER_NEAREST)
                face_batch = np.expand_dims(face_resized, axis=0)
                face_normalized = face_batch / 255.0
                predictions = self.model.predict(face_normalized, verbose=0)[0]
                pred_index = np.argmax(predictions)
                
                # For static images/videos, we don't use smoothing, just the direct prediction.
                emotion = self.classes[pred_index]
                confidence = predictions[pred_index]
                
                text = f"{emotion} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

        return annotated_frame

    def predict_image(self, image):
        """Processes a single uploaded image."""
        if image is None:
            return None
        # Gradio provides image as RGB, which is what MTCNN expects.
        return self._process_frame(image)

    def predict_video(self, video_path, progress=gr.Progress()):
        """Processes an uploaded video file frame by frame."""
        if video_path is None:
            return None
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a video writer to save the output
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in progress.tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = self._process_frame(frame_rgb)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame_bgr)
            
        cap.release()
        out.release()
        return output_path

    def predict_live(self, frame, current_log):
        """Processes a single frame from the LIVE video stream with smoothing."""
        if frame is None:
            return None, None, current_log

        faces = self.face_detector.detect_faces(frame)
        annotated_frame = frame.copy()
        current_prediction_label = None

        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+height, x:x+width]

            if face_roi.size > 0:
                face_resized = cv2.resize(face_roi, self.input_image_size, interpolation=cv2.INTER_NEAREST)
                face_batch = np.expand_dims(face_resized, axis=0)
                face_normalized = face_batch / 255.0
                predictions = self.model.predict(face_normalized, verbose=0)[0]
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
                
                current_prediction_label = {emotion: f"{prob:.2f}" for emotion, prob in zip(self.classes, predictions)}
        
        if current_prediction_label:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (timestamp, self.stable_prediction)
            if not current_log or current_log[-1][1] != self.stable_prediction:
                current_log.append(log_entry)

        return annotated_frame, current_prediction_label, current_log