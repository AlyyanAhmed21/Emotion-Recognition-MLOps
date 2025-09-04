import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN
from collections import deque, Counter
from datetime import datetime

class Predictor:
    def __init__(self, model_path, classes, input_image_size=(224, 224), smoothing_window=15, confidence_threshold=0.5):
        """
        Initializes the predictor with the trained model, face detector, and parameters.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.face_detector = MTCNN()
        self.classes = classes
        self.input_image_size = input_image_size
        self.confidence_threshold = confidence_threshold
        
        # This is our temporal smoothing buffer for the live feed
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"

    def predict_single_frame(self, frame):
        """
        Processes a single frame without temporal smoothing.
        Ideal for static image uploads.
        """
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
                face_resized = cv2.resize(face_roi, self.input_image_size, interpolation=cv2.INTER_NEAREST)
                face_batch = np.expand_dims(face_resized, axis=0)
                face_normalized = face_batch / 255.0
                predictions = self.model.predict(face_normalized, verbose=0)[0]
                pred_index = np.argmax(predictions)
                
                emotion = self.classes[pred_index]
                confidence = predictions[pred_index]
                
                text = f"{emotion} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                
                # Create a JSON-serializable dictionary of probabilities
                probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
                
        return annotated_frame, probabilities

    def predict_live(self, frame, current_log=[]):
        """
        Processes a single frame from the video stream to detect and classify emotions.
        Implements temporal smoothing for stable predictions.
        """
        if frame is None:
            return None, None, current_log

        faces = self.face_detector.detect_faces(frame)
        annotated_frame = frame.copy()
        probabilities = None

        # Loop through each face found
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            
            face_roi = frame[y:y+height, x:x+width]

            if face_roi.size > 0:
                # Preprocess the face for the model
                face_resized = cv2.resize(face_roi, self.input_image_size, interpolation=cv2.INTER_NEAREST)
                face_batch = np.expand_dims(face_resized, axis=0)
                face_normalized = face_batch / 255.0

                # Make Prediction
                predictions = self.model.predict(face_normalized, verbose=0)[0]
                pred_index = np.argmax(predictions)
                confidence = predictions[pred_index]

                # --- Temporal Smoothing Logic ---
                if confidence > self.confidence_threshold:
                    self.recent_predictions.append(pred_index)
                
                if self.recent_predictions:
                    most_common_pred = Counter(self.recent_predictions).most_common(1)[0][0]
                    self.stable_prediction = self.classes[most_common_pred]

                # Display the STABLE prediction
                text = f"{self.stable_prediction} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                
                probabilities = {emotion: float(prob) for emotion, prob in zip(self.classes, predictions)}

        # --- Log Management ---
        if probabilities:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (timestamp, self.stable_prediction)
            
            if not current_log or current_log[-1][1] != self.stable_prediction:
                current_log.append(log_entry)

        return annotated_frame, probabilities, current_log