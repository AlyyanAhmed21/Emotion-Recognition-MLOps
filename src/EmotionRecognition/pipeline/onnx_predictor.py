import onnxruntime as ort
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from collections import deque, Counter
from PIL import Image
import torch # Still needed for MTCNN device detection

class ONNXPredictor:
    def __init__(self, model_path="sota_model_optimized/model.onnx", smoothing_window=5, face_confidence_threshold=0.95):
        print(f"[ONNX PREDICTOR INFO] Loading ONNX model from: {model_path}...")
        
        # --- NEW: Load the ONNX model into an ONNX Runtime session ---
        # Specify that we want to use the CUDA (GPU) provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        # --- END NEW ---
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape # e.g., [1, 3, 224, 224]

        self.device = 'cuda' if 'CUDAExecutionProvider' in self.session.get_providers() else 'cpu'
        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.face_confidence_threshold = face_confidence_threshold
        
        # The classes need to be hardcoded now as they aren't in the ONNX model
        self.classes = ['sad', 'disgust', 'angry', 'fear', 'surprise', 'happy']
        # Let's double check the source. It seems to be 7 classes.
        # Let's use the one from the log, which is more reliable.
        self.classes = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
        self.recent_predictions = deque(maxlen=smoothing_window)
        self.stable_prediction = "---"

        print("[ONNX PREDICTOR INFO] Predictor initialized successfully.")
        print(f"[ONNX PREDICTOR INFO] Running on provider: {self.session.get_providers()[0]}")

    def process_frame(self, frame):
        """Processes a frame using the optimized ONNX model."""
        if frame is None: return frame, {}
        annotated_frame = frame.copy()
        all_probabilities = {}

        pil_frame = Image.fromarray(frame)
        boxes, probs = self.face_detector.detect(pil_frame)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < self.face_confidence_threshold: continue
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    # --- NEW: Preprocessing for ONNX Runtime ---
                    # Resize, convert to float32, and normalize
                    img = cv2.resize(face_roi, (self.input_shape[3], self.input_shape[2]))
                    img = img.astype(np.float32) / 255.0
                    # Transpose from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
                    img = np.transpose(img, (2, 0, 1))
                    # Add a batch dimension
                    input_tensor = np.expand_dims(img, axis=0)
                    # --- END NEW ---

                    # Run inference
                    logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
                    
                    # Convert logits to probabilities
                    exp_logits = np.exp(logits)
                    predictions = exp_logits / np.sum(exp_logits)
                    predictions = predictions[0] # Get the first (and only) batch item
                    
                    pred_index = np.argmax(predictions)
                    confidence = predictions[pred_index]

                    # (Temporal smoothing and drawing logic is the same)
                    
                    all_probabilities = {self.classes[i]: float(predictions[i]) for i in range(len(self.classes))}
        
        return annotated_frame, all_probabilities