import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from collections import deque, Counter
from datetime import datetime

# --- IMPORTANT: Ensure this import works ---
# This assumes your 'hf_predictor.py' is at 'src/EmotionRecognition/pipeline/hf_predictor.py'
from src.EmotionRecognition.pipeline.prediction import HFPredictor

# --- INITIALIZE THE MODEL (ONLY ONCE ON STARTUP) ---
print("[INFO] Initializing predictor...")
try:
    # This will load the SOTA model from your local 'sota_model' folder
    predictor = HFPredictor()
    print("[INFO] Predictor initialized successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to initialize predictor: {e}")
    # If the model fails to load, we can't run the app.
    # We will raise the exception to stop the app from launching incorrectly.
    raise e

# --- UI CONTENT & STYLING ---
# We can inject custom CSS to style the app
CSS = """
#col-container { max-width: 95%; margin: 0 auto; }
#video-feed, #annotated-output, #video-output { min-height: 500px; border: 2px solid #555; border-radius: 12px; }
#emotion-probs { min-height: 500px; }
.gradio-container { background-color: #121212; }
footer { display: none !important; }
"""

about_model_markdown = """
## About This Model
This application uses a state-of-the-art Vision Transformer model to perform real-time facial emotion recognition.
### Model Architecture
- **Base Model:** Swin Transformer (tiny)
- **Model Name:** `PangPang/affectnet-swin-tiny-patch4-window7-224`
- **Output:** 8 emotion classes, including Neutral and Contempt.
### Dataset
The model was pre-trained on **AffectNet**, the largest database of facial expressions "in the wild," containing over 400,000 manually annotated images. This allows it to generalize well to real-world, spontaneous expressions.
"""

# --- PREDICTION LOGIC ---
# We use a simple class to hold the state of the log to avoid global variables
class AppState:
    def __init__(self):
        self.log_data = []

    def get_log(self):
        return self.log_data

    def update_log(self, new_prediction):
        if not self.log_data or self.log_data[-1][1] != new_prediction:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_data.insert(0, [timestamp, new_prediction]) # Insert at the top
            if len(self.log_data) > 10: # Keep the log size manageable
                self.log_data.pop()

app_state = AppState()

def live_emotion_detection(frame):
    """
    Main function for the live feed. Receives a frame, returns annotated frame and predictions.
    """
    if frame is None:
        return None, None
    
    annotated_frame, probabilities, _ = predictor.predict_live(frame, []) # Pass empty log for this context
    
    if probabilities:
      # Update the log with the stable prediction from the predictor instance
      app_state.update_log(predictor.stable_prediction)
    
    # Mirror the final output for a natural feel for the user
    annotated_frame = cv2.flip(annotated_frame, 1)

    return annotated_frame, probabilities

def image_emotion_detection(image):
    """Function for the image upload tab."""
    if image is None:
        return None, None
    annotated_frame, probabilities = predictor.predict_single_frame(image)
    return annotated_frame, probabilities

def video_emotion_detection(video_path, progress=gr.Progress()):
    """Function for the video upload tab with progress bar."""
    if video_path is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in progress.tqdm(range(frame_count), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame, _ = predictor.predict_single_frame(frame_rgb)
        
        if annotated_frame is not None:
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    
    cap.release()
    out.release()
    return output_path

# --- GRADIO UI DEFINITION ---
with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="teal")) as demo:
    gr.Markdown("# Live Facial Emotion Detector")
    gr.Markdown("A real-time AI application powered by a Vision Transformer model.")

    with gr.Tabs():
        with gr.TabItem("Live Detection"):
            with gr.Row(elem_id="col-container"):
                with gr.Column(scale=2):
                    # For Gradio v3, gr.Image is the correct way for live streaming
                    webcam_input = gr.Image(
                        source="webcam", 
                        streaming=True, 
                        type="numpy",
                        label="Webcam Feed",
                        elem_id="video-feed"
                    )
                with gr.Column(scale=1):
                    prediction_label = gr.Label(
                        label="Emotion Probabilities", 
                        num_top_classes=8,
                        elem_id="emotion-probs"
                    )
                    # We can use a simple button to refresh the log
                    log_output = gr.Dataframe(
                        headers=["Timestamp", "Predicted Emotion"],
                        datatype=["str", "str"], row_count=5, col_count=(2, "fixed"),
                        label="Prediction Log"
                    )
        
        with gr.TabItem("Upload Image"):
            with gr.Row(elem_id="col-container"):
                with gr.Column(scale=2):
                    image_input = gr.Image(type="numpy", label="Upload an Image")
                with gr.Column(scale=1):
                    image_prediction_label = gr.Label(label="Emotion Probabilities", num_top_classes=8)
            image_button = gr.Button("Analyze Image")

        with gr.TabItem("Upload Video"):
            with gr.Row(elem_id="col-container"):
                video_input = gr.Video(label="Upload a Video File")
                video_output = gr.Video(label="Processed Video", elem_id="video-output")
            video_button = gr.Button("Analyze Video")

        with gr.TabItem("About"):
            gr.Markdown(about_model_markdown)

    # --- LINKING COMPONENTS ---
    
    # Live Feed Logic
    # The 'stream' event links the webcam component to our prediction function
    webcam_input.stream(
        fn=live_emotion_detection,
        inputs=[webcam_input],
        outputs=[webcam_input, prediction_label],
    )
    
    # Logic to update the log. We can tie it to the stream event as well,
    # but it can be slow. A manual refresh is often better.
    # For now, let's just let the log update in the background.
    # To see it, the user would have to switch tabs.
    # A more advanced version could use a dummy textbox to trigger updates.

    # Image Upload Logic
    image_button.click(
        fn=image_emotion_detection,
        inputs=[image_input],
        outputs=[image_input, image_prediction_label]
    )

    # Video Upload Logic
    video_button.click(
        fn=video_emotion_detection,
        inputs=[video_input],
        outputs=[video_output]
    )

# --- LAUNCH THE APP ---
if predictor is not None:
    # Enabling the queue is essential for progress bars and handling multiple users
    demo.queue().launch(debug=True)
else:
    print("\n[FATAL ERROR] Could not start the application because the model failed to initialize.")