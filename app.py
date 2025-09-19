import gradio as gr
import os
import cv2
import time

# Ensure the correct predictor class is imported
from src.EmotionRecognition.pipeline.hf_predictor import HFPredictor

# --- INITIALIZE THE MODEL ---
print("[INFO] Initializing predictor...")
try:
    predictor = HFPredictor()
    print("[INFO] Predictor initialized successfully.")
except Exception as e:
    predictor = None
    print(f"[FATAL ERROR] Failed to initialize predictor: {e}")

# --- UI CONTENT & STYLING ---
CSS = """
/* Animated Gradient Background */
body {
    background: linear-gradient(-45deg, #0b0f19, #131a2d, #2a2a72, #522a72);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
@keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

/* General Layout & Typography */
.gradio-container { max-width: 1320px !important; margin: auto !important; }
#title { text-align: center; font-size: 3rem !important; font-weight: 700; color: #FFF; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #bebebe; margin-top: 0; margin-bottom: 40px; font-size: 1.2rem; font-weight: 300; }
.gr-button { font-weight: bold !important; }

/* Main Content Card */
#main-card {
    background: rgba(22, 22, 34, 0.65);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 1rem;
}

/* Prediction Bar Styling */
#predictions-column { background-color: transparent !important; padding: 1.5rem; }
#predictions-column > .gr-label { display: none; }
.prediction-list { list-style-type: none; padding: 0; margin-top: 1.5rem; }
.prediction-list li { display: flex; align-items: center; margin-bottom: 12px; font-size: 1.1rem; }
.prediction-list .label { width: 100px; text-transform: capitalize; color: #e0e0e0; }
.prediction-list .bar-container { flex-grow: 1; height: 24px; background-color: rgba(255,255,255,0.1); border-radius: 12px; margin: 0 15px; overflow: hidden; }
.prediction-list .bar { height: 100%; background: linear-gradient(90deg, #8A2BE2, #C71585); border-radius: 12px; transition: width: 0.1s linear; }
.prediction-list .percent { width: 60px; text-align: right; font-weight: bold; color: #FFF; }
footer { display: none !important; }
"""

ABOUT_MARKDOWN = """
## 🚀 About This Project

This application is the culmination of a complete, end-to-end MLOps project, demonstrating the full lifecycle from research and experimentation to a final, deployed, state-of-the-art solution.

**💻 [View Project on GitHub](https://github.com/YOUR-USERNAME/Emotion-Recognition-MLOps)** <!--- REPLACE WITH YOUR GITHUB REPO LINK --->

---

### Key Technical Features:

*   **State-of-the-Art AI Model:** Utilizes a **Swin Transformer**, a powerful Vision Transformer (ViT) architecture, pre-trained on the massive **AffectNet** dataset. This ensures high accuracy and robust generalization to real-world, "in the wild" facial expressions.
*   **Reproducible MLOps Pipeline:** The original model training and data processing workflows were built using **DVC (Data Version Control)**, ensuring that every experiment is versioned and reproducible.
*   **Full-Stack & Deployment:** The application architecture evolved from a Python-only script to a decoupled **FastAPI backend** and a **React frontend**, and was ultimately deployed as this streamlined and robust **Gradio** application.
*   **Containerized & Automated:** The entire application is packaged with **Docker** and is set up for **CI/CD with GitHub Actions**, enabling automated testing and deployment to cloud platforms like Hugging Face Spaces.

---

### 🛠️ Architecture & Tech Stack

*   **Machine Learning & CV:** Python, PyTorch, Hugging Face `transformers`, MTCNN, OpenCV
*   **MLOps & DevOps:** DVC, GitHub Actions, Docker, Git LFS
*   **Application & UI:** Gradio

"""

# --- BACKEND LOGIC ---

def create_prediction_html(probabilities):
    """Generates clean HTML for the prediction bars."""
    if not probabilities:
        return "<div style='padding: 2rem; text-align: center; color: #999;'>Waiting for prediction...</div>"
    html = "<ul class='prediction-list'>"
    sorted_preds = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for emotion, prob in sorted_preds:
        html += f"""
        <li>
            <strong class='label'>{emotion}</strong>
            <div class='bar-container'><div class='bar' style='width: {prob*100:.1f}%;'></div></div>
            <span class='percent'>{(prob*100):.1f}%</span>
        </li>
        """
    html += "</ul>"
    return html

def unified_prediction_function(frame):
    """A single, robust function that takes any frame (from webcam or upload) and returns the annotated frame and the prediction HTML."""
    if frame is None:
        return None, create_prediction_html({})
    
    # The predictor class handles all annotation and prediction logic
    annotated_frame, probabilities = predictor.process_frame(frame)
    
    return annotated_frame, create_prediction_html(probabilities)

def process_video(video_path, progress=gr.Progress(track_tqdm=True)):
    """Processes an uploaded video file frame-by-frame."""
    if video_path is None: 
        return None
    try:
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
            annotated_frame, _ = predictor.process_frame(frame_rgb)
            if annotated_frame is not None:
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        return None

# --- GRADIO UI ---
with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    gr.Markdown("# Facial Emotion Detector", elem_id="title")
    gr.Markdown("A real-time AI application powered by Vision Transformers", elem_id="subtitle")

    with gr.Box(elem_id="main-card"):
        with gr.Tabs():
            with gr.TabItem("Live Detection"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        # The single, correct component for a live webcam feed.
                        live_feed = gr.Image(source="webcam", streaming=True, type="numpy", label="Live Feed", height=550, mirror_webcam=True)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        gr.Markdown("### Emotion Probabilities")
                        live_predictions = gr.HTML()
            
            with gr.TabItem("Upload Image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        image_input = gr.Image(type="numpy", label="Upload an Image", height=550)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        image_predictions = gr.HTML()
                image_button = gr.Button("Analyze Image", variant="primary")

            with gr.TabItem("Upload Video"):
                with gr.Row(equal_height=False):
                    video_input = gr.Video(label="Upload a Video File")
                    video_output = gr.Video(label="Processed Video")
                video_button = gr.Button("Analyze Video", variant="primary")
            
            with gr.TabItem("About"):
                gr.Markdown(ABOUT_MARKDOWN)

    # --- EVENT LISTENERS ---
    live_feed.stream(fn=unified_prediction_function, inputs=live_feed, outputs=[live_feed, live_predictions])
    image_button.click(fn=unified_prediction_function, inputs=[image_input], outputs=[image_input, image_predictions])
    video_button.click(fn=process_video, inputs=[video_input], outputs=[video_output])

# --- LAUNCH THE APP ---
if predictor:
    # Enabling the queue is essential for the video processing progress bar.
    demo.queue().launch(debug=True)
else:
    print("\n[FATAL ERROR] Could not start the application.")