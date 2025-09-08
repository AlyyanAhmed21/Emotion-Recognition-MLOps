import gradio as gr
import os
import cv2
import time

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
# In app.py

CSS = """
/* Animated Gradient Background */
body {
    background: linear-gradient(-45deg, #0b0f19, #131a2d, #2a2a72, #522a72);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: #e0e0e0;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* General Layout & Typography */
.gradio-container { max-width: 1320px !important; margin: auto !important; }
#title { text-align: center; font-size: 3rem !important; font-weight: 700; color: #FFF; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #bebebe; margin-top: 0; margin-bottom: 40px; font-size: 1.2rem; font-weight: 300; }
.gr-button { font-weight: bold !important; }

/* --- NEW: The "Glass Card" effect --- */
#main-card {
    background: rgba(22, 22, 34, 0.65); /* Semi-transparent dark background */
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px); /* The "frosted glass" effect */
    -webkit-backdrop-filter: blur(12px); /* For Safari */
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 1rem;
}
/* --- END NEW --- */

/* Prediction Bar Styling - now inside the card */
#predictions-column { background-color: transparent !important; border-radius: 12px; padding: 1.5rem; }
#predictions-column > .gr-label { display: none; }
.prediction-list { list-style-type: none; padding: 0; margin-top: 0; }
.prediction-list li { display: flex; align-items: center; margin-bottom: 12px; font-size: 1.1rem; }
.prediction-list .label { width: 100px; text-transform: capitalize; color: #e0e0e0; }
.prediction-list .bar-container { flex-grow: 1; height: 24px; background-color: rgba(255,255,255,0.1); border-radius: 12px; margin: 0 15px; overflow: hidden; }
.prediction-list .bar { height: 100%; background: linear-gradient(90deg, #8A2BE2, #C71585); border-radius: 12px; transition: width 0.2s ease-in-out; }
.prediction-list .percent { width: 60px; text-align: right; font-weight: bold; color: #FFF; }
footer { display: none !important; }
"""

ABOUT_MARKDOWN = """
### Model: Vision Transformer (ViT)
This application uses a Vision Transformer model, fine-tuned for facial emotion recognition.
### Dataset
The model was fine-tuned on the **Emotion Recognition Dataset** from Kaggle, a large, curated collection of labeled facial images. This diverse dataset allows the model to generalize to a wide variety of real-world faces and expressions.
*Dataset Link:* [https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)
### MLOps Pipeline
This entire application, from data processing to training and deployment, was built using a reproducible MLOps pipeline, ensuring consistency and quality at every step.
"""

# --- BACKEND LOGIC ---
def create_prediction_html(probabilities):
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

def live_detection_stream():
    """A generator function that runs the live feed loop. This is the definitive fix."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, probabilities = predictor.process_frame(frame_rgb)
            yield annotated_frame, create_prediction_html(probabilities)
            time.sleep(0.05) # Controls FPS. 0.05 = ~20 FPS target. The model inference will be the main bottleneck.
    finally:
        print("[INFO] Live feed stopped. Releasing webcam.")
        cap.release()

def process_image(image):
    if image is None: return None, create_prediction_html({})
    annotated_frame, probabilities = predictor.process_frame(image)
    return annotated_frame, create_prediction_html(probabilities)

def process_video(video_path, progress=gr.Progress(track_tqdm=True)):
    if video_path is None: return None
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

    # --- NEW: Wrapper for the glass card effect ---
    with gr.Box(elem_id="main-card"):
        with gr.Tabs():
            with gr.TabItem("Live Detection"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        live_output = gr.Image(label="Live Feed", interactive=False, height=550)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        gr.Markdown("### Emotion Probabilities") # Title for the panel
                        live_predictions = gr.HTML()
                with gr.Row():
                    start_button = gr.Button("Start Webcam", variant="primary", scale=1)
                    stop_button = gr.Button("Stop Webcam", variant="secondary", scale=1)
                
                stream_state = gr.State("Stop")

            with gr.TabItem("Upload Image"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        image_input = gr.Image(type="numpy", label="Upload an Image", height=550)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        gr.Markdown("### Emotion Probabilities")
                        image_predictions = gr.HTML()
                image_button = gr.Button("Analyze Image", variant="primary")

            with gr.TabItem("Upload Video"):
                with gr.Row(equal_height=True):
                    video_input = gr.Video(label="Upload a Video File")
                    video_output = gr.Video(label="Processed Video")
                video_button = gr.Button("Analyze Video", variant="primary")
            
            with gr.TabItem("About"):
                gr.Markdown(ABOUT_MARKDOWN)
    # --- END WRAPPER ---

    # --- EVENT LISTENERS (No changes needed here) ---
    start_event = start_button.click(lambda: "Start", None, stream_state, queue=False)
    live_stream = start_event.then(live_detection_stream, stream_state, [live_output, live_predictions])
    
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[live_stream])

    image_button.click(process_image, [image_input], [image_input, image_predictions])
    video_button.click(process_video, [video_input], [video_output])

# --- LAUNCH THE APP ---
if predictor:
    demo.queue().launch(debug=True, share=True)
else:
    print("\n[FATAL ERROR] Could not start the application.")