import gradio as gr
import os
import cv2
import time

from src.EmotionRecognition.pipeline.onnx_predictor import ONNXPredictor

print("[INFO] Initializing ONNX predictor...")
try:
    predictor = ONNXPredictor()
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

/* Main Content Card "Glassmorphism" effect */
#main-card {
    background: rgba(22, 22, 34, 0.65);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 1rem;
}

/* Prediction Bar Styling & Layout */
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

def live_detection_stream(stream_state):
    """A generator function that runs the live feed loop. This is the definitive fix."""
    if stream_state != "Start":
        # This function is called once with "Stop" to terminate the loop.
        print("[INFO] Stop signal received. Terminating live feed.")
        yield None, create_prediction_html({}) # Clear the UI
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    try:
        print("[INFO] Webcam started. Starting live prediction loop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, probabilities = predictor.process_frame(frame_rgb)
            
            # This 'yield' is the key to streaming. It sends the data back to the UI.
            yield annotated_frame, create_prediction_html(probabilities)
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

    with gr.Box(elem_id="main-card"):
        with gr.Tabs():
            with gr.TabItem("Live Detection"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        live_output = gr.Image(label="Live Feed", interactive=False, height=550)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        gr.Markdown("### Emotion Probabilities")
                        live_predictions = gr.HTML()
                with gr.Row():
                    start_button = gr.Button("Start Webcam", variant="primary", scale=1)
                    stop_button = gr.Button("Stop Webcam", variant="secondary", scale=1)
                
                # Hidden state to robustly control the loop
                stream_state = gr.State("Stop")

            with gr.TabItem("Upload Image"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        img_upload_input = gr.Image(type="numpy", label="Upload an Image", height=550)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        img_upload_predictions = gr.HTML()
                img_upload_button = gr.Button("Analyze Image", variant="primary")

            with gr.TabItem("Upload Video"):
                with gr.Row(equal_height=True):
                    video_upload_input = gr.Video(label="Upload a Video File")
                    video_upload_output = gr.Video(label="Processed Video")
                video_upload_button = gr.Button("Analyze Video", variant="primary")
            
            with gr.TabItem("About"):
                gr.Markdown(ABOUT_MARKDOWN)

    # --- EVENT LISTENERS ---
    
    # This is the definitive, robust way to handle a start/stop generator in Gradio
    start_event = start_button.click(lambda: "Start", None, stream_state, queue=False)
    live_stream = start_event.then(live_detection_stream, stream_state, [live_output, live_predictions])
    
    # Stop button's click event cancels the running live_stream event.
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[live_stream])

    img_upload_button.click(process_image, [img_upload_input], [img_upload_input, img_upload_predictions])
    video_upload_button.click(process_video, [video_upload_input], [video_upload_output])

# --- LAUNCH THE APP ---
if predictor:
    # Enabling the queue is essential for the 'cancels' and progress bars to work
    demo.queue().launch(debug=True)
else:
    print("\n[FATAL ERROR] Could not start the application.")