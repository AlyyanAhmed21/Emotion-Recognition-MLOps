import gradio as gr
import os
import cv2
import time

# Ensure the correct predictor class is imported
from src.EmotionRecognition.pipeline.hf_predictor import HFPredictor

# --- CONFIGURATION ---
HOLD_TIME = 2.0  # (seconds)

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
/* Your existing CSS is perfect and remains unchanged */
.prediction-list .bar {
    height: 100%;
    background: linear-gradient(90deg, #8A2BE2, #C71585);
    border-radius: 12px;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
body {
    background: linear-gradient(-45deg, #0b0f19, #131a2d, #2a2a72, #522a72);
    background-size: 400% 400%; animation: gradient 15s ease infinite;
}
@keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
.gradio-container { max-width: 1320px !important; margin: auto !important; }
#title { text-align: center; font-size: 3rem !important; font-weight: 700; color: #FFF; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #bebebe; margin-top: 0; margin-bottom: 40px; font-size: 1.2rem; font-weight: 300; }
#main-card { background: rgba(22, 22, 34, 0.65); border-radius: 16px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.18); padding: 1rem; }
#predictions-column { background-color: rgba(15, 15, 25, 0.7); backdrop-filter: blur(5px); border-radius: 12px; padding: 1.5rem !important; }
#predictions-column > .gr-label { display: none; }
.prediction-list { list-style-type: none; padding: 0; margin-top: 1.5rem; }
.prediction-list li { display: flex; align-items: center; margin-bottom: 12px; font-size: 1.1rem; }
.prediction-list .label { width: 100px; text-transform: capitalize; color: #e0e_0e0; }
.prediction-list .bar-container { flex-grow: 1; height: 24px; background-color: rgba(255,255,255,0.1); border-radius: 12px; margin: 0 15px; overflow: hidden; }
.prediction-list .percent { width: 60px; text-align: right; font-weight: bold; color: #FFF; }
footer { display: none !important; }
"""

ABOUT_MARKDOWN = """
## 🚀 About This Project
This application is the culmination of a complete, end-to-end MLOps project, demonstrating the full lifecycle from research and experimentation to a final, deployed, state-of-the-art solution.
---
### 🛠️ Architecture & Tech Stack
*   **Machine Learning & CV:** Python, PyTorch, Hugging Face `transformers`, facenet-pytorch, OpenCV
*   **Application & UI:** Gradio
"""

# --- BACKEND LOGIC ---
def create_static_html(probabilities):
    html = "<ul class='prediction-list'>"
    sorted_preds = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for emotion, prob in sorted_preds:
        html += f"""
        <li>
            <strong class='label'>{emotion.capitalize()}</strong>
            <div class='bar-container'><div class='bar' style='width: {prob*100:.1f}%;'></div></div>
            <span class='percent'>{(prob*100):.1f}%</span>
        </li>"""
    html += "</ul>"
    return html

def handle_static_image(frame):
    # For a static image, we just use the video frame logic once
    annotated_frame = predictor.predict_video_frame(frame) 
    # Since we don't have probabilities back from the tracker, we just show the annotated image
    return annotated_frame, "<div style='padding: 2rem; text-align: center; color: #999;'>Analysis complete.</div>"

def handle_stream(frame, state):
    current_time = time.time()
    if state['last_update'] == 0:
        predictor.reset_tracker()
    annotated_frame, probabilities = predictor.predict_smoothed(frame)
    if not probabilities:
        return annotated_frame, gr.update(), state
    current_top_emotion = max(probabilities, key=probabilities.get)
    emotion_changed = current_top_emotion != state['last_emotion']
    hold_time_expired = (current_time - state['last_update']) > HOLD_TIME
    if emotion_changed or hold_time_expired:
        new_state = {'last_emotion': current_top_emotion, 'last_update': current_time}
        return annotated_frame, probabilities, new_state
    else:
        return annotated_frame, gr.update(), state

def handle_video(video_path, progress=gr.Progress(track_tqdm=True)):
    if video_path is None: return None
    predictor.reset_tracker()
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "processed_video_stable_final.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for _ in progress.tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = predictor.predict_video_frame(frame_rgb)
            if annotated_frame is not None:
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        cap.release(); out.release()
        return output_path
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        return None

def dummy_function(data):
    pass

# --- GRADIO UI with JavaScript Injection ---
with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    live_feed_state = gr.State({'last_emotion': None, 'last_update': 0})
    FIXED_EMOTION_ORDER = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    
    gr.Markdown("# Facial Emotion Detector", elem_id="title")
    gr.Markdown("A real-time AI application powered by Vision Transformers", elem_id="subtitle")

    with gr.Box(elem_id="main-card"):
        with gr.Tabs():
            with gr.TabItem("Live Detection"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        live_feed = gr.Image(source="webcam", streaming=True, type="numpy", label="Live Feed", height=550, mirror_webcam=True)
                    with gr.Column(scale=2, elem_id="predictions-column"):
                        gr.Markdown("### Emotion Probabilities")
                        html_content = "<ul class='prediction-list'>"
                        for emotion in FIXED_EMOTION_ORDER:
                            html_content += f"""<li><strong class='label'>{emotion.capitalize()}</strong><div class='bar-container'><div class='bar' id='bar-{emotion}'></div></div><span class='percent' id='percent-{emotion}'>0.0%</span></li>"""
                        html_content += "</ul>"
                        live_predictions = gr.HTML(html_content)
                        live_predictions_data = gr.JSON(visible=False)

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

    live_predictions_data.change(fn=dummy_function, inputs=live_predictions_data, outputs=None, _js="""
        (data) => {
            if (!data) return;
            const sorted_preds = Object.entries(data).sort(([,a],[,b]) => b-a);
            const list = document.querySelector('#predictions-column .prediction-list');
            if (list) {
                sorted_preds.forEach(([emotion, prob]) => {
                    const listItem = list.querySelector(`#bar-${emotion}`).closest('li');
                    if (listItem) { list.appendChild(listItem); }
                });
            }
            for (const emotion in data) {
                const prob = data[emotion];
                const bar = document.getElementById(`bar-${emotion}`);
                const percent = document.getElementById(`percent-${emotion}`);
                if (bar && percent) {
                    bar.style.width = (prob * 100).toFixed(1) + '%';
                    percent.textContent = (prob * 100).toFixed(1) + '%';
                }
            }
        }""")
    
    live_feed.stream(fn=handle_stream, inputs=[live_feed, live_feed_state], outputs=[live_feed, live_predictions_data, live_feed_state])
    image_button.click(fn=handle_static_image, inputs=[image_input], outputs=[image_input, image_predictions])
    video_button.click(fn=handle_video, inputs=[video_input], outputs=[video_output])

# --- LAUNCH THE APP ---
if predictor:
    demo.queue().launch(debug=True)
else:
    print("\n[FATAL ERROR] Could not start the application.")