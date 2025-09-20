import gradio as gr
import os
import cv2
import time

from src.EmotionRecognition.pipeline.hf_predictor import HFPredictor

# --- CONFIGURATION ---
HOLD_TIME = 2.0

# --- INITIALIZE THE MODEL ---
print("[INFO] Initializing predictor...")
try:
    predictor = HFPredictor()
    print("[INFO] Predictor initialized successfully.")
except Exception as e:
    predictor = None
    print(f"[FATAL ERROR] Failed to initialize predictor: {e}")

# --- UI CONTENT & STYLING (Complete Overhaul) ---
CSS = """
/* --- 1. Animated Black & Dark Blue Background --- */
body {
  background: linear-gradient(-45deg, #000000, #0b0f19, #131a2d, #000033);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* --- 2. Professional Theming & Layout --- */
.gradio-container { background: transparent !important; }
#title, #subtitle { color: #FFF; text-align: center; }
#title { font-size: 3rem !important; font-weight: 700; margin-bottom: 0.5rem; }
#subtitle { font-size: 1.2rem; font-weight: 300; margin-bottom: 40px; }
#main-card {
    background: rgba(11, 15, 25, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 2rem !important; /* Fixes text on border issue */
}

/* --- 3. Interactive, Button-like Tabs --- */
.tabs > .tab-nav {
    justify-content: center !important; /* Center aligns the tabs */
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}
.tabs > .tab-nav > button {
    background: transparent !important;
    border: none !important;
    color: #a0aec0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 6px 6px 0 0 !important;
    transition: all 0.3s ease !important;
}
.tabs > .tab-nav > button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #fff !important;
}
.tabs > .tab-nav > button.selected {
    color: #fff !important;
    border-bottom: 2px solid #3b82f6 !important; /* Blue accent for selected tab */
}

/* --- 4. Prediction Panel & FPS Counter --- */
#predictions-column { background-color: transparent !important; border: none !important; padding: 1.5rem !important; }
.prediction-list li { display: flex; align-items: center; margin-bottom: 12px; font-size: 1.1rem; }
.prediction-list .label { width: 100px; text-transform: capitalize; color: #FFF; }
.prediction-list .bar-container { flex-grow: 1; height: 24px; background-color: rgba(255,255,255,0.1); border-radius: 12px; margin: 0 15px; overflow: hidden; }
.prediction-list .bar { height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6); border-radius: 12px; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.prediction-list .percent { width: 60px; text-align: right; font-weight: bold; color: #FFF; }
#fps-counter { text-align: center; font-weight: bold; }

/* --- 5. Hide the unprofessional "Record" button --- */
.record-button { display: none !important; }
"""

ABOUT_MARKDOWN = """
## 🚀 About This Project

This application demonstrates a complete, end-to-end MLOps workflow, resulting in a high-performance, real-time facial emotion recognition system. It leverages a state-of-the-art AI model and incorporates advanced techniques for a robust and polished user experience.

**[View the Project on GitHub](https://github.com/your-username/your-repo-name)** <!--- REPLACE WITH YOUR GITHUB REPO LINK --->

---

### ✨ Key Technical Features

*   **High-FPS Live Performance:** The live feed is optimized by resizing the input frame before processing and using mixed-precision inference (`torch.cuda.amp`) to achieve a smooth, high-FPS experience on GPU.

*   **Advanced Multi-Face Tracking & Stability:** For video analysis, an **IOU (Intersection-over-Union) tracker** maintains the identity of each person, while **temporal smoothing (hysteresis)** prevents distracting label flicker. Bounding boxes are stabilized with an **Exponential Moving Average (EMA)** for a smooth, cinematic feel.

*   **Flicker-Free UI:** The prediction panel is updated using a custom JavaScript listener that animates changes smoothly in the browser, providing a seamless and professional user experience without any jarring flashes.

---

### 👥 The Team

*   **Alyyan Ahmed** - AI & MLOps Engineer
*   **Munim Akbar** - AI & MLOps Engineer
"""

# --- BACKEND LOGIC ---
def create_static_html(probabilities):
    if not probabilities: return "<div style='padding: 2rem; text-align: center; color: #999;'>No face detected.</div>"
    html = "<ul class='prediction-list'>"
    sorted_preds = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for emotion, prob in sorted_preds:
        html += f"""<li><strong class='label'>{emotion.capitalize()}</strong><div class='bar-container'><div class='bar' style='width: {prob*100:.1f}%;'></div></div><span class='percent'>{(prob*100):.1f}%</span></li>"""
    html += "</ul>"
    return html

def handle_static_image(frame):
    predictor.reset_tracker()
    annotated_frame, probabilities = predictor.predict_static_with_probs(frame)
    return annotated_frame, create_static_html(probabilities)

def handle_stream(frame, state):
    start_time = time.time()
    current_time = time.time()
    if state['last_update'] == 0:
        predictor.reset_tracker()
    annotated_frame, probabilities = predictor.predict_live_frame(frame)
    duration = time.time() - start_time
    fps = 1 / duration if duration > 0 else 0
    fps_text = f"{fps:.1f} FPS"
    if not probabilities:
        return annotated_frame, gr.update(), fps_text, state
    current_top_emotion = max(probabilities, key=probabilities.get)
    emotion_changed = current_top_emotion != state['last_emotion']
    hold_time_expired = (current_time - state['last_update']) > HOLD_TIME
    if emotion_changed or hold_time_expired:
        new_state = {'last_emotion': current_top_emotion, 'last_update': current_time}
        return annotated_frame, probabilities, fps_text, new_state
    else:
        return annotated_frame, gr.update(), fps_text, state

def handle_video(video_path, progress=gr.Progress(track_tqdm=True)):
    predictor.reset_tracker()
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "processed_video_final.mp4"
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

# --- GRADIO UI ---
with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo:
    live_feed_state = gr.State({'last_emotion': None, 'last_update': 0})
    FIXED_EMOTION_ORDER = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    
    with gr.Column():
        gr.Markdown("# Facial Emotion Detector", elem_id="title")
        gr.Markdown("A real-time AI application powered by Vision Transformers", elem_id="subtitle")

        with gr.Group(elem_id="main-card"):
            with gr.Tabs():
                with gr.Tab("Live Detection"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            live_feed = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Live Feed", height=550)
                            fps_counter = gr.Textbox(label="Live FPS", interactive=False, elem_id="fps-counter")
                        with gr.Column(scale=2, elem_id="predictions-column"):
                            gr.Markdown("### Emotion Probabilities")
                            html_content = "<ul class='prediction-list'>"
                            for emotion in FIXED_EMOTION_ORDER:
                                html_content += f"""<li><strong class='label'>{emotion.capitalize()}</strong><div class='bar-container'><div class='bar' id='bar-{emotion}'></div></div><span class='percent' id='percent-{emotion}'>0.0%</span></li>"""
                            html_content += "</ul>"
                            live_predictions = gr.HTML(html_content)
                            live_predictions_data = gr.JSON(visible=False)

                with gr.Tab("Upload Image"):
                    with gr.Row():
                        image_input = gr.Image(type="numpy", label="Upload an Image", height=550)
                        image_predictions = gr.HTML()
                    image_button = gr.Button("Analyze Image", variant="primary")

                with gr.Tab("Upload Video"):
                    with gr.Row():
                        video_input = gr.Video(label="Upload a Video File")
                        video_output = gr.Video(label="Processed Video")
                    video_button = gr.Button("Analyze Video", variant="primary")
                
                with gr.Tab("About"):
                    with gr.Column():
                        gr.Markdown(ABOUT_MARKDOWN)

    # JavaScript listener
    live_predictions_data.change(fn=dummy_function, inputs=live_predictions_data, outputs=None, js="""
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
    
    # Event listeners
    live_feed.stream(fn=handle_stream, inputs=[live_feed, live_feed_state], outputs=[live_feed, live_predictions_data, fps_counter, live_feed_state])
    image_button.click(fn=handle_static_image, inputs=[image_input], outputs=[image_input, image_predictions])
    video_button.click(fn=handle_video, inputs=[video_input], outputs=[video_output])

# --- LAUNCH THE APP ---
if predictor:
    demo.queue().launch(debug=True)
else:
    print("\n[FATAL ERROR] Could not start the application.")