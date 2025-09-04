# File: app.py

import gradio as gr
from src.EmotionRecognition.pipeline.prediction import Predictor
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "training", "model.keras")
CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# --- INSTANTIATE THE PREDICTOR ---
print("[INFO] Initializing predictor...")
predictor = Predictor(model_path=MODEL_PATH, classes=CLASSES)
print("[INFO] Predictor initialized successfully.")

# --- UI CONTENT ---
about_model_markdown = """
# About This Model

This application uses a deep learning model to perform real-time facial emotion recognition.

### Model Architecture
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Classifier Head:** A custom head with a Dense layer (128 neurons, ReLU activation, L2 regularization) and a Dropout layer was added for classification.
- **Output:** 7 emotion classes.

### Dataset
- The model was trained on the **CK+ (The Extended Cohn-Kanade Dataset)**.

### Performance
- **Final Validation Accuracy:** Achieved very high accuracy (~96%) on the validation set. Performance on "in-the-wild" faces may vary.
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Emotion Detector") as demo:
    gr.Markdown("# Facial Emotion Recognition")
    
    with gr.Tabs():
        # --- TAB 1: LIVE DETECTION ---
        with gr.TabItem("Live Feed"):
            gr.Markdown("## Real-time Emotion Detection from Your Webcam")
            with gr.Row():
                webcam_input = gr.Image(source="webcam", streaming=True, label="Webcam Feed", type="numpy")
                with gr.Column():
                    annotated_output = gr.Image(label="Processed Feed")
                    prediction_label = gr.Label(label="Emotion Probabilities", num_top_classes=len(CLASSES))
            
            with gr.Accordion("Prediction Log", open=False):
                # We will still use a state object to hold the log data
                log_state = gr.State([]) 
                log_output = gr.Dataframe(
                    headers=["Timestamp", "Predicted Emotion"],
                    datatype=["str", "str"], row_count=10, col_count=(2, "fixed")
                )
                # Add a button to manually refresh the log
                refresh_log_button = gr.Button("Refresh Log")

            # --- TAB 2: IMAGE UPLOAD ---
            with gr.TabItem("Upload Image"):
                gr.Markdown("## Get Emotion Prediction for a Single Image")
                with gr.Row():
                    image_input = gr.Image(type="numpy", label="Upload an Image")
                    image_output = gr.Image(label="Result")
                image_button = gr.Button("Analyze Image")

            # --- TAB 3: VIDEO UPLOAD ---
            with gr.TabItem("Upload Video"):
                gr.Markdown("## Get Emotion Prediction for a Video File")
                with gr.Row():
                    video_input = gr.Video(label="Upload a Video")
                    video_output = gr.Video(label="Result")
                video_button = gr.Button("Analyze Video")

            # --- TAB 4: ABOUT ---
            with gr.TabItem("About the Model"):
                gr.Markdown(about_model_markdown)

 # --- LINKING LOGIC ---
    # Live Feed Logic
    webcam_input.stream(
        fn=predictor.predict_live,
        inputs=[webcam_input, log_state],
        outputs=[annotated_output, prediction_label, log_state]
    )

    # --- THIS IS THE FIX ---
    # Remove the log_state.change() line.
    # Instead, link the refresh button to update the log display.
    # The function is a simple lambda that takes the current state and returns it.
    refresh_log_button.click(
        fn=lambda log: log, 
        inputs=[log_state], 
        outputs=[log_output]
    )

    # Image Upload Logic
    image_button.click(
        fn=predictor.predict_image,
        inputs=[image_input],
        outputs=[image_output]
    )

    # Video Upload Logic
    video_button.click(
        fn=predictor.predict_video,
        inputs=[video_input],
        outputs=[video_output]
    )


if __name__ == "__main__":
    # --- THIS IS THE FIX ---
    # Chain the .queue() method before .launch() to enable the processing queue.
    demo.queue().launch(debug=True, share=True)