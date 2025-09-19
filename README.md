# 🚀 Real-Time Emotion Recognition with Advanced Tracking & Smoothing

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.5%2B%20(CUDA)-orange.svg)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.x-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An advanced, end-to-end computer vision application that performs real-time facial emotion recognition with state-of-the-art stability, tracking, and a polished user interface. This project moves beyond a simple prediction script to a robust, GPU-accelerated demo showcasing advanced techniques for a professional-grade user experience.

---

### 🌟 Live Demo

*(Strongly recommend you record a GIF of your app and replace this image!)*

*Live detection running smoothly with GPU acceleration, stabilized bounding boxes, and a flicker-free UI.*

---

## ✨ Key Features & Technical Highlights

This project implements several advanced features to overcome common challenges in real-time computer vision:

#### 🚀 High-Performance GPU Acceleration
*   The entire inference pipeline, from face detection (`facenet-pytorch`) to emotion classification (`Swin Transformer`), runs on the **GPU using PyTorch with CUDA**.
*   This enables high-FPS, real-time processing of live webcam feeds and dramatically accelerates the analysis of pre-recorded videos.

#### 🧠 Advanced Multi-Face Tracking & Stability
*   **Temporal Emotion Smoothing (Hysteresis):** To prevent distracting, single-frame flickers, the system uses a confirmation-based approach. An emotion label for a person will not change unless the new emotion has been consistently detected for several consecutive frames, reflecting true emotional states.
*   **Bounding Box Smoothing (EMA):** Bounding boxes are smoothed using an Exponential Moving Average (EMA). This eliminates the common "jitter" artifact, resulting in boxes that glide smoothly and track faces with a stable, cinematic feel.
*   **Robust Identity Tracking (IOU):** An Intersection-over-Union (IOU) tracker is used in video processing to maintain the identity of each person from one frame to the next, ensuring that smoothing and hysteresis are applied correctly to each individual.

#### 🖥️ Polished & Flicker-Free User Interface
*   The Gradio UI features a **100% flicker-free** prediction panel.
*   Instead of re-rendering HTML, the Python backend sends raw probability data to a hidden JSON component. A custom **JavaScript listener** then smoothly animates the bar widths and re-orders the list in the browser, providing a seamless and professional user experience.

#### 🎯 High-Accuracy Detection
*   **State-of-the-Art Model:** Utilizes a **Swin Transformer**, a powerful Vision Transformer architecture, for high-accuracy emotion classification.
*   **Robust Face Detection:** Employs `facenet-pytorch` MTCNN for fast, GPU-accelerated face detection, with confidence thresholding to eliminate false positives on non-face objects.

---

## 🛠️ Tech Stack & Architecture

*   **Machine Learning / CV:** PyTorch (CUDA), Hugging Face Transformers, `facenet-pytorch` (for MTCNN), OpenCV, Pillow, NumPy
*   **Application & UI:** Gradio, JavaScript
*   **MLOps & Environment:** Python `venv`, Git

**Flow Diagram:**
`Input (Video/Webcam)` -> `MTCNN (GPU)` -> `Crop Faces` -> `Swin Transformer (GPU)` -> `IOU Tracker & Smoothing` -> `Annotate Frame` -> `Gradio UI`

---

## 👥 Meet the Team
This project was a collaborative effort by:
* Alyyan Ahmed - AI & MLOps Engineer
* Munim Akbar - AI & MLOps Engineer
## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
