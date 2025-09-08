# 🎭 End-to-End Facial Emotion Recognition

 <!-- Replace with a link to your final app screenshot -->

This repository contains a complete, end-to-end MLOps pipeline and a production-ready web application for real-time facial emotion recognition. The project leverages a state-of-the-art Vision Transformer model and is deployed as a user-friendly Gradio application on Hugging Face Spaces.

**Live Demo:** [🚀 Click here to try the application on Hugging Face Spaces!](https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME) <!-- Replace with your HF Space URL -->

---

## ✨ Features

-   **Real-time Emotion Detection:** Analyzes your webcam feed to predict emotions in real-time.
-   **High Accuracy:** Powered by a pre-trained Swin Transformer model fine-tuned on the massive AffectNet dataset for superior performance on "in the wild" faces.
-   **Static Image & Video Analysis:** Upload your own images or videos for emotion prediction.
-   **Polished UI:** A professional and responsive user interface with an animated background, built with Gradio.
-   **Reproducible MLOps Pipeline:** The entire model training and data processing workflow is managed by DVC, ensuring 100% reproducibility.
-   **Containerized for Deployment:** The application is packaged with Docker for easy and consistent deployment anywhere.

## 🛠️ Tech Stack

-   **Model:** Swin Transformer (`PangPang/affectnet-swin-tiny-patch4-window7-224`)
-   **ML/Ops:** Python, TensorFlow/Keras, DVC, MLflow, Hugging Face `transformers`
-   **Backend & UI:** Gradio
-   **Face Detection:** MTCNN
-   **Deployment:** Hugging Face Spaces, Docker

## 🚀 Getting Started

Follow these steps to run the project locally.

### Prerequisites

-   Python 3.10+
-   Git and Git LFS ([installation guide](https://git-lfs.github.com))
-   An NVIDIA GPU with CUDA drivers is recommended for the training pipeline, but the deployed app runs on CPU.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Emotion-Recognition-MLOps.git
cd Emotion-Recognition-MLOps