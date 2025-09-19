# File: convert_model.py

import torch
import onnx
# --- THIS IS THE FIX ---
# Change the import to the new ONNX predictor, as HFPredictor is what we're converting
from src.EmotionRecognition.pipeline.hf_predictor import HFPredictor
# --- END FIX ---
import os

# --- CONFIGURATION ---
SOURCE_MODEL_DIR = "sota_model"
OPTIMIZED_MODEL_DIR = "sota_model_optimized"
ONNX_MODEL_PATH = os.path.join(OPTIMIZED_MODEL_DIR, "model.onnx")

def main():
    """
    Converts the PyTorch Hugging Face model to the ONNX format.
    """
    print("--- Starting Model Conversion to ONNX ---")
    
    os.makedirs(OPTIMIZED_MODEL_DIR, exist_ok=True)
    
    print(f"Loading original PyTorch model from '{SOURCE_MODEL_DIR}'...")
    # Use the HFPredictor to load the model
    predictor = HFPredictor()
    model = predictor.model
    model.eval()
    
    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 3, 224, 224, device=predictor.device)
    
    print(f"Exporting model to ONNX at '{ONNX_MODEL_PATH}'...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        # --- THIS IS THE FIX ---
        # Update the opset version to 14 or higher, as recommended by the error message.
        opset_version=14
        # --- END FIX ---
    )
    
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    
    print("\n--- Verification ---")
    print(f"ONNX model has been saved to: {ONNX_MODEL_PATH}")
    print(f"Model Inputs: {[input.name for input in onnx_model.graph.input]}")
    print(f"Model Outputs: {[output.name for output in onnx_model.graph.output]}")
    print("\n--- Conversion Successful! ---")


if __name__ == "__main__":
    main()