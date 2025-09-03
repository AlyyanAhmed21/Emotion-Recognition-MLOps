import os
import tensorflow as tf

# --- THE WORKAROUND ---
# Define the full path to the CUDA bin directory
cuda_bin_path = r"E:\Nvidia\CUDA\v11.2\bin"

# Add this path to the OS environment's DLL search path
# This MUST be done BEFORE importing tensorflow
try:
    os.add_dll_directory(cuda_bin_path)
    print(f"Successfully added {cuda_bin_path} to DLL search path.")
except AttributeError:
    # This function was added in Python 3.8. For older versions, you might need
    # to add the path to the system PATH environment variable manually.
    print("os.add_dll_directory not available. Ensure CUDA bin is in the system PATH.")
# --- END WORKAROUND ---


print(f"TensorFlow Version: {tf.__version__}")
print("-" * 30)

# Check for GPU devices
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpu_devices)}")
print("-" * 30)

if gpu_devices:
    print("GPU Device Details:")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"- {gpu.name}, Type: {gpu.device_type}")
    print("\nSUCCESS: TensorFlow is configured to use the GPU!")
else:
    print("\nFAILURE: TensorFlow did not detect a GPU.")


import tensorflow as tf
from tensorflow.python.client import device_lib

print("Verbose device list:")
print(device_lib.list_local_devices())