import os
from pathlib import Path
import logging

# Configure logging to provide feedback during script execution
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the project name
project_name = "EmotionRecognition"

# List of files and directories to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_preprocessing.py",
    f"src/{project_name}/pipeline/stage_04_model_training.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/01_data_exploration.ipynb",
    "templates/index.html", # For a simple web UI if needed
    ".dvcignore",
    ".gitignore"
]

# Loop through the list to create the files and directories
for filepath_str in list_of_files:
    filepath = Path(filepath_str)
    filedir, filename = os.path.split(filepath)

    # 1. Create the directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # 2. Create the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

logging.info("Project structure creation complete!")