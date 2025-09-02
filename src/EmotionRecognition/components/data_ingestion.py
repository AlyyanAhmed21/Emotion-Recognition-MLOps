# In src/EmotionRecognition/components/data_ingestion.py

import os
import zipfile
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataIngestionConfig
from pathlib import Path
import kaggle  # Make sure the kaggle library is installed and authenticated

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset from Kaggle if it doesn't already exist.
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info("Dataset zip file not found. Downloading from Kaggle...")
            # Note: This uses the kaggle API. Ensure kaggle.json is set up.
            kaggle.api.dataset_download_files(
                self.config.kaggle_dataset_id,
                path=self.config.root_dir,
                unzip=False # We will unzip manually
            )
            # Kaggle API downloads with a different name, let's rename it
            # The downloaded file is typically dataset-name.zip
            downloaded_file_path = os.path.join(self.config.root_dir, f"{self.config.kaggle_dataset_id.split('/')[1]}.zip")
            os.rename(downloaded_file_path, self.config.local_data_file)
            logger.info(f"Dataset downloaded and saved as {self.config.local_data_file}")
        else:
            logger.info(f"Dataset zip file already exists at: {self.config.local_data_file}")


    def unzip_file(self):
        """
        Unzips the downloaded data file.
        """
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
            logger.info(f"Successfully unzipped data into: {self.config.unzip_dir}")