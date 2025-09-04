# In src/EmotionRecognition/components/data_ingestion.py
import os
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def check_files_exist(self):
        """
        Checks if the required CSV files exist.
        """
        logger.info("Checking for required data files...")
        
        pixels_path = self.config.pixels_csv_path
        labels_path = self.config.labels_csv_path

        if not os.path.exists(pixels_path):
            logger.error(f"Pixel data file not found at: {pixels_path}")
            raise FileNotFoundError(f"Pixel data file not found at: {pixels_path}. Please download fer2013.csv.")
        
        if not os.path.exists(labels_path):
            logger.error(f"Label data file not found at: {labels_path}")
            raise FileNotFoundError(f"Label data file not found at: {labels_path}. Please download fer2013new.csv.")

        logger.info("All required data files found.")