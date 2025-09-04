# File: src/EmotionRecognition/components/data_ingestion.py
import os
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def validate_source_data(self):
        """
        Validates the existence of all raw source data files and folders.
        """
        logger.info("Validating source data files and folders...")
        
        all_paths = [
            self.config.root_dir,
            self.config.ferplus_pixels_csv,
            self.config.ferplus_labels_csv,
            self.config.ckplus_dir
        ]

        for path in all_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required raw data source: {path}")

        logger.info("All raw data sources found successfully.")