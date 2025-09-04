import os
import shutil
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataIngestionConfig
import kaggle
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_prepare_dataset(self):
        final_output_path = self.config.unzip_dir / 'CK+'
        
        if not os.path.exists(final_output_path):
            logger.info("Dataset not found. Downloading and preparing from Kaggle...")
            
            temp_download_dir = self.config.root_dir / "temp_download"
            os.makedirs(temp_download_dir, exist_ok=True)
            
            kaggle.api.dataset_download_files(
                self.config.kaggle_dataset_id,
                path=temp_download_dir,
                unzip=True
            )
            
            source_path = temp_download_dir / 'CK+48'
            
            if os.path.exists(source_path):
                logger.info(f"Found data folder at '{source_path}'. Moving it to final destination.")
                shutil.move(str(source_path), str(final_output_path))
            else:
                logger.error(f"Could not find the expected 'CK+48' folder inside the unzipped data at '{temp_download_dir}'.")
                shutil.rmtree(temp_download_dir)
                raise FileNotFoundError("Could not process the downloaded dataset structure.")

            shutil.rmtree(temp_download_dir)
            logger.info(f"Dataset successfully prepared at: {final_output_path}")
        else:
            logger.info(f"Dataset already exists at: {final_output_path}")