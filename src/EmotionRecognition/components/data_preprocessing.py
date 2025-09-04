# File: src/EmotionRecognition/components/data_preprocessing.py
import os
import shutil
import random
import glob
from tqdm import tqdm
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataPreprocessingConfig
from pathlib import Path

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig, params: dict):
        self.config = config
        self.params = params.DATA_PARAMS

    def _log_and_get_stats(self, directory):
        """Helper to get and log image counts for a directory."""
        stats = {}
        logger.info(f"Statistics for directory: {directory}")
        for emotion in sorted(self.params.CLASSES):
            path = Path(directory) / emotion
            count = len(glob.glob(str(path / '*.png')))
            stats[emotion] = count
            logger.info(f"- {emotion}: {count} images")
        return stats

    def balance_dataset(self):
        """
        Applies a hybrid oversampling and undersampling strategy to balance the training data.
        """
        logger.info("--- Starting Hybrid Data Balancing Stage ---")
        
        logger.info("Source Training Set Distribution:")
        self._log_and_get_stats(self.config.source_train_dir)
        
        if os.path.exists(self.config.balanced_train_dir): shutil.rmtree(self.config.balanced_train_dir)
        os.makedirs(self.config.balanced_train_dir, exist_ok=True)

        target_count = self.config.target_samples_per_class
        logger.info(f"\nBalancing all training classes to {target_count} samples each...")

        for emotion in tqdm(self.params.CLASSES, desc="Balancing Classes"):
            source_emotion_dir = Path(self.config.source_train_dir) / emotion
            dest_emotion_dir = Path(self.config.balanced_train_dir) / emotion
            dest_emotion_dir.mkdir(parents=True, exist_ok=True)
            
            image_files = os.listdir(source_emotion_dir)
            
            if not image_files:
                logger.warning(f"No images found for class '{emotion}'. Skipping.")
                continue

            current_count = len(image_files)

            if current_count > target_count:
                # Undersampling: Randomly select 'target_count' unique images
                selected_files = random.sample(image_files, target_count)
            else:
                # Oversampling: Select with replacement to reach 'target_count'
                selected_files = random.choices(image_files, k=target_count)

            # --- THIS IS THE BUG FIX ---
            # Copy the selected files, giving duplicates new names.
            for i, filename in enumerate(selected_files):
                # Get the original file's extension
                base_name, extension = os.path.splitext(filename)
                
                # If oversampling, create a unique name for each copy to prevent overwriting
                if current_count < target_count:
                    dest_filename = f"{base_name}_copy{i}{extension}"
                else:
                    dest_filename = filename # For undersampling, names are already unique

                shutil.copy(source_emotion_dir / filename, dest_emotion_dir / dest_filename)
            # --- END BUG FIX ---

        # Copy the test set without changes
        logger.info("\nCopying test set...")
        if os.path.exists(self.config.balanced_test_dir): shutil.rmtree(self.config.balanced_test_dir)
        shutil.copytree(self.config.source_test_dir, self.config.balanced_test_dir)
        
        logger.info("\n--- Final Balanced Dataset Statistics ---")
        self._log_and_get_stats(self.config.balanced_train_dir)
        self._log_and_get_stats(self.config.balanced_test_dir)

        logger.info("--- Data Balancing Stage Complete ---")