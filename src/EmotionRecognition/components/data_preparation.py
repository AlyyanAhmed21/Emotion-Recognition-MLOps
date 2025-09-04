# File: src/EmotionRecognition/components/data_preparation.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataPreparationConfig
from pathlib import Path
import glob

class DataPreparation:
    def __init__(self, config: DataPreparationConfig, params: dict):
        self.config = config
        self.params = params.DATA_PARAMS

    def _process_and_save(self, best_emotion_name, usage, index, pixels, dataset_prefix):
        """Helper function to handle the merging logic and save images."""
        
        # --- MERGING LOGIC ---
        # If the emotion is 'contempt', we re-label it as 'disgust'.
        if best_emotion_name == 'contempt':
            final_emotion_name = 'disgust'
        else:
            final_emotion_name = best_emotion_name
        # --- END MERGING LOGIC ---
            
        # Check if this emotion is one of our final target classes
        if final_emotion_name in self.params.CLASSES:
            if usage == 'Training':
                output_dir = self.config.combined_train_dir
            elif usage == 'PublicTest':
                output_dir = self.config.ferplus_test_dir
            else:
                return # Skip other usages like PrivateTest

            image = Image.fromarray(pixels)
            emotion_folder = Path(output_dir) / final_emotion_name
            emotion_folder.mkdir(parents=True, exist_ok=True)
            image.save(emotion_folder / f"{dataset_prefix}_{index}.png")

    def _prepare_ferplus(self):
        logger.info("Starting preparation of FER+ dataset...")
        pixels_df = pd.read_csv(self.config.ferplus_pixels_csv)
        labels_df = pd.read_csv(self.config.ferplus_labels_csv)
        
        ferplus_emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        
        for index, row in tqdm(pixels_df.iterrows(), total=len(pixels_df), desc="Processing FER+ Images"):
            label_votes = labels_df.iloc[index][ferplus_emotion_columns].values
            source_emotion_name = ferplus_emotion_columns[np.argmax(label_votes)]

            # --- STANDARDIZE THE NAME ---
            # Default to the source name
            our_emotion_name = source_emotion_name
            if source_emotion_name == 'happiness': our_emotion_name = 'happy'
            if source_emotion_name == 'sadness': our_emotion_name = 'sad'
            if source_emotion_name == 'anger': our_emotion_name = 'angry'
            if source_emotion_name == 'contempt': our_emotion_name = 'disgust' # MERGE
            
            if our_emotion_name in self.params.CLASSES:
                usage = row['Usage']
                if usage == 'Training': output_dir = self.config.combined_train_dir
                elif usage == 'PublicTest': output_dir = self.config.ferplus_test_dir
                else: continue

                pixels = np.array(row['pixels'].split(), 'uint8').reshape((48, 48))
                image = Image.fromarray(pixels)
                emotion_folder = Path(output_dir) / our_emotion_name
                emotion_folder.mkdir(parents=True, exist_ok=True)
                image.save(emotion_folder / f"ferplus_{index}.png")
                
        logger.info("FER+ dataset preparation complete.")

    def _prepare_ckplus(self):
        logger.info("Starting preparation of CK+ dataset...")
        
        for ckplus_folder_name in tqdm(os.listdir(self.config.ckplus_dir), desc="Processing CK+ Folders"):
            source_emotion_dir = Path(self.config.ckplus_dir) / ckplus_folder_name
            
            # --- STANDARDIZE THE NAME ---
            our_emotion_name = ckplus_folder_name # Default
            if ckplus_folder_name == 'contempt': our_emotion_name = 'disgust' # MERGE
            
            if our_emotion_name in self.params.CLASSES and source_emotion_dir.is_dir():
                dest_emotion_dir = Path(self.config.combined_train_dir) / our_emotion_name
                dest_emotion_dir.mkdir(parents=True, exist_ok=True)
                
                for img_file in os.listdir(source_emotion_dir):
                    shutil.copy(source_emotion_dir / img_file, dest_emotion_dir / f"ckplus_{img_file}")
        
        logger.info("CK+ dataset preparation complete.")
        
    def _log_dataset_statistics(self):
        logger.info("--- Final Dataset Statistics ---")
        logger.info("Training Set:")
        for emotion in sorted(self.params.CLASSES):
            count = len(glob.glob(str(self.config.combined_train_dir / emotion / '*.png')))
            logger.info(f"- {emotion}: {count} images")
        
        logger.info("\nTest Set:")
        for emotion in sorted(self.params.CLASSES):
            count = len(glob.glob(str(self.config.ferplus_test_dir / emotion / '*.png')))
            logger.info(f"- {emotion}: {count} images")
        logger.info("---------------------------------")

    def combine_and_prepare_data(self):
        logger.info("--- Starting Data Preparation Stage ---")
        if os.path.exists(self.config.combined_train_dir): shutil.rmtree(self.config.combined_train_dir)
        if os.path.exists(self.config.ferplus_test_dir): shutil.rmtree(self.config.ferplus_test_dir)
        os.makedirs(self.config.combined_train_dir, exist_ok=True)
        os.makedirs(self.config.ferplus_test_dir, exist_ok=True)
        
        self._prepare_ferplus()
        self._prepare_ckplus()
        self._log_dataset_statistics()
        logger.info("--- Data Preparation Stage Complete ---")