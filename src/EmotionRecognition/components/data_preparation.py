import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataPreparationConfig

class DataPreparation:
    def __init__(self, config: DataPreparationConfig, params: dict):
        self.config = config
        self.params = params.DATA_PARAMS

    def prepare_data_folders(self):
        logger.info("Loading pixel data from fer2013.csv...")
        pixels_df = pd.read_csv(self.config.pixels_csv_path)
        
        logger.info("Loading new labels from fer2013new.csv...")
        labels_df = pd.read_csv(self.config.labels_csv_path)

        # The FER+ emotion labels order in the CSV columns
        # These are the columns we will use to find the max vote
        ferplus_emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        
        # Get our desired classes directly from params.yaml
        our_classes = self.params.CLASSES

        logger.info("Preparing and saving images to structured folders. This will take a while...")
        
        # Create output directories
        os.makedirs(self.config.prepared_train_dir, exist_ok=True)
        os.makedirs(self.config.prepared_test_dir, exist_ok=True)

        # Iterate through each row of the dataset
        for index, row in tqdm(pixels_df.iterrows(), total=len(pixels_df), desc="Processing Images"):
            # Get the new label votes for the current image from the relevant columns
            label_votes = labels_df.iloc[index][ferplus_emotion_columns].values
            
            # Find the emotion with the most votes
            # This gives the name of the column with the highest vote, e.g., 'happiness'
            best_emotion_name = ferplus_emotion_columns[np.argmax(label_votes)]
            
            # If the best label is one of our target classes
            if best_emotion_name in our_classes:
                emotion = best_emotion_name # The name is already correct
                
                # Determine if it's for training or testing
                usage = row['Usage']
                if usage == 'Training':
                    output_dir = self.config.prepared_train_dir
                elif usage == 'PublicTest': # PublicTest is our validation/test set
                    output_dir = self.config.prepared_test_dir
                else:
                    continue # Skip PrivateTest for now

                # Convert pixel string to image
                pixels = np.array(row['pixels'].split(), 'uint8')
                image = pixels.reshape((48, 48))
                pil_image = Image.fromarray(image)

                # Create the emotion-specific subfolder and save the image
                emotion_folder = os.path.join(output_dir, emotion)
                os.makedirs(emotion_folder, exist_ok=True)
                
                image_filename = f"image_{index}.png"
                pil_image.save(os.path.join(emotion_folder, image_filename))

        logger.info("Data preparation complete. Image folders are ready.")