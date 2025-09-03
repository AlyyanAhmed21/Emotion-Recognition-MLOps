import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataPreprocessingConfig
from pathlib import Path

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig, params: dict):
        self.config = config
        self.params = params.DATA_PARAMS # Access the DATA_PARAMS sub-dictionary

    def _build_data_pipeline(self, data_dir: Path, augment: bool):
        """Builds a tf.data pipeline for either training or testing."""
        
        # 1. Create a dataset from the directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=self.params.IMAGE_SIZE,
            interpolation='nearest',
            batch_size=self.params.BATCH_SIZE,
            shuffle=True,
            color_mode='grayscale'
        )

        # 2. Define preprocessing steps
        def preprocess(image, label):
            # Convert grayscale to RGB
            image = tf.image.grayscale_to_rgb(image)
            
            # --- THIS IS THE FIX ---
            # Cast image to float32 before division
            image = tf.cast(image, tf.float32) 
            
            # Normalize pixel values to [0, 1]
            image = image / 255.0
            return image, label

        # 3. Define data augmentation steps (only for training)
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

        # 4. Apply preprocessing and augmentation
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                  num_parallel_calls=tf.data.AUTOTUNE)

        # 5. Prefetch for performance
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    def create_and_save_datasets(self):
        """Creates and saves the training and testing datasets."""
        
        logger.info("Starting data preprocessing for the training set...")
        train_dataset = self._build_data_pipeline(self.config.train_data_dir, augment=True)
        
        logger.info("Starting data preprocessing for the test set...")
        test_dataset = self._build_data_pipeline(self.config.test_data_dir, augment=False)
        
        # Save the datasets
        logger.info(f"Saving training dataset to: {self.config.train_dataset_path}")
        tf.data.experimental.save(train_dataset, str(self.config.train_dataset_path))
        
        logger.info(f"Saving test dataset to: {self.config.test_dataset_path}")
        tf.data.experimental.save(test_dataset, str(self.config.test_dataset_path))
        
        logger.info("Data preprocessing and saving complete.")