import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import ModelTrainerConfig
from pathlib import Path
import os
from EmotionRecognition.utils.common import create_mobilenetv2_model

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: dict):
        self.config = config
        self.params = params

    def _get_class_datasets(self, data_dir: Path):
        """Creates a separate tf.data.Dataset for each class directory."""
        data_params = self.params.DATA_PARAMS
        class_datasets = []
        for class_index, class_name in enumerate(data_params.CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            
            # --- THIS IS THE FIX ---
            # Change the file pattern from '*.jpg' to '*.png'
            file_paths = tf.data.Dataset.list_files(str(Path(class_dir) / '*.png'), shuffle=True)
            # --- END FIX ---
            
            def load_and_label(path):
                img = tf.io.read_file(path)
                # We also need to decode a PNG file, not a JPEG
                img = tf.image.decode_png(img, channels=1) # Load as grayscale
                label = tf.one_hot(class_index, data_params.NUM_CLASSES)
                return img, label

            class_ds = file_paths.map(load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
            class_datasets.append(class_ds)
            
        return class_datasets


    def _build_balanced_data_pipeline(self, data_dir: Path, augment: bool):
        """Builds a balanced tf.data pipeline using oversampling."""
        logger.info("Building balanced dataset with oversampling...")
        data_params = self.params.DATA_PARAMS
        
        # Get a list of datasets, one for each class
        class_datasets = self._get_class_datasets(data_dir)
        
        # Create a balanced dataset by resampling from each class dataset
        # Each class has an equal probability of being chosen
        balanced_ds = tf.data.experimental.sample_from_datasets(
            class_datasets,
            weights=[1/data_params.NUM_CLASSES] * data_params.NUM_CLASSES # Equal weights
        )

        def preprocess(image, label):
            image = tf.image.resize(image, data_params.IMAGE_SIZE, method='nearest')
            image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2), # Increase rotation
            tf.keras.layers.RandomZoom(0.2),    # Increase zoom
#            tf.keras.layers.RandomContrast(0.2),
           tf.keras.layers.RandomBrightness(0.2)
        ])

        balanced_ds = balanced_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            balanced_ds = balanced_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch the final balanced dataset
        return balanced_ds.batch(data_params.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # ...
    def _build_test_pipeline(self, data_dir: Path):
        """Builds the standard pipeline for the test set (no balancing)."""
        data_params = self.params.DATA_PARAMS
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=data_params.CLASSES, # <--- ADD THIS LINE
            image_size=data_params.IMAGE_SIZE,
            interpolation='nearest',
            batch_size=data_params.BATCH_SIZE,
            shuffle=False, # Use False for validation/test sets
            color_mode='grayscale'
        )
        def preprocess(image, label):
            image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # In model_trainer.py
    def build_and_train_model(self):
        """
        Builds the model and trains ONLY the new classifier head, keeping the
        pre-trained base model completely frozen. This is the most stable approach.
        """
        data_params = self.params.DATA_PARAMS
        training_params = self.params.TRAINING_PARAMS
        
        logger.info("Building the model with a frozen MobileNetV2 base...")
        input_shape = data_params.IMAGE_SIZE + [data_params.CHANNELS]

        model = create_mobilenetv2_model(
            input_shape=input_shape,
            num_classes=data_params.NUM_CLASSES,
            dropout_rate=training_params.DROPOUT_RATE
        )
        
        # Get a reference to the base model and ensure it's frozen
        base_model = model.layers[1]
        base_model.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_params.LEARNING_RATE),
            loss=training_params.LOSS_FUNCTION, 
            metrics=training_params.METRICS
        )
        model.summary()

        logger.info("Loading and balancing data for training...")
        train_ds = self._build_balanced_data_pipeline(self.config.train_data_dir, augment=True)
        test_ds = self._build_test_pipeline(self.config.test_data_dir)

        logger.info(f"--- Starting training for {training_params.EPOCHS} epochs ---")
        model.fit(
            train_ds, 
            epochs=training_params.EPOCHS, 
            validation_data=test_ds
        )
        
        self.model = model
        self.save_model()

    def save_model(self):
        """Saves the entire model (architecture + weights)."""
        model_path = str(self.config.trained_model_path)
        self.model.save(model_path)
        logger.info(f"Full model saved successfully to: {model_path}")