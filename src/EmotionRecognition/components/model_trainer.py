import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import ModelTrainerConfig
from pathlib import Path
import os

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
            
            # Create a dataset for the current class
            # We use list_files and map to handle empty or small directories gracefully
            file_paths = tf.data.Dataset.list_files(str(Path(class_dir) / '*.jpg'), shuffle=True)
            
            def load_and_label(path):
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=1) # Load as grayscale
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
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

        balanced_ds = balanced_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            balanced_ds = balanced_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch the final balanced dataset
        return balanced_ds.batch(data_params.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


    def _build_test_pipeline(self, data_dir: Path):
        """Builds the standard pipeline for the test set (no balancing)."""
        data_params = self.params.DATA_PARAMS
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir, labels='inferred', label_mode='categorical',
            image_size=data_params.IMAGE_SIZE, interpolation='nearest',
            batch_size=data_params.BATCH_SIZE, shuffle=True, color_mode='grayscale'
        )
        def preprocess(image, label):
            image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)


    def build_and_train_model(self):
        """Builds, trains, and saves the model."""
        data_params = self.params.DATA_PARAMS
        training_params = self.params.TRAINING_PARAMS

        logger.info("Building the model with MobileNetV2...")
        input_shape = data_params.IMAGE_SIZE + [data_params.CHANNELS]

        # --- THIS IS THE KEY CHANGE ---
        # Switch from EfficientNetB0 to MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights='imagenet'
        )
        # --- END CHANGE ---

        base_model.trainable = False

        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # We can keep the regularized Dense layer as it's good practice
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(training_params.DROPOUT_RATE)(x)

        outputs = tf.keras.layers.Dense(data_params.NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_params.LEARNING_RATE),
            loss=training_params.LOSS_FUNCTION, metrics=training_params.METRICS
        )

        logger.info("Loading and balancing data for training...")
        train_ds = self._build_balanced_data_pipeline(self.config.train_data_dir, augment=True)
        test_ds = self._build_test_pipeline(self.config.test_data_dir)

        initial_epochs = 5
        logger.info(f"--- Starting initial training for {initial_epochs} epochs ---")
        model.fit(train_ds, epochs=initial_epochs, validation_data=test_ds)

        logger.info("--- Starting fine-tuning ---")
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 20
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_params.LEARNING_RATE / 100),
            loss=training_params.LOSS_FUNCTION, metrics=training_params.METRICS
        )

        total_epochs = training_params.EPOCHS
        model.fit(
            train_ds, epochs=total_epochs,
            initial_epoch=initial_epochs, validation_data=test_ds
        )
        self.model = model
        self.save_model()

    def save_model(self):
        """Saves model weights and architecture separately to avoid serialization errors."""
        model_path = str(self.config.trained_model_path)
        # Use the .weights.h5 format for saving, which is more robust
        self.model.save_weights(model_path)
        logger.info(f"Model weights saved successfully to: {model_path}")