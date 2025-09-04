import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import ModelTrainerConfig
from EmotionRecognition.utils.common import create_mobilenetv2_model
from pathlib import Path

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: dict):
        self.config = config
        self.params = params
        self.model = None

    def get_datasets(self):
        data_params = self.params.DATA_PARAMS
        logger.info("Loading prepared train and test datasets...")

        # Create a training dataset from the combined, imbalanced data
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.train_data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=data_params.CLASSES,
            image_size=data_params.IMAGE_SIZE,
            interpolation='nearest',
            batch_size=data_params.BATCH_SIZE,
            shuffle=True,
            color_mode='grayscale' # <--- ADD THIS LINE
        )

        # Create a validation/test dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.test_data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=data_params.CLASSES,
            image_size=data_params.IMAGE_SIZE,
            interpolation='nearest',
            batch_size=data_params.BATCH_SIZE,
            shuffle=False,
            color_mode='grayscale' # <--- AND ADD THIS LINE
        )
        
        def preprocess(image, label):
            # This dataset is already in PNG format, so we decode PNG
            # It's also already grayscale (1 channel)
            image = tf.image.grayscale_to_rgb(image) # Models expect 3 channels
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1)
        ])

        train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

    def build_and_train_model(self):
        data_params = self.params.DATA_PARAMS
        training_params = self.params.TRAINING_PARAMS
        
        logger.info("Building model with a frozen MobileNetV2 base...")
        input_shape = data_params.IMAGE_SIZE + [data_params.CHANNELS]

        self.model = create_mobilenetv2_model(
            input_shape=input_shape,
            num_classes=data_params.NUM_CLASSES,
            dropout_rate=training_params.DROPOUT_RATE
        )
        
        base_model = self.model.layers[1]
        base_model.trainable = False
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_params.LEARNING_RATE),
            loss=training_params.LOSS_FUNCTION, 
            metrics=training_params.METRICS
        )
        self.model.summary(print_fn=logger.info)

        train_ds, val_ds = self.get_datasets()

        logger.info(f"--- Starting training for {training_params.EPOCHS} epochs ---")
        self.model.fit(
            train_ds, 
            epochs=training_params.EPOCHS, 
            validation_data=val_ds
        )
        
        self.save_model()

    def save_model(self):
        model_path = str(self.config.trained_model_path)
        self.model.save(model_path)
        logger.info(f"Full model saved successfully to: {model_path}")