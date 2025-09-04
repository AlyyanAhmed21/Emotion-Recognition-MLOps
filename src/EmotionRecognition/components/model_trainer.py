import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import ModelTrainerConfig
from EmotionRecognition.utils.common import create_mobilenetv2_model
from pathlib import Path
import glob # For finding files

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: dict):
        self.config = config
        self.params = params
        self.model = None

    def get_train_validation_datasets_manual(self):
        """
        Manually scans for files and builds robust tf.data.Datasets.
        This bypasses the buggy image_dataset_from_directory behavior.
        """
        data_params = self.params.DATA_PARAMS
        logger.info("Manually scanning for image files and creating datasets...")

        all_filepaths = []
        all_labels = []

        # Create a mapping from class name to integer index
        class_to_idx = {name: i for i, name in enumerate(data_params.CLASSES)}

        # Manually find all .png files in each class subdirectory
        for class_name in data_params.CLASSES:
            class_dir = Path(self.config.data_dir) / class_name
            filepaths = glob.glob(str(class_dir / '*.png'))
            all_filepaths.extend(filepaths)
            # Create a label for each file found
            all_labels.extend([class_to_idx[class_name]] * len(filepaths))

        if not all_filepaths:
            raise ValueError("No image files found in the data directory. Check the path and file extensions.")
        
        logger.info(f"Found a total of {len(all_filepaths)} images belonging to {len(data_params.CLASSES)} classes.")

        # Create a full dataset from the file paths and labels
        full_dataset = tf.data.Dataset.from_tensor_slices((all_filepaths, all_labels))
        
        # --- Shuffle and Split the dataset ---
        full_dataset = full_dataset.shuffle(buffer_size=len(all_filepaths), seed=123, reshuffle_each_iteration=False)
        
        dataset_size = len(all_filepaths)
        val_size = int(data_params.VALIDATION_SPLIT * dataset_size)
        
        val_ds = full_dataset.take(val_size)
        train_ds = full_dataset.skip(val_size)

        # --- Preprocessing and Augmentation ---
        def load_and_preprocess(filepath, label):
            img = tf.io.read_file(filepath)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, data_params.IMAGE_SIZE, method='nearest')
            # One-hot encode the label
            label = tf.one_hot(label, data_params.NUM_CLASSES)
            return img, label
            
        def finalize_dataset(dataset, augment=False):
            dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

            def preprocess_final(image, label):
                image = tf.image.grayscale_to_rgb(image)
                image = tf.cast(image, tf.float32) / 255.0
                return image, label
            
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1)
            ])

            dataset = dataset.map(preprocess_final, num_parallel_calls=tf.data.AUTOTUNE)
            if augment:
                dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            
            return dataset.batch(data_params.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        train_ds = finalize_dataset(train_ds, augment=True)
        val_ds = finalize_dataset(val_ds, augment=False)
        
        return train_ds, val_ds

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

        # Call our new manual data loader
        train_ds, val_ds = self.get_train_validation_datasets_manual()

        logger.info(f"--- Starting training for {training_params.EPOCHS} epochs ---")
        self.model.fit(train_ds, epochs=training_params.EPOCHS, validation_data=val_ds)
        self.save_model()

    def save_model(self):
        model_path = str(self.config.trained_model_path)
        self.model.save(model_path)
        logger.info(f"Full model saved successfully to: {model_path}")