import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
import mlflow.keras
from EmotionRecognition.utils.common import read_yaml, create_directories, save_json, create_mobilenetv2_model
import numpy as np

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, params: dict):
        self.config = config
        self.params = params

    def _build_test_pipeline(self, data_dir: Path):
        """Builds the standard pipeline for the test set."""
        data_params = self.params.DATA_PARAMS
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=data_params.CLASSES, # <--- ADD THIS LINE
            image_size=data_params.IMAGE_SIZE,
            interpolation='nearest',
            batch_size=data_params.BATCH_SIZE,
            shuffle=False, 
            color_mode='grayscale'
        )
        def preprocess(image, label):
            image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        return dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    def _build_model(self):
        """Rebuilds the model architecture using the shared function."""
        data_params = self.params.DATA_PARAMS
        training_params = self.params.TRAINING_PARAMS
        input_shape = data_params.IMAGE_SIZE + [data_params.CHANNELS]
        
        # --- REPLACE MODEL BUILDING LOGIC ---
        model = create_mobilenetv2_model(
            input_shape=input_shape,
            num_classes=data_params.NUM_CLASSES,
            dropout_rate=training_params.DROPOUT_RATE,
            is_training=False
        )
        # --- END REPLACEMENT ---
        return model

    def evaluate_and_log(self):
        """Evaluates the model and logs the experiment to MLflow."""
        logger.info("Preparing test dataset...")
        test_ds = self._build_test_pipeline(self.config.test_data_dir)
        
        logger.info("Loading full trained model from disk...")
        # This one line replaces rebuilding and loading weights separately
        model = tf.keras.models.load_model(str(self.config.trained_model_weights_path))
        
        logger.info("Evaluating model on test set...")
        score = model.evaluate(test_ds)
        
        scores = {"loss": score[0], "accuracy": score[1]}
        logger.info(f"Evaluation scores: {scores}")
        
        save_json(path=self.config.metrics_file_name, data=scores)
        
        # --- MLflow Logging ---
        logger.info("Starting MLflow logging...")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Emotion Recognition Experiment")

        with mlflow.start_run():
            # Log all hyperparameters from params.yaml
            mlflow.log_params(self.params.DATA_PARAMS)
            mlflow.log_params(self.params.TRAINING_PARAMS)
            
            # Log the final evaluation metrics
            mlflow.log_metrics(scores)
            
            # Log the model itself. MLflow's Keras integration is excellent.
            # It will save the model in a way that can be easily loaded later.
            mlflow.keras.log_model(model, "model")

        logger.info("MLflow logging complete.")