import tensorflow as tf
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
import mlflow.keras
from EmotionRecognition.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, params: dict):
        self.config = config
        self.params = params

    def get_validation_dataset(self):
        """
        Loads the prepared validation/test dataset from disk.
        """
        data_params = self.params.DATA_PARAMS
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.test_data_dir,
            labels='inferred',
            label_mode='categorical',
            class_names=data_params.CLASSES,
            image_size=data_params.IMAGE_SIZE,
            interpolation='nearest',
            batch_size=data_params.BATCH_SIZE,
            shuffle=False,
            color_mode='grayscale' # <--- THIS IS THE FIX
        )
        
        def preprocess(image, label):
            image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
            
        return val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def evaluate_and_log(self):
        logger.info("Preparing validation dataset for evaluation...")
        val_ds = self.get_validation_dataset()
        
        logger.info("Loading full trained model from disk...")
        model = tf.keras.models.load_model(str(self.config.trained_model_path))
        
        logger.info("Evaluating model...")
        score = model.evaluate(val_ds)
        
        scores = {"loss": score[0], "accuracy": score[1]}
        logger.info(f"Evaluation scores: {scores}")
        save_json(path=self.config.metrics_file_name, data=scores)
        
        logger.info("Starting MLflow logging...")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Emotion Recognition Experiment")

        with mlflow.start_run():
            mlflow.log_params(self.params.DATA_PARAMS)
            mlflow.log_params(self.params.TRAINING_PARAMS)
            mlflow.log_metrics(scores)
            mlflow.keras.log_model(model, "model")

        logger.info("MLflow logging complete.")