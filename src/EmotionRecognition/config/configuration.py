from EmotionRecognition.utils.common import read_yaml, create_directories
from EmotionRecognition.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = Path("config/config.yaml"),
        params_filepath = Path("params.yaml")):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            kaggle_dataset_id=config.kaggle_dataset_id,
            unzip_dir=Path(config.unzip_dir)
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            unzip_data_dir=Path(config.unzip_data_dir)
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir), # <-- This is correct
            trained_model_path=Path(config.trained_model_path)
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            trained_model_path=Path(config.trained_model_path),
            metrics_file_name=Path(config.metrics_file_name),
            mlflow_uri=config.mlflow_uri
        )