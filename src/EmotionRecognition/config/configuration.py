from EmotionRecognition.utils.common import read_yaml, create_directories
from EmotionRecognition.entity.config_entity import (
    DataPreparationConfig,
    DataPreprocessingConfig,
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

    def get_data_preparation_config(self) -> DataPreparationConfig:
        prep_config = self.config.data_preparation
        # Note: Raw data paths are now defined in data_preparation, not a separate ingestion config
        create_directories([prep_config.root_dir])
        return DataPreparationConfig(
            root_dir=Path(prep_config.root_dir),
            ferplus_pixels_csv=Path(prep_config.ferplus_pixels_csv),
            ferplus_labels_csv=Path(prep_config.ferplus_labels_csv),
            ckplus_dir=Path(prep_config.ckplus_dir),
            combined_train_dir=Path(prep_config.combined_train_dir),
            ferplus_test_dir=Path(prep_config.ferplus_test_dir)
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        preprocess_config = self.config.data_preprocessing
        create_directories([preprocess_config.root_dir])
        return DataPreprocessingConfig(
            root_dir=Path(preprocess_config.root_dir),
            source_train_dir=Path(preprocess_config.source_train_dir),
            source_test_dir=Path(preprocess_config.source_test_dir),
            balanced_train_dir=Path(preprocess_config.balanced_train_dir),
            balanced_test_dir=Path(preprocess_config.balanced_test_dir),
            target_samples_per_class=preprocess_config.target_samples_per_class
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_dir=Path(config.train_data_dir),
            test_data_dir=Path(config.test_data_dir),
            trained_model_path=Path(config.trained_model_path)
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_dir=Path(config.test_data_dir), # Corrected from data_dir
            trained_model_path=Path(config.trained_model_path),
            metrics_file_name=Path(config.metrics_file_name),
            mlflow_uri=config.mlflow_uri
        )