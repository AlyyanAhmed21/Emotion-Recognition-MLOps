# In src/EmotionRecognition/config/configuration.py
from pathlib import Path
from EmotionRecognition.utils.common import read_yaml, create_directories
from EmotionRecognition.entity.config_entity import (DataIngestionConfig, 
                                                     DataValidationConfig,
                                                     DataPreparationConfig,
                                                     ModelTrainerConfig,
                                                     ModelEvaluationConfig) 


# Assuming you have constants defined for file paths, or you can hardcode them here
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            pixels_csv_path=Path(config.pixels_csv_path),
            labels_csv_path=Path(config.labels_csv_path)
        )

        return data_ingestion_config
    

    def get_data_preparation_config(self) -> DataPreparationConfig:
        # ingestion_config = self.config.data_ingestion # <-- DELETE THIS LINE
        prep_config = self.config.data_preparation

        create_directories([prep_config.root_dir])

        data_preparation_config = DataPreparationConfig(
            root_dir=Path(prep_config.root_dir),
            # Read paths directly from the prep_config
            pixels_csv_path=Path(prep_config.pixels_csv_path),
            labels_csv_path=Path(prep_config.labels_csv_path),
            prepared_train_dir=Path(prep_config.prepared_train_dir),
            prepared_test_dir=Path(prep_config.prepared_test_dir)
        )
        return data_preparation_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            required_files=config.required_files # Changed
        )

        return data_validation_config

    '''def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            train_data_dir=Path(config.train_data_dir),
            test_data_dir=Path(config.test_data_dir),
            train_dataset_path=Path(config.train_dataset_path),
            test_dataset_path=Path(config.test_dataset_path)
        )

        return data_preprocessing_config'''

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_dir=Path(config.train_data_dir),
            test_data_dir=Path(config.test_data_dir),
            trained_model_path=Path(config.trained_model_path)
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_dir=Path(config.test_data_dir),
            trained_model_weights_path=Path(config.trained_model_weights_path),
            metrics_file_name=Path(config.metrics_file_name),
            mlflow_uri=config.mlflow_uri
        )

        return model_evaluation_config