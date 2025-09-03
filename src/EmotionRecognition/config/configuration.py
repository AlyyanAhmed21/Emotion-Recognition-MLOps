# In src/EmotionRecognition/config/configuration.py
from pathlib import Path
from EmotionRecognition.utils.common import read_yaml, create_directories
from EmotionRecognition.entity.config_entity import (DataIngestionConfig,
                                                    DataValidationConfig,
                                                    ModelTrainerConfig)


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
            kaggle_dataset_id=config.kaggle_dataset_id,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            unzip_data_dir=Path(config.unzip_data_dir),
            status_file=Path(config.status_file)
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