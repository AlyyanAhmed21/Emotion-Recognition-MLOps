# In src/EmotionRecognition/config/configuration.py

from EmotionRecognition.utils.common import read_yaml, create_directories
from EmotionRecognition.entity.config_entity import DataIngestionConfig
from pathlib import Path

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