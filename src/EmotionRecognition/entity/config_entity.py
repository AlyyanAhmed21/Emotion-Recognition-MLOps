from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    pixels_csv_path: Path
    labels_csv_path: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    pixels_csv_path: Path
    labels_csv_path: Path
    prepared_train_dir: Path
    prepared_test_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    required_files: list 

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    train_data_dir: Path
    test_data_dir: Path
    train_dataset_path: Path
    test_dataset_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_dir: Path
    test_data_dir: Path
    trained_model_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_dir: Path
    trained_model_weights_path: Path
    metrics_file_name: Path
    mlflow_uri: str