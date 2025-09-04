from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    # Inputs from raw data
    ferplus_pixels_csv: Path
    ferplus_labels_csv: Path
    ckplus_dir: Path
    # Outputs of this stage
    combined_train_dir: Path
    ferplus_test_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    # Inputs from the Data Preparation stage
    source_train_dir: Path
    source_test_dir: Path
    # Outputs of this stage
    balanced_train_dir: Path
    balanced_test_dir: Path
    # Parameter for the balancing strategy
    target_samples_per_class: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    # Inputs from the Data Preprocessing stage
    train_data_dir: Path
    test_data_dir: Path
    # Output of this stage
    trained_model_path: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_dir: Path
    trained_model_path: Path # <-- Make sure this is the name used
    metrics_file_name: Path
    mlflow_uri: str