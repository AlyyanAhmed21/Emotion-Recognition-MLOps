import os
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataValidationConfig
import pandas as pd # You might need to add pandas to requirements.txt

class DataValidation:
    def __init__(self, config: DataValidationConfig, params: dict):
        self.config = config
        self.params = params

    def validate_all_directories_exist(self) -> bool:
        try:
            validation_status = True
            
            # The required classes from params.yaml
            required_classes = self.params.DATA_PARAMS.CLASSES
            
            # The main directories to check
            data_dirs_to_check = ['train', 'test']

            for data_dir in data_dirs_to_check:
                full_path = os.path.join(self.config.unzip_data_dir, data_dir)
                if not os.path.isdir(full_path):
                    validation_status = False
                    logger.error(f"Missing required directory: {full_path}")
                else:
                    # Check for all class subdirectories
                    found_classes = os.listdir(full_path)
                    for req_class in required_classes:
                        if req_class not in found_classes:
                            validation_status = False
                            logger.error(f"Missing required class sub-directory '{req_class}' in {full_path}")

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            if validation_status:
                logger.info("Data validation successful. All required directories and sub-directories exist.")
            else:
                logger.error("Data validation failed. Please check the logs for missing directories.")
            
            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e