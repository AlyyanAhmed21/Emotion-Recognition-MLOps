# In src/EmotionRecognition/components/data_validation.py
import os
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            
            # Check for all required files
            for required_file in self.config.required_files:
                if not os.path.exists(required_file):
                    validation_status = False
                    logger.error(f"Missing required file: {required_file}")

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            if validation_status:
                logger.info("Data validation successful. All required files exist.")
            else:
                logger.error("Data validation failed. Please check the logs for missing files.")
            
            return validation_status

        except Exception as e:
            logger.exception(e)
            raise e