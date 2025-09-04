import os
from EmotionRecognition import logger
from EmotionRecognition.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig, params: dict):
        self.config = config
        self.params = params

    def validate_all_directories_exist(self) -> bool:
        try:
            validation_status = True
            main_dir = self.config.unzip_data_dir
            
            if not os.path.isdir(main_dir):
                validation_status = False
                logger.error(f"Missing main data directory: {main_dir}")
            else:
                required_classes = self.params.DATA_PARAMS.CLASSES
                found_classes = os.listdir(main_dir)
                for req_class in required_classes:
                    if req_class not in found_classes:
                        validation_status = False
                        logger.error(f"Missing required class sub-directory '{req_class}' in {main_dir}")

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            if not validation_status:
                raise Exception("Data validation failed. Check logs for missing directories.")
            
            logger.info("Data validation successful. All required directories exist.")
            return validation_status
        except Exception as e:
            raise e