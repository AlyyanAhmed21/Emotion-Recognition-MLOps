# In src/EmotionRecognition/pipeline/stage_02_data_validation.py
from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.data_validation import DataValidation
from EmotionRecognition import logger

STAGE_NAME = "Data Validation Stage"
class DataValidationTrainingPipeline:
    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            # Pass params so the component knows which class folders to look for
            data_validation = DataValidation(config=data_validation_config, params=config_manager.params)
            data_validation.validate_all_directories_exist() # Call the correct method
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e