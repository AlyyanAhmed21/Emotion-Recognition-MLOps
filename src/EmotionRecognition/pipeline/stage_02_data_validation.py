from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.data_validation import DataValidation
from EmotionRecognition import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            # We don't need params for this simple check anymore
            data_validation = DataValidation(config=data_validation_config) 
            data_validation.validate_all_files_exist() # Call the new method
        except Exception as e:
            logger.exception(e)
            raise e
# ...

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e