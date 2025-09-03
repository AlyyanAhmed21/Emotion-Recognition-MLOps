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
            # Pass the params to the component
            data_validation_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config, params=config_manager.params)
            data_validation.validate_all_directories_exist()
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