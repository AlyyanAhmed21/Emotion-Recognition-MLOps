from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.data_preprocessing import DataPreprocessing
from EmotionRecognition import logger

STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_preprocessing_config = config_manager.get_data_preprocessing_config()
            data_preprocessing = DataPreprocessing(config=data_preprocessing_config, params=config_manager.params)
            data_preprocessing.create_and_save_datasets()
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e