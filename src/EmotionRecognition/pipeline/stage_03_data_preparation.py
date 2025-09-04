from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.data_preparation import DataPreparation
from EmotionRecognition import logger

STAGE_NAME = "Data Preparation Stage"

class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_prep_config = config_manager.get_data_preparation_config()
            data_preparation = DataPreparation(config=data_prep_config, params=config_manager.params)
            data_preparation.prepare_data_folders()
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e