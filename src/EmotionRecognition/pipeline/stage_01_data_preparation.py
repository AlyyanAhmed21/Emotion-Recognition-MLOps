# File: src/EmotionRecognition/pipeline/stage_01_data_preparation.py
from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.data_preparation import DataPreparation
from EmotionRecognition import logger

STAGE_NAME = "Data Preparation Stage"

class DataPreparationPipeline:
    def main(self):
        config_manager = ConfigurationManager()
        data_prep_config = config_manager.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_prep_config, params=config_manager.params)
        
        # --- THIS IS THE FIX ---
        # Call the correct method name from the component
        data_preparation.combine_and_prepare_data()
        # --- END FIX ---

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        pipeline = DataPreparationPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e