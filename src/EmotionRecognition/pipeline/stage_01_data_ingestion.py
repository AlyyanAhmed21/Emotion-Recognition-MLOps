from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.data_ingestion import DataIngestion
from EmotionRecognition import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # --- THIS IS THE FIX ---
        # Change the method name to match the component
        data_ingestion.download_and_prepare_dataset()
        # --- END FIX ---

# This boilerplate is needed to make the script runnable
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e