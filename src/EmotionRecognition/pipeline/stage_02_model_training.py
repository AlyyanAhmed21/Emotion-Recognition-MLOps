# File: src/EmotionRecognition/pipeline/stage_02_model_training.py

from EmotionRecognition.config.configuration import ConfigurationManager
from EmotionRecognition.components.model_trainer import ModelTrainer
from EmotionRecognition import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def main(self):
        try:
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config, params=config_manager.params)
            model_trainer.build_and_train_model()
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e