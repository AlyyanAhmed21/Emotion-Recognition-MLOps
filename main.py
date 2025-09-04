from EmotionRecognition import logger
from EmotionRecognition.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from EmotionRecognition.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from EmotionRecognition.pipeline.stage_03_data_preparation import DataPreparationPipeline
from EmotionRecognition.pipeline.stage_04_model_training import ModelTrainingPipeline
from EmotionRecognition.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline 

# Data Ingestion Stage
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Data Validation Stage
STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Data Preprocessing Stage
#STAGE_NAME = "Data Preprocessing Stage"
#try:
#    logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
#    obj = DataPreprocessingTrainingPipeline()
#    obj.main()
#    logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
#except Exception as e:
#    logger.exception(e)
#    raise e

STAGE_NAME = "Data Preparation Stage"
try:
    logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
    obj = DataPreparationPipeline()
    obj.main()
    logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Model Evaluation Stage
STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> Stage '{STAGE_NAME}' started <<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> Stage '{STAGE_NAME}' completed successfully <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e