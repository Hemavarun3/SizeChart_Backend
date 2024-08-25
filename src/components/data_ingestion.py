import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd




from src.components.data_transformer import CustomTransformer
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.components.data_updation import DataUpdationConfig
from src.components.data_updation import DataUpdation

    

class DataIngestion:
    def __init__(self):
        self.data_paths = []

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:

            self.data_paths.append("artifacts\men_lower.csv")
            self.data_paths.append("artifacts\men_upper.csv")
            self.data_paths.append("artifacts\women_lower.csv")
            self.data_paths.append("artifacts\women_upper.csv")

            logging.info("Ingestion of the data is completed")

            return self.data_paths

            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":

    obj=DataIngestion()
    train_data=obj.initiate_data_ingestion()


    data_transformation=CustomTransformer()
    np_paths,_=data_transformation.initiate_data_transformation(train_data)

    modeltrainer=ModelTrainer()
    labels = modeltrainer.initiate_model_trainer(np_paths , train_data)

#    data_updation = DataUpdation()
#    data_updation.initiate_update(labels)

    logging.info("Completed Everything!")

    


