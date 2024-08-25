import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataUpdationConfig:
    train_data_path: str=os.path.join('artifacts',"men_upper.csv")


class DataUpdation:
    def __init__(self):
        self.ingestion_config=DataUpdationConfig()
    
    def initiate_update(self , labels):
        
        logging.info("Entered the data ingestion method or component")

        try:
            df=pd.read_csv('artifacts/men_upper_without_outliers.csv')
            df['Cluster'] = labels
            df.to_csv('artifacts/labels_updated.csv' , index=False)
            logging.info("Successfully updated!")


        except Exception as e:
            raise CustomException(e,sys)