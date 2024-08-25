import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self , features):
        
        try:
            numerical_columns =  features
            

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )


            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path):

        try:
            train_df=pd.read_csv(train_path)

            logging.info("Read data")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            numerical_columns = ['sho_gi'	,'che_gi'	,'wai_gi'	,'nav_gi']

            logging.info("Removing outliers from training data")
            train_df = self.remove_outliers(train_df, numerical_columns)
        
            logging.info(
                f"Applying preprocessing object on training dataframe"
            )

            folder_path = 'C:/Users/Ram/Desktop/Code/Projects/end-to-end/artifacts'  # Update with your target folder path
            file_name = 'men_upper_without_outliers.csv'
            file_path = os.path.join(folder_path, file_name)
            train_df.to_csv(file_path, index=False)
            
            train_arr=preprocessing_obj.fit_transform(train_df)            

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

    
    def remove_outliers(self, df, numerical_columns, threshold=1.5):
        try:
            if not all(col in df.columns for col in numerical_columns):
                raise ValueError("Some columns in 'numerical_columns' do not exist in the DataFrame.")

            for col in numerical_columns:
                if df[col].dtype not in [np.float64, np.int64]:
                    raise TypeError(f"Column {col} is not numeric.")

                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Determine outlier thresholds
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)

                # Log outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    for idx, outlier_value in outliers[col].items():
                        logging.info(f"Outlier detected in column '{col}' at index {idx}: {outlier_value}")

                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            return df

        except Exception as e:
            logging.error(f"Error in remove_outliers: {e}")
            raise CustomException(e, sys)

