import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self, gender: str, category: str):
        self.gender = gender.lower()
        self.category = category.lower()

    def get_model_and_preprocessor_paths(self):
        """
        Determine model and preprocessor paths based on gender and category.
        """
        try:
            if self.gender == "male" and self.category == "top":
                model_path = os.path.join("artifacts", "men_upper.csv.npy.pkl")
                preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            elif self.gender == "male" and self.category == "bottom":
                model_path = os.path.join("artifacts", "men_lower.csv.npy.pkl")
                preprocessor_path = os.path.join("artifacts", "proprocessor_bottom.pkl")
            elif self.gender == "female" and self.category == "top":
                model_path = os.path.join("artifacts", "women_upper.csv.npy.pkl")
                preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            elif self.gender == "female" and self.category == "bottom":
                model_path = os.path.join("artifacts", "women_lower.csv.npy.pkl")
                preprocessor_path = os.path.join("artifacts", "proprocessor_bottom.pkl")
            else:
                raise CustomException("Invalid gender or category provided.", sys)
            return model_path, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            # Get the model and preprocessor paths
            model_path, preprocessor_path = self.get_model_and_preprocessor_paths()
            print("Before Loading")

            # Load the model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            print(features)
            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, category: str, **kwargs):
        self.gender = gender
        self.category = category
        self.attributes = kwargs

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary based on the attributes
            data_dict = {key: [value] for key, value in self.attributes.items()}

            # Return as a DataFrame
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)
