import os
import sys
from dataclasses import dataclass
import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path="artifacts/pickle_files"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, np_paths , train_data):

        for path in np_paths:    
        
            try:
                logging.info("Preparing for clustering")
                scaled_array = np.load(path) 
                # Number of clusters
                n_clusters = 4  # You can adjust the number of clusters as needed
                
                # Initialize KMeans model
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                
                # Fit the model to the training data
                kmeans.fit(scaled_array)
                
                # Predict cluster labels for the training data
                cluster_labels = kmeans.predict(scaled_array)
                
                # Evaluate the clustering performance using silhouette score
                silhouette_avg = silhouette_score(scaled_array, cluster_labels)
                
                logging.info(f"Silhouette Score for K-means clustering: {silhouette_avg}")

                # Save the trained KMeans model
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path + path + '.pkl',
                    obj=kmeans
                )
                
                # Return the cluster_labels
                #return??

            except Exception as e:
                raise CustomException(e, sys)   