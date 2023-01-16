"""Predictions on test set."""

import logging
import warnings
import torch
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

class Inference():
    """Load test data, predict data and save predictions.

    Args:
        conf: the config file
    Returns:
        train_loader, val_loader: torch dataloaders for training and validation sets.   
    """
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
    
    def load_test_data(self):
        """Load test data.
        
        Args:
            None.
        Returns:
            The test dataset.
        """
        directory = self.conf["paths"]["directory"]
        input_folder = self.conf["paths"]["input_folder"]
        path_test = self.conf["paths"]["test_data_bs"]
        test_set = pd.read_csv(directory + input_folder + path_test, header=None)
        test_X = np.array(test_set.iloc[:,:30].values.astype('float32'))
        test_X = torch.Tensor(test_X)
        return test_X
    
    def predict(self):
        """Make predictions and save in dataframe.
        
        Args:
            None.
        Returns:
            The predictions dataframe.
        """
        test_X = self.load_test_data()
        test_X = test_X.unsqueeze(-1)
        test_prediction = self.model(test_X)
        predictions = pd.DataFrame(test_prediction.detach().numpy().transpose())
        return predictions
    
    def save_predictions(self):
        """Make and save predictions in a csv file.
        
        Args:
            None.
        Returns:
            None.
        """
        predictions = self.predict()
        directory = self.conf["paths"]["directory"]
        output_folder = self.conf["paths"]["output_folder"]
        inference_folder = self.conf["paths"]["inference_folder"]
        output_file = self.conf["paths"]["output_file"]
        predictions.to_csv(directory + output_folder +
                           inference_folder + output_file,
                           header=None, index=False)
        return None
        