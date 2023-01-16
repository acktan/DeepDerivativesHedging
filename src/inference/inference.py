"""Predictions on test set."""

import logging
import warnings
import torch
import pandas as pd

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
        test_X = self.load_test_data()
        predictions = self.model(test_X)