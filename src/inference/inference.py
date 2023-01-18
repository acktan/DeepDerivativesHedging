"""Predictions on test set."""

import logging
import warnings
import torch
import pandas as pd
import numpy as np
from src.utils.utils import prepare_input
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
    
    def load_test_data_bs(self):
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
    
    def load_test_data_hest(self):
        """Load test data.
        
        Args:
            None.
        Returns:
            The test dataset.
        """
        directory = self.conf["paths"]["directory"]
        input_folder = self.conf["paths"]["input_folder"]
        path_S = self.conf["paths"]["test_data_hest_1"]
        path_var = self.conf["paths"]["test_data_hest_2"]
        test_S = pd.read_csv(directory + input_folder + path_S, header=None)
        test_var= pd.read_csv(directory + input_folder + path_var, header=None)
        test_S = np.array(test_S.iloc[:,:30].values.astype('float32'))
        test_var = np.array(test_var.iloc[:,:30].values.astype('float32'))
        test_S = torch.Tensor(test_S)
        test_var = torch.Tensor(test_var)
        return test_S, test_var
    
    def predict(self):
        """Make predictions and save in dataframe.
        
        Args:
            None.
        Returns:
            The predictions dataframe.
        """
        if self.conf["model_init"]["bs_model"]:
            test_X = self.load_test_data_bs()
            test_X = test_X.unsqueeze(-1)
            test_X = prepare_input(test_X)
            test_prediction = self.model(test_X)
            predictions = pd.DataFrame(test_prediction.detach().numpy().transpose())
            return predictions
        else:
            test_S, test_var = self.load_test_data_hest()
            test_S = prepare_input(test_S.unsqueeze(-1))
            test_var = prepare_input(test_var.unsqueeze(-1))
            test_X = torch.cat((test_S, test_var), 2)
            test_prediction = self.model(test_X)
            predictions_S = pd.DataFrame(test_prediction[:,:,0].detach().numpy().transpose())
            predictions_V = pd.DataFrame(test_prediction[:,:,1].detach().numpy().transpose())
            return predictions_S, predictions_V
    
    def save_predictions(self):
        """Make and save predictions in a csv file.
        
        Args:
            None.
        Returns:
            None.
        """
        directory = self.conf["paths"]["directory"]
        output_folder = self.conf["paths"]["output_folder"]
        inference_folder = self.conf["paths"]["inference_folder"]
        output_file = self.conf["paths"]["output_file"]
        if self.conf["model_init"]["bs_model"]:
            output_file = output_file + "_bs" + ".csv"
            predictions = self.predict()
            predictions.to_csv(directory + output_folder +
                           inference_folder + output_file,
                           header=None, index=False)
        else:
            output_file = output_file + "_hest" + ".csv"
            predictions_S, predictions_V = self.predict()
            predictions_S.to_csv(directory + output_folder +
                           inference_folder + "price_" + output_file,
                           header=None, index=False)
            predictions_V.to_csv(directory + output_folder +
                           inference_folder + "var_" + output_file,
                           header=None, index=False)
        return None
        