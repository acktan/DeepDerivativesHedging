"""Preprocess data for training."""

import logging
import warnings
import torch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

class DataPreprocessor():
    """Preprocess and split training data and return dataloaders.

    Args:
        conf: the config file
        df_train: dataframe with stock price for train set.
        pay_off: dataframe with payoff for train set.
        df_growth: dataframe with absolute growth of stock price for train set.
    Returns:
        train_loader, val_loader: torch dataloaders for training and validation sets.   
    """
    def __init__(self, S, pay_off, var, conf):
        self.S = S
        self.var = var
        self.pay_off = pay_off
        self.conf = conf

    def train_val_split(self, split_percent=0.8):
        """Split the training set into validation and training tensors.

        Args:
            split_percent: percentage of data that is for training.
        Returns:
            train_X, val_X: training and validation tensor of stock prices.
            train_payoff, val_payoff: training and validation tensor of payoffs.
            train_costs, val_costs: training and validation tensor of costs.
            If Heston:
                val_var, train_var: training and validation variance swap.
        """
        S = np.array(self.S.values.astype('float32'))
        pay_off = np.array(self.pay_off.values.astype('float32'))
        n = len(S)
        # Point for splitting data into train and val
        split = int(n*split_percent)
        train_S = torch.Tensor(S[range(split)])
        val_S = torch.Tensor(S[split:])
        train_payoff = torch.Tensor(pay_off[range(split)])
        val_payoff = torch.Tensor(pay_off[split:])

        if self.conf["model_init"]["bs_model"]:
            cost = self.conf["model_init"]["cost_bs"]
            train_costs = torch.Tensor([cost] * train_S.shape[0])
            val_costs = torch.Tensor([cost] * val_S.shape[0])
            train_var = torch.Tensor([0.0] * train_S.shape[0])
            val_var = torch.Tensor([0.0] * val_S.shape[0])
        else:
            cost = self.conf["model_init"]["cost_hest"]
            train_costs = torch.Tensor([cost] * train_S.shape[0])
            val_costs = torch.Tensor([cost] * val_S.shape[0])
            var = np.array(self.var.values.astype('float32'))
            train_var = torch.Tensor(var[range(split)])
            val_var = torch.Tensor(var[split:])
        
        return (train_S, val_S, train_payoff, val_payoff,
                train_costs, val_costs, train_var, val_var)
    
    def get_train_val_dataloader(self):
        """Create training and validation dataloaders.

        Args:
            None.
        Returns:
            train_loader, val_loader: torch dataloaders for training and validation sets.
        """   
        (train_S, val_S, train_payoff, val_payoff,
         train_costs, val_costs, train_var, val_var) = self.train_val_split()
        batch_size = self.conf["model_init"]["batch_size"]
        dataset_train = torch.utils.data.TensorDataset(train_S, train_var,
                                                       train_payoff, train_costs)
        dataset_val = torch.utils.data.TensorDataset(val_S, val_var,
                                                     val_payoff, val_costs)
        
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
        