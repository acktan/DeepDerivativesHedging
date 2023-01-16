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
    def __init__(self, df_train, pay_off, df_growth, conf):
        self.df_train = df_train
        self.pay_off = pay_off
        self.df_growth = df_growth
        self.conf = conf
    
    def standardize(self, S):
        """Standardize at row level.

        Args:
            S: dataframe of stock prices.
        Returns:
            S_stand: dataframe of standardized stock prices.   
        """
        mean = S.mean(axis=1)
        std = S.std(axis=1)
        keys = np.arange(0, 30, 1)
        S_std = pd.concat([std]*len(keys), keys=keys, axis=1)
        S_mean = pd.concat([mean]*len(keys), keys=keys, axis=1)
        S_stand = (S - S_mean)/S_std
        return S_stand

    def train_val_split(self, split_percent=0.8):
        """Split the training set into validation and training sets.

        Args:
            split_percent: percentage of data that is for training.
        Returns:
            train_data, val_data: training and validation array of stock prices.
            train_payoff, val_payoff: training and validation array of payoffs.
            train_growth, val_growth: training and validation array of stock prices absolute growth.
        """
        df_train = self.standardize(self.df_train.iloc[:,:30])
        data = np.array(df_train.values.astype('float32'))
        pay_off = np.array(self.pay_off)
        n = len(data)
        # Point for splitting data into train and test
        split = int(n*split_percent)
        train_data = data[range(split)]
        val_data = data[split:]
        n = len(df_train)
        split = int(n*split_percent)
        train_payoff = pay_off[range(split)]
        train_growth = np.array(self.df_growth.iloc[0:split, 1:])
        val_payoff = pay_off[split:]
        val_growth = np.array(self.df_growth.iloc[split:, 1:])
        return train_data, val_data, train_payoff, val_payoff, train_growth, val_growth
    
    def transform_to_tensor(self):
        """Transform training and validation arrays to tensors.

        Args:
            None.
        Returns:
            train_X, val_X: training and validation tensor of stock prices.
            train_payoff, val_payoff: training and validation tensor of payoffs.
            train_growth, val_growth: training and validation tensor of stock prices absolute growth.
            train_costs, val_costs: training and validation tensor of costs.
        """
        train_data, val_data, train_payoff, val_payoff, train_growth, val_growth = self.train_val_split()
        train_payoff = torch.Tensor(train_payoff)
        val_payoff = torch.Tensor(val_payoff)
        train_growth = torch.Tensor(train_growth)
        val_growth = torch.Tensor(val_growth)
        train_X = torch.Tensor(train_data)
        val_X = torch.Tensor(val_data)
        if self.conf["model_init"]["bs_model"]:
            cost = self.conf["model_init"]["cost_bs"]
            train_costs = torch.Tensor([cost] * train_X.shape[0])
            val_costs = torch.Tensor([cost] * val_X.shape[0])
        else:
            cost = self.conf["model_init"]["cost_hest"]
            train_costs = torch.Tensor([cost] * train_X.shape[0])
            val_costs = torch.Tensor([cost] * val_X.shape[0])
        return train_X, val_X, train_growth, val_growth, train_payoff, val_payoff, train_costs, val_costs
    
    def get_train_val_dataloader(self):
        """Create training and validation dataloaders.

        Args:
            None.
        Returns:
            train_loader, val_loader: torch dataloaders for training and validation sets.
        """   
        train_X, val_X, train_growth, val_growth, train_payoff, val_payoff, train_costs, val_costs = self.transform_to_tensor()
        batch_size = self.conf["model_init"]["batch_size"]
        dataset_train = torch.utils.data.TensorDataset(train_X, train_payoff, train_growth, train_costs)
        dataset_val = torch.utils.data.TensorDataset(val_X, val_payoff, val_growth, val_costs)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
        