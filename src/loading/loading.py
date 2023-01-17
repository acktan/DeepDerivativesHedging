"""Load data for training."""

import logging
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")


class DataLoader():
    """Load the train data.

    Args:
        conf: the config file
    Returns:
        df_train: train data containing stock prices.
        df_growth: growth of stock prices absolute for train data.
        pay_off: pay_off for train data.
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.directory = conf["paths"]["directory"]
        self.input_folder = conf["paths"]["input_folder"]

    def col_numeric_names(self, df):
        """Return df with column names from 0 to 30.

        Args:
            df: dataframe.
        Returns:
            df with column names from 0 to 30.
        """
        df = df.rename(columns={x:y for x,y in zip(df.columns,range(0,len(df.columns)))})
        return df

    def absolute_growth(self, df):
        """Get the absolute growth in stock price for each path i in dataframe.
        
        Args:
            df: dataframe containing stock price.
        Returns:
            Dataframe containing stock price absolute growth.
        """
        len_df = df.shape[1]
        temp = df.iloc[:, 1:len_df]
        temp = self.col_numeric_names(temp)
        temp2 = df.iloc[:, :len_df-1]
        df2 = temp-temp2
        df2.insert(0, "0", 0)
        df2 = self.col_numeric_names(df2)
        return df2
    
    def percentage_growth(self, df):
        """Get the percentage growth in stock price for each path i in dataframe.
        
        Args:
            df: dataframe containing stock price.
        Returns:
            Dataframe containing stock price percentage growth.
        """
        df2 = self.absolute_growth(df)
        df2 = df2.drop(columns=[0])
        df2 = self.col_numeric_names(df2)
        df3 = df2.div(df.iloc[:, :df.shape[1]-1])
        df3.insert(0, "0", 0)
        df3 = self.col_numeric_names(df3)
        return df3

    def return_filenames(self):
        """Return the files in the input folder.
        
        Args:
            conf: the config file.
        Returns:
            Name of files within the input directory.
        """
        return os.listdir(self.directory + self.input_folder)
    
    def get_train_data(self):
        """Return the training data for Black & Scholes or Heston model.
        
        Args:
            conf: the config file.
        Returns:
            train1: training data containing stock prices.
            train2: training data containing variance swap for Heston,
            None for for BS.
            pay_off: training data containing payoffs.
        """
        path = self.directory + self.input_folder
        if self.conf["model_init"]["bs_model"]:
            trainPath = path + self.conf["paths"]["train_data_bs"]
            targetPath = path + self.conf["paths"]["pay_off_bs"]
            S = pd.read_csv(trainPath,
                            header=None)
            var = None
            pay_off = pd.read_csv(targetPath,
                                  header=None)
        else:
            trainPath1 = path + self.conf["paths"]["train_data_hest_1"]
            trainPath2 = path + self.conf["paths"]["train_data_hest_2"]
            targetPath = path + self.conf["paths"]["pay_off_hest"]
            S = pd.read_csv(trainPath1, header=None)
            var = pd.read_csv(trainPath2, header=None)
            pay_off = pd.read_csv(targetPath, header=None)
            
        return S, var, pay_off
        