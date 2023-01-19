"""Create model architecture."""

import logging
import os
import warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepHedging_BS(nn.Module):
    """Create the model architecture for the BS model.
    
    Args:
        conf: the config file.
    Returns:
        The BS RNN model.
    """
    def __init__(self, conf):
        super(DeepHedging_BS, self).__init__()
        self.input_size = conf["BS_model"]["input_size"]
        self.HL_size = conf["BS_model"]["HL_size"]
        self.output_size = conf["BS_model"]["output_size"]
        self.num_layers = conf["BS_model"]["num_layers"]
        self.rnn = torch.nn.GRU(input_size=self.input_size,
                                hidden_size=self.HL_size,
                                num_layers=self.num_layers,
                                batch_first=True).to(device)
        self.linear = torch.nn.Linear(self.HL_size, self.output_size).to(device)
        
    def forward(self, S):
        """Create the forward function for the model."""
        #h0 = torch.zeros(self.num_layers, S.size(0), self.HL_size).requires_grad_()
        h0 = torch.empty(self.num_layers, S.size(0), self.HL_size).to(device)
        h0 = torch.nn.init.xavier_uniform_(h0).requires_grad_()
        out, _ = self.rnn(S, h0)
        out = self.linear(out)
        return out.squeeze()


class DeepHedging_Hest(nn.Module):
    """Create the model architecture for the Heston model.
    
    Args:
        conf: the config file.
    Returns:
        The Heston RNN model.
    """
    def __init__(self, conf):
        super(DeepHedging_Hest, self).__init__()
        self.input_size = conf["Hest_model"]["input_size"]
        self.HL_size = conf["Hest_model"]["HL_size"]
        self.output_size = conf["Hest_model"]["output_size"]
        self.num_layers = conf["Hest_model"]["num_layers"]
        #self.rnn = torch.nn.GRU(input_size=self.input_size,
        #                        hidden_size=self.HL_size,
        #                        num_layers=self.num_layers,
        #                        batch_first=True).to(device)
        self.lstm = torch.nn.GRU(input_size=self.input_size,
                                hidden_size=self.HL_size,
                                num_layers=self.num_layers,
                                batch_first=True).to(device)
        #self.dropout = nn.Dropout(0.2).to(device)
        self.linear = torch.nn.Linear(self.HL_size, self.output_size).to(device)
    def forward(self, S, hidden):
        """Create the forward function for the model."""
        #h0 = torch.empty(self.num_layers, S.size(0), self.HL_size).to(device)
        #h0 = torch.nn.init.xavier_uniform_(h0).requires_grad_()
        #c0 = torch.empty(self.num_layers, S.size(0), self.HL_size).to(device)
        #c0 = torch.nn.init.xavier_uniform_(c0).requires_grad_()
        
        #out, _ = self.rnn(S, h0)
        #logger.info("Hidden size before: {}".format(hidden.size()))
        out, hidden = self.lstm(S, hidden)
        #logger.info("Hidden size after: {}".format(hidden.size()))
        #out = self.dropout(out)
        out = self.linear(out)
        return out.squeeze(), hidden