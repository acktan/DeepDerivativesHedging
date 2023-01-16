"""Create model architecture."""

import logging
import os
import warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

class DeepHedging_BS(nn.Module):
    """Create the model architecture for the BS model.
    
    Args:
        conf: the config file.
    Returns:
        The BS RNN model.
    """
    def __init__(self, conf):
        super(DeepHedging_BS, self).__init__()
        input_size = conf["BS_model"]["input_size"]
        HL_size = conf["BS_model"]["HL_size"]
        output_size = conf["BS_model"]["output_size"]
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=HL_size,
                                num_layers=1)
        self.linear = torch.nn.Linear(HL_size, output_size)
        
    def forward(self, S):
        """Create the forward function for the model."""
        out, _ = self.rnn(S)
        out = self.linear(out)
        return out.squeeze()