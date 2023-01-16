"""Train model."""

import logging
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.model.model import DeepHedging_BS

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train():
    """Train RNN model.
    
    Args:
        conf: the config file.
        model: the model.
        train_loader: the training dataloader.
        val_loader: the validation dataloader.
    Returns:
        The trianing and validation losses.
    """
    def __init__(self, conf, model, train_loader, val_loader):
        self.conf = conf
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.q1 = torch.tensor((0.), dtype=torch.float32, requires_grad=True, device=device)
        self.q2 = torch.tensor((0.), dtype=torch.float32, requires_grad=True, device=device)
        learning_rate = self.conf["BS_model"]["lr"]
        self.optimizer = optim.Adam(list(model.parameters()) + [self.q1, self.q2], lr=learning_rate)
    
    def loss(self, deltas, pay_off, growth, costs, q1, q2, beta=1):
        """Create custom loss function for training.
        
        Args:
            deltas: the output of the model.
            pay_off: the payoff of the training set.
            growth: the growth of stock prices for the training set.
            costs: the costs.
            beta: a constant equal to 1 to account for extreme losses.
        Returns:
            The risk measure.
        """
            
        S_delta_T = torch.transpose(growth, 0, 1)
        delta_S = torch.diagonal(torch.matmul(deltas, S_delta_T))
        loss = torch.sub(pay_off.squeeze(), delta_S)
        es_99 = (F.relu(loss-q1).mean())/(1-0.99) + q1
        es_50 = (F.relu(loss-q2).mean())/(1-0.5) + q2
        risk_measure = (es_50 + es_99*beta)/(1+beta)
        return risk_measure
    
    def evaluation(self, deltas, val_payoff, val_growth, costs):
        """Create custom evaluation function for training.
        
        Args:
            deltas: the output of the model.
            pay_off: the payoff of the val set.
            growth: the growth of stock prices for the val set.
            costs: the costs.
            beta: a constant equal to 1 to account for extreme losses.
        Returns:
            The risk measure.
        """
        S_delta_T = torch.transpose(growth, 0, 1)
        delta_S = torch.diagonal(torch.matmul(deltas, S_delta_T))
        loss = torch.sub(pay_off.squeeze(), delta_S)
        shape = loss.shape[0]
        es_99 = torch.mean(- torch.topk(-loss, math.ceil(0.01*shape))[0])
        es_50 = torch.mean(- torch.topk(-loss, math.ceil(0.5*shape))[0])
        risk_measure = (es_50 + es_99*beta)/(1+beta)
        return risk_measure
        
    def train(self):
        """Train model and get training/val losses.
        
        Args:
            None.
        Returns:
            val_loss, train_loss: array of val and train losses.
        """
        train_loss = []
        val_loss = []
        num_epochs = self.conf["BS_model"]["num_epochs"]
        
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss = 0.0
            counter = 0

            for train_X, train_payoff, train_growth, costs in self.train_loader:
                counter += 1

                # Calculate gradients and update model weights
                self.optimizer.zero_grad()
                train_X = train_X.unsqueeze(-1)
                deltas = self.model(train_X)
                losses = self.loss(deltas, train_payoff, train_growth, costs, self.q1, self.q2)
                running_loss += losses.item()

                losses.backward()
                self.optimizer.step()

            epoch_train_loss = running_loss / counter
            train_loss.append(epoch_train_loss)
            logger.info("Train loss: {}".format(epoch_train_loss))

            running_loss_test = 0.0
            counter = 0

            for val_X, val_payoff, val_growth, costs in self.val_loader:
                self.model.eval()

                counter += 1
                val_X = val_X.unsqueeze(-1)
                deltas = self.model(val_X)
                losses_test = self.loss(deltas, val_payoff, val_growth, costs, self.q1, self.q2)
                running_loss_test += losses_test.item()

            epoch_test_loss = running_loss_test / counter
            val_loss.append(epoch_test_loss)
            logger.info("Test loss: {}".format(epoch_test_loss))
        return train_loss, val_loss
    
    def save_model(self, model_name):
        """Save model in output folder."""
        directory = self.conf["paths"]["directory"]
        output_folder = self.conf["paths"]["output_folder"]
        model_out_folder = self.conf["paths"]["model_out_folder"]
        PATH = directory + output_folder + model_out_folder + model_name
        torch.save(self.model.state_dict(), PATH)
        return None
    
    def load_saved_model(self, model_name):
        """Load saved model."""
        directory = self.conf["paths"]["directory"]
        output_folder = self.conf["paths"]["output_folder"]
        model_out_folder = self.conf["paths"]["model_out_folder"]
        PATH = directory + output_folder + model_out_folder + model_name
        if self.conf["model_init"]["bs_model"]:
            model_loaded = DeepHedging_BS(self.conf)
            model_loaded.load_state_dict(torch.load(PATH))
        return model_loaded