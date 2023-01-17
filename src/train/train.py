"""Train model."""

import logging
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.model.model import DeepHedging_BS, DeepHedging_Hest

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
    
    def loss(self, deltas, S, payoff, var, costs):
        """Calculate loss incurred.
        
        Args:
            deltas: the output of the model.
            pay_off: the payoff of the training set.
            growth: the growth of stock prices for the training set.
            costs: the costs.
        Returns:
            The loss incurred.
        """
        S_delta = torch.diff(S, dim=1)
        S_delta_T = torch.transpose(S_delta, 0, 1).squeeze()
        logger.info(f"Deltas shape: {deltas.shape}")
        logger.info(f"S_T shape: {S_delta_T.shape}")
        delta_S = torch.diagonal(torch.matmul(deltas, S_delta_T))
        first_element = torch.abs(deltas[:,0]).unsqueeze(-1)
        third_element = torch.abs(deltas[:,-1]).unsqueeze(-1)
        delta_diff = torch.cat([first_element, torch.abs(torch.diff(deltas, dim=1)), third_element], dim=1)
        costs_S = costs * torch.diagonal(torch.matmul(delta_diff, S))
        loss = payoff.squeeze() - delta_S + costs_S
        if not self.conf["model_init"]["bs_model"]:
            costs_V = costs * torch.diagonal(torch.matmul(delta_diff, var))
            V_delta = torch.diff(var, dim=1)
            V_delta_T = torch.transpose(V_delta, 0, 1).squeeze()
            logger.info(f"Deltas shape: {deltas.shape}")
            logger.info(f"V shape: {V_delta.shape}")
            logger.info(f"V_T shape: {V_delta_T.shape}")
            delta_var = torch.diagonal(torch.matmul(deltas, V_delta_T))
            loss = loss - delta_var + costs_V
        return loss
    
    
    def risk_measure(self, loss, q1, q2, beta=1):
        """Calculate custom loss function for BS Model for training.
        
        Args:
            loss: the loss incurred by predicted deltas.
            q1, q2: parameters to optimize, representing VaR.
            beta: constant, taken as 1 to consider extreme losses.
        Returns:
            The risk measure.
        """
        es_99 = (F.relu(loss-q1).mean())/(1-0.99) + q1
        es_50 = (F.relu(loss-q2).mean())/(1-0.5) + q2
        risk_measure = (es_50 + es_99*beta)/(1+beta)
        return risk_measure
    
    def evaluation(self, loss, alpha1=0.5, alpha2=0.99, beta=1):
        """Create custom evaluation function for validation.
        
        Args:
            loss: the incurred loss.
            alpha1, alpha2: 50% and 99%, referring to the alpha quantile of losses. 
            beta: a constant equal to 1 to account for extreme losses.
        Returns:
            The risk measure.
        """
        q1 = torch.quantile(loss, alpha1)
        q2 = torch.quantile(loss, alpha2)
        es_50 = loss[loss>=q1].mean()
        es_99 = loss[loss>=q2].mean()
        return (es_50 + es_99*beta)/(1+beta)
        
    def prepare_input(self, S):
        """Standardize at row level.

        Args:
            S: tensor of stock prices.
        Returns:
            S: tensor of standardized stock prices.   
        """
        S_mean = torch.mean(S, 1, True)
        S_std = torch.std(S, 1, True).unsqueeze(-1)
        S = (S[:, :-1, :] - S_mean)/S_std
        return S
    
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
        learning_rate = self.conf["BS_model"]["lr"]
        optimizer = optim.Adam(list(self.model.parameters()) + [self.q1, self.q2], lr=learning_rate)
        
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss = 0.0
            counter = 0

            for train_S, train_var, train_payoff, train_costs in self.train_loader:
                counter += 1

                # Calculate gradients and update model weights
                optimizer.zero_grad()
                S = train_S.to(device).unsqueeze(-1)
                var = train_var.to(device).unsqueeze(-1)
                payoff = train_payoff.to(device)
                costs = train_costs.to(device)
                
                train_S = self.prepare_input(S)
                if not self.conf["model_init"]["bs_model"]:
                    train_var = self.prepare_input(var)
                    train_S = torch.cat((train_S, train_var), 2)
                    
                deltas = self.model(train_S)
                losses = self.loss(deltas, S, payoff, var, costs)
                training_loss = self.risk_measure(losses, self.q1, self.q2)
                running_loss += training_loss.item()

                training_loss.backward()
                optimizer.step()

            epoch_train_loss = running_loss / counter
            train_loss.append(epoch_train_loss)
            logger.info("Train loss: {}".format(epoch_train_loss))

            running_loss_val = 0.0
            counter = 0

            for val_S, val_var, val_payoff, val_costs in self.val_loader:
                self.model.eval()

                counter += 1
                S = val_S.to(device).unsqueeze(-1)
                var = val_var.to(device).unsqueeze(-1)
                payoff = val_payoff.to(device)
                costs = val_costs.to(device)
                
                val_S = self.prepare_input(S)
                if not self.conf["model_init"]["bs_model"]:
                    val_var = self.prepare_input(var)
                    val_S = torch.cat((val_S, val_var), 2)
                    
                deltas = self.model(val_S)
                losses = self.loss(deltas, S, payoff, var, costs)
                validation_loss = self.evaluation(losses)
                running_loss_val += validation_loss.item()

            epoch_val_loss = running_loss_val / counter
            val_loss.append(epoch_val_loss)
            logger.info("Validation loss: {}".format(epoch_val_loss))
        return train_loss, val_loss
    
    def save_model(self, model_name):
        """Save model in output folder."""
        directory = self.conf["paths"]["directory"]
        output_folder = self.conf["paths"]["output_folder"]
        model_out_folder = self.conf["paths"]["model_out_folder"]
        if self.conf["model_init"]["bs_model"]:
            model_name = model_name + "_bs_"
        else:
            model_name = model_name + "_hest_"
        PATH = directory + output_folder + model_out_folder + model_name
        timestr = time.strftime("%Y%m%d-%H%M%S")
        torch.save(self.model.state_dict(), PATH + timestr)
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
        else:
            model_loaded = DeepHedging_Hest(self.conf)
            model_loaded.load_state_dict(torch.load(PATH))
        return model_loaded