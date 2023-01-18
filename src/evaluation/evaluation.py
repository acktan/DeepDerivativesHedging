"""Evaluation of model."""

import logging
import warnings
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.style as style

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

class Evaluator():
    """Evaluate the model and save an image."""
    def __init__(self, conf):
        self.conf = conf
    
    def evaluate_model(self, train_loss, val_loss):
        """Evaluate the training model and save an image.
        Args:
            history: history of the fitted model.
        Return:
            Saved training and validation loss curves.
        """
        logger.info(f"Evaluating model...")
        directory = self.conf["paths"]["directory"]
        output_folder = self.conf["paths"]["output_folder"]
        eval_folder = self.conf["paths"]["eval_folder"]
        evaluation_file = self.conf["paths"]["evaluation_file"]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.conf["model_init"]["bs_model"]:
            evaluation_file = evaluation_file + "_bs_" + timestr + ".jpg"
        else:
            evaluation_file = evaluation_file + "_hest_" + timestr + ".jpg"
        path = directory + output_folder + eval_folder + evaluation_file

        epochs = len(train_loss)

        style.use("bmh")
        plt.figure(figsize=(8, 8))

        plt.subplot(2, 1, 1)
        plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(path)
        plt.show()
        plt.clf()
        return None
    
    def evaluate_train_dataset(self, model, S, v, payoff, train_class):
        """Evaluate the model on the full training data.
        Args:
            model:
            S, v, payoff: the input stock prices, the variance swap (only for Heston) and the payoff.
        Return:
            Print the emprirical risk measure on the full training set.
        """
        S = torch.Tensor(S).unsqueeze(-1)
        payoff = torch.Tensor(payoff)
        if self.conf["model_init"]["bs_model"]:
            costs = torch.Tensor([0.0] * train_S.shape[0])
            var = torch.Tensor([0.0] * train_S.shape[0])
            deltas = model(S)
            loss = train_class.loss(deltas, S, payoff, var, costs)
            train_class.evaluation(loss)
        
        