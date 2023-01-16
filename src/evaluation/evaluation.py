"""Evaluation of model."""

import logging
import warnings
import matplotlib.pyplot as plt
import matplotlib.style as style

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")

class Evaluator():
    """Evaluate the model and save an image."""
    def __init__(self, conf, train_loss, val_loss):
        self.conf = conf
        self.train_loss = train_loss
        self.val_loss = val_loss
    
    def evaluate_model(self):
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
        path = directory + output_folder + eval_folder + evaluation_file
        val_loss = self.val_loss
        train_loss = self.train_loss

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