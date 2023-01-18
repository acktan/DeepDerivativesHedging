"""Main script to run src."""

import json
import random
import torch
import numpy as np
from time import time
from src.utils.utils import my_get_logger
from src.loading.loading import DataLoader
from src.preprocessing.preprocessing import DataPreprocessor
from src.model.model import DeepHedging_BS, DeepHedging_Hest
from src.train.train import Train
from src.inference.inference import Inference
from src.evaluation.evaluation import Evaluator

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def main(logger, conf):
    """Main function launching step by step the pipeline.
    Args:
        logger: logger file.
        conf: config file.
    """
    START = time()
    data_loader = DataLoader(conf)
    S, var, pay_off = data_loader.get_train_data()
    time_1 = time()
    logger.debug(
        "Time to load data:" + str(time_1 - START)
    )
    data_preprocessor = DataPreprocessor(S, pay_off, var, conf)
    train_loader, val_loader = data_preprocessor.get_train_val_dataloader()
    time_2 = time()
    logger.debug(
        "Time to preprocess data:" + str(time_2 - time_1)
    )

    if conf["model_init"]["bs_model"]:
        model = DeepHedging_BS(conf)
        logger.debug("Creating BS RNN model architecture.")
    else:
        model = DeepHedging_Hest(conf)
        logger.debug("Creating Heston RNN model architecture.")
    train_class = Train(conf, model, train_loader, val_loader)
    if conf["model_init"]["train"]:
        training_loss, val_loss = train_class.train()
        train_class.save_model("model")
        eval_class = Evaluator(conf)
        eval_class.evaluate_model(training_loss, val_loss)
        risk_measure = eval_class.evaluate_train_dataset(model, S, var, pay_off, train_class)
        time_3 = time()
        logger.debug(
            "Time to create, train and evaluate the model:" + str(time_3 - time_2)
        )
    else:
        model = train_class.load_saved_model("model_test_2_bs_20230117-175453")
        time_3 = time()
        logger.debug(
            "Time to load a saved model:" + str(time_3 - time_2)
        )

    inference_class = Inference(conf, model)
    inference_class.save_predictions()
    time_4 = time()
    logger.debug(
        "Time to make predictions:" + str(time_4 - time_3)
    )

if __name__ == "__main__":
    path_conf = "params/config.json"
    conf = json.load(open(path_conf, "r"))
    path_log = conf["path_log"]  # "../log/my_log_file.txt"
    log_level = conf["log_level"]  # "DEBUG"
    # instanciation of the logger
    logger = my_get_logger(path_log, log_level, my_name="main_logger")
    try:
        main(logger=logger, conf=conf)

    except Exception:
        logger.error("Error during execution", exc_info=True)