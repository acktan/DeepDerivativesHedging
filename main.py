"""Main script to run src."""

import json
import random
from time import time
from src.utils.utils import my_get_logger
from src.loading.loading import DataLoader
from src.preprocessing.preprocessing import DataPreprocessor

random.seed(42)

def main(logger, conf):
    """Main function launching step by step the pipeline.
    Args:
        logger: logger file.
        conf: config file.
    """
    START = time()
    data_loader = DataLoader(conf)
    df_train, df_growth, pay_off = data_loader.get_train_data_bs()
    time_1 = time()
    logger.debug(
        "Time to load data:" + str(time_1 - START)
    )
    data_preprocessor = DataPreprocessor(df_train, pay_off, df_growth, conf)
    train_loader, val_loader = data_preprocessor.get_train_val_dataloader()
    time_2 = time()
    logger.debug(
        "Time to preprocess data:" + str(time_2 - time_1)
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