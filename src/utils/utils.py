"""Store utils functions"""

import logging

logger = logging.getLogger("main_logger")


def my_get_logger(path_log, log_level, my_name=""):
    """Instanciation of logger and parametrization.
    Args:
        param path_log: path of log file
        param log_level: level of log
    Returns:
       Log folder.
    """
    log_level_dict = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    LOG_LEVEL = log_level_dict[log_level]

    if my_name != "":
        logger = logging.getLogger(my_name)
        logger.setLevel(LOG_LEVEL)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(LOG_LEVEL)

    # create a file handler
    handler = logging.FileHandler(path_log)
    handler.setLevel(LOG_LEVEL)

    # create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(funcName)s - %(levelname)-8s: %(message)s"
    )
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
