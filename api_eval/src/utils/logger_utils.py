import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue


def setup_logger(name: str = "ApplicationLogger", log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a thread-safe logger using a QueueHandler and QueueListener.
    
    Args:
        name (str): The name of the logger.
        log_level (int): The logging level, e.g., logging.INFO or logging.DEBUG.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

