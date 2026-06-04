# File: utils/logger.py
import logging
import sys

def setup_logger(name="synapse_logger", log_file="training.log", level=logging.INFO):
    """
    Sets up and returns a logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger
