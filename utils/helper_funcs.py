import logging
from datetime import timezone
from logging.handlers import RotatingFileHandler

# Configure logging
def setup_logging(log_path = './data/logs/pipeline.log'):
    logger = logging.getLogger('stock_pipeline')
    logger.setLevel(logging.INFO)

    # Create a file handler for logging to a file
    file_handler = RotatingFileHandler(log_path, maxBytes=2000000, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # Create a console handler for logging to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger