import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="moodlebackend"):
    """
    Configures a logger that writes to both the console and a text file.
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs in reloads
    if not logger.handlers:
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Rotating to prevent massive files)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global instance
logger = setup_logger()
