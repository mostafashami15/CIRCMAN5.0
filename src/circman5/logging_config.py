"""Logging configuration for the CIRCMAN5.0 system."""

import logging
import os
from datetime import datetime
from circman5.config.project_paths import project_paths


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Configure logging system with file and console handlers.

    Args:
        name: Logger name
        log_dir: Optional custom log directory. If None, uses default from project_paths

    Returns:
        logging.Logger: Configured logger instance
    """
    # Use project paths if no specific log_dir provided
    if log_dir is None:
        log_dir = project_paths.get_path("LOGS_DIR")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not logger.handlers:
        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        # File handler - detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Console handler - info level and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger
