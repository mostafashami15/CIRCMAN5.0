"""Logging configuration module for the CIRCMAN5 system."""

import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path
from ..utils.results_manager import results_manager


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging system with file and console handlers.

    Args:
        name: Name of the logger
        log_dir: Optional custom log directory path
        file_level: Logging level for file output
        console_level: Logging level for console output

    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        # Get log directory from results_manager if not specified
        if log_dir is None:
            log_dir_path = results_manager.get_path("LOGS_DIR")
        else:
            log_dir_path = Path(log_dir)

        # Remove existing handlers from logger if it exists
        logger = logging.getLogger(name)
        logger.handlers = []  # Clear any existing handlers
        logger.setLevel(logging.DEBUG)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"{name}_{timestamp}.log"

        # File handler setup
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # Log initial message to verify file creation
        logger.debug(f"Logger initialized. Log file: {log_file}")
        logger.handlers[0].flush()  # Force flush the file handler

        return logger

    except Exception as e:
        # Fallback to basic console logging
        fallback_logger = logging.getLogger(name)
        fallback_logger.handlers = []
        fallback_logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        fallback_logger.addHandler(ch)
        fallback_logger.error(f"Failed to setup logger: {str(e)}")
        return fallback_logger
