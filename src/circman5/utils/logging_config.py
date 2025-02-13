# src/circman5/utils/logging_config.py

import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path
from circman5.config.project_paths import project_paths


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure logging system with file and console handlers."""
    try:
        # Get log directory
        if log_dir is None:
            log_dir = project_paths.get_path("LOGS_DIR")

        # Ensure log directory exists
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

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
