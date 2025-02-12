"""Logging configuration for the CIRCMAN5.0 system."""

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
    """
    Configure logging system with file and console handlers.

    Args:
        name: Logger name
        log_dir: Optional custom log directory. If None, uses default from project_paths
        file_level: Logging level for file handler (default: DEBUG)
        console_level: Logging level for console handler (default: INFO)

    Returns:
        logging.Logger: Configured logger instance

    Raises:
        OSError: If there are issues creating log directory or files
        ValueError: If invalid logging configuration is provided
    """
    try:
        # Use project paths if no specific log_dir provided
        if log_dir is None:
            log_dir = project_paths.get_path("LOGS_DIR")

        # Ensure log directory exists
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Base level for logger

        # Prevent duplicate handlers
        if not logger.handlers:
            # Create timestamped log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir_path / f"{name}_{timestamp}.log"

            try:
                # File handler - detailed logging
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(file_level)
                file_format = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_format)
                logger.addHandler(file_handler)

                # Console handler - info level and above
                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_level)
                console_format = logging.Formatter("%(levelname)s: %(message)s")
                console_handler.setFormatter(console_format)
                logger.addHandler(console_handler)

                logger.debug(f"Logger initialized. Log file: {log_file}")

            except Exception as e:
                # If file handler setup fails, log the error and try to continue with console only
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_format = logging.Formatter("%(levelname)s: %(message)s")
                console_handler.setFormatter(console_format)
                logger.addHandler(console_handler)
                logger.warning(
                    f"Failed to setup file handler: {str(e)}. Continuing with console logging only."
                )

        # Verify logger configuration
        if not logger.handlers:
            raise ValueError("No handlers were created for the logger")

        return logger

    except Exception as e:
        # Fallback to basic console logging if setup fails
        fallback_logger = logging.getLogger(name)
        if not fallback_logger.handlers:
            fallback_logger.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_format)
            fallback_logger.addHandler(console_handler)
        fallback_logger.error(f"Failed to setup logging configuration: {str(e)}")
        return fallback_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def cleanup_old_logs(max_age_days: int = 30) -> None:
    """
    Clean up log files older than specified days.

    Args:
        max_age_days: Maximum age of log files in days

    Raises:
        OSError: If there are issues accessing or removing log files
    """
    try:
        log_dir = Path(project_paths.get_path("LOGS_DIR"))
        if not log_dir.exists():
            return

        current_time = datetime.now()
        logger = get_logger("log_cleanup")

        for log_file in log_dir.glob("*.log"):
            try:
                file_age = current_time - datetime.fromtimestamp(
                    log_file.stat().st_mtime
                )
                if file_age.days > max_age_days:
                    log_file.unlink()
                    logger.info(f"Removed old log file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to process log file {log_file}: {str(e)}")

    except Exception as e:
        logger = get_logger("log_cleanup")
        logger.error(f"Failed to cleanup old logs: {str(e)}")
