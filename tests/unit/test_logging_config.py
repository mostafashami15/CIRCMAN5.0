# tests/unit/test_logging_config.py
import os
import logging
from pathlib import Path
from circman5.logging_config import setup_logger
from circman5.config.project_paths import project_paths


def test_logger_creation():
    """Test that logger is created with correct configuration."""

    # Setup logger
    logger = setup_logger("test_logger")

    # Verify logger level
    assert logger.level == logging.DEBUG

    # Verify handlers
    assert len(logger.handlers) == 2

    # Verify file handler
    file_handler = logger.handlers[0]
    assert isinstance(file_handler, logging.FileHandler)
    assert file_handler.level == logging.DEBUG

    # Verify console handler
    console_handler = logger.handlers[1]
    assert isinstance(console_handler, logging.StreamHandler)
    assert console_handler.level == logging.INFO

    # Verify log file creation
    log_dir = Path(project_paths.get_path("LOGS_DIR"))
    log_files = list(log_dir.glob("test_logger_*.log"))
    assert len(log_files) > 0

    # Test logging
    test_message = "Test log message"
    logger.info(test_message)

    # Verify message was written to file
    with open(log_files[-1], "r") as f:
        log_content = f.read()
        assert test_message in log_content
