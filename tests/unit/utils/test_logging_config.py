# tests/unit/utils/test_logging_config.py

import os
import logging
import shutil
import time
from pathlib import Path
import pytest
from circman5.utils.logging_config import setup_logger
from circman5.config.project_paths import project_paths


@pytest.fixture(scope="function")
def test_log_dir(tmp_path):
    """Create a temporary directory for test logs."""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir(parents=True)
    yield log_dir
    # Cleanup
    if log_dir.exists():
        shutil.rmtree(log_dir)


def verify_log_file_creation(
    directory: Path, pattern: str = "test_logger_*.log", timeout: float = 1.0
) -> list:
    """Helper function to verify log file creation with timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        log_files = list(directory.glob(pattern))
        if log_files:
            return log_files
        time.sleep(0.1)
    return []


def test_logger_creation(test_log_dir):
    """Test that logger is created with correct configuration."""
    logger = setup_logger("test_logger", str(test_log_dir))

    # Test log writing
    test_message = "Test log message"
    logger.info(test_message)

    # Verify log file creation
    log_files = verify_log_file_creation(test_log_dir)
    assert len(log_files) > 0, "Log file was not created"

    # Verify message in log file
    with open(log_files[0], "r") as f:
        log_content = f.read()
        assert test_message in log_content, "Test message not found in log file"


@pytest.mark.parametrize(
    "log_level,message",
    [
        (logging.DEBUG, "Debug message"),
        (logging.INFO, "Info message"),
        (logging.WARNING, "Warning message"),
        (logging.ERROR, "Error message"),
    ],
)
def test_logger_levels(test_log_dir, log_level, message):
    """Test logger handles different logging levels correctly."""
    logger = setup_logger("test_logger", str(test_log_dir))

    # Get logging function for the level
    log_func = getattr(logger, logging.getLevelName(log_level).lower())
    log_func(message)

    # Verify log file creation and content
    log_files = verify_log_file_creation(test_log_dir)
    assert (
        len(log_files) > 0
    ), f"Log file not created for {logging.getLevelName(log_level)}"

    with open(log_files[0], "r") as f:
        log_content = f.read()
        assert (
            message in log_content
        ), f"Message not found for level {logging.getLevelName(log_level)}"


def test_custom_log_directory(test_log_dir):
    """Test logger creation with custom directory."""
    logger = setup_logger("test_logger", str(test_log_dir))
    logger.info("Test message")

    # Verify log file creation
    log_files = verify_log_file_creation(test_log_dir)
    assert len(log_files) == 1, "Expected exactly one log file"

    with open(log_files[0], "r") as f:
        log_content = f.read()
        assert "Test message" in log_content


def test_duplicate_handler_prevention(test_log_dir):
    """Test that duplicate handlers are not added."""
    logger = setup_logger("test_logger", str(test_log_dir))
    initial_handlers = len(logger.handlers)

    # Setup logger again with same name
    logger = setup_logger("test_logger", str(test_log_dir))
    assert len(logger.handlers) == initial_handlers
