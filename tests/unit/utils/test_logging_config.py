# tests/unit/utils/test_logging_config.py

import logging
import pytest
from pathlib import Path
import os
from circman5.utils.logging_config import setup_logger
from circman5.utils.results_manager import ResultsManager


@pytest.fixture
def results_manager():
    """Create ResultsManager instance."""
    manager = ResultsManager()
    yield manager
    # Cleanup log files
    log_dir = manager.get_path("LOGS_DIR")
    for file in log_dir.glob("test_logger_*.log"):
        try:
            file.unlink()
        except OSError:
            pass


def test_logger_creation(results_manager):
    """Test logger creation with ResultsManager paths."""
    log_dir = results_manager.get_path("LOGS_DIR")
    logger = setup_logger("test_logger", str(log_dir))

    test_message = "Test log message"
    logger.info(test_message)

    # Verify log file creation
    log_files = list(log_dir.glob("test_logger_*.log"))
    assert len(log_files) > 0, "Log file not created"

    # Verify message in log
    with open(log_files[0], "r") as f:
        log_content = f.read()
        assert test_message in log_content, "Test message not found in log"


@pytest.mark.parametrize(
    "log_level,message",
    [
        (logging.DEBUG, "Debug message"),
        (logging.INFO, "Info message"),
        (logging.WARNING, "Warning message"),
        (logging.ERROR, "Error message"),
    ],
)
def test_logger_levels(results_manager, log_level, message):
    """Test logger handles different logging levels correctly."""
    log_dir = results_manager.get_path("LOGS_DIR")
    logger = setup_logger("test_logger", str(log_dir))

    log_func = getattr(logger, logging.getLevelName(log_level).lower())
    log_func(message)

    log_files = list(log_dir.glob("test_logger_*.log"))
    assert len(log_files) > 0, "Log file not created"

    with open(log_files[0], "r") as f:
        log_content = f.read()
        assert (
            message in log_content
        ), f"Message not found for level {logging.getLevelName(log_level)}"
