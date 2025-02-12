# tests/unit/utils/test_logging_config.py
import os
import logging
import shutil
from pathlib import Path
import pytest
from circman5.utils.logging_config import setup_logger
from circman5.config.project_paths import project_paths


@pytest.fixture
def test_log_dir(tmp_path):
    """Create a temporary directory for test logs."""
    return tmp_path / "test_logs"


@pytest.fixture
def cleanup_logs():
    """Cleanup fixture to remove test logs after each test."""
    yield
    log_dir = Path(project_paths.get_path("LOGS_DIR"))
    for log_file in log_dir.glob("test_logger_*.log"):
        try:
            log_file.unlink()
        except FileNotFoundError:
            pass


def test_logger_creation(cleanup_logs):
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


@pytest.mark.parametrize(
    "log_level,message",
    [
        (logging.DEBUG, "Debug message"),
        (logging.INFO, "Info message"),
        (logging.WARNING, "Warning message"),
        (logging.ERROR, "Error message"),
    ],
)
def test_logger_levels(cleanup_logs, log_level, message):
    """Test logger handles different logging levels correctly."""
    logger = setup_logger("test_logger")

    # Get logging function for the level
    log_func = getattr(logger, logging.getLevelName(log_level).lower())
    log_func(message)

    # Verify in log file
    log_dir = Path(project_paths.get_path("LOGS_DIR"))
    log_files = list(log_dir.glob("test_logger_*.log"))
    with open(log_files[-1], "r") as f:
        log_content = f.read()
        assert message in log_content


def test_custom_log_directory(tmp_path, cleanup_logs):
    """Test logger creation with custom directory."""
    custom_dir = tmp_path / "custom_logs"
    custom_dir.mkdir()

    logger = setup_logger("test_logger", str(custom_dir))
    logger.info("Test message")

    # Verify log file was created in custom directory
    log_files = list(custom_dir.glob("test_logger_*.log"))
    assert len(log_files) == 1


def test_duplicate_handler_prevention(cleanup_logs):
    """Test that duplicate handlers are not added."""
    logger = setup_logger("test_logger")
    initial_handlers = len(logger.handlers)

    # Setup logger again with same name
    logger = setup_logger("test_logger")
    assert len(logger.handlers) == initial_handlers
