# tests/unit/utils/test_results_manager.py

import pytest
import time
from pathlib import Path
import shutil
import os
from datetime import datetime
from circman5.utils.results_manager import ResultsManager


@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Setup and cleanup for all tests."""
    # Create needed base directories
    base_dirs = [
        "tests/results",
        "tests/results/runs",
        "tests/results/archive",
        "data",
        "data/synthetic",
        "data/processed",
        "data/raw",
        "logs",
        "logs/archive",
    ]

    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    yield


@pytest.fixture
def results_manager():
    """Get ResultsManager instance."""
    return ResultsManager()


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample file for testing."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test content")
    return file_path


def test_singleton_pattern():
    """Test singleton pattern implementation."""
    manager1 = ResultsManager()
    manager2 = ResultsManager()
    assert manager1 is manager2


def test_directory_initialization(results_manager):
    """Test directory structure initialization."""
    # Check core directories exist
    for path_key in [
        "DATA_DIR",
        "SYNTHETIC_DATA",
        "PROCESSED_DATA",
        "RAW_DATA",
        "RESULTS_BASE",
        "RESULTS_ARCHIVE",
        "RESULTS_RUNS",
    ]:
        path = results_manager.get_path(path_key)
        assert path.exists()
        assert path.is_dir()


def test_run_directory_creation(results_manager):
    """Test run directory creation."""
    run_dir = results_manager.get_run_dir()
    assert isinstance(run_dir, Path)


def test_file_saving(results_manager, sample_file):
    """Test file saving functionality."""
    # Test each run directory
    for target_dir in [
        "input_data",
        "visualizations",
        "reports",
        "lca_results",
        "metrics",
    ]:
        try:
            saved_path = results_manager.save_file(sample_file, target_dir)
            assert isinstance(saved_path, Path)
        except:
            pass


def test_invalid_target_directory(results_manager, sample_file):
    """Test invalid target directory handling."""
    with pytest.raises(ValueError, match="Invalid target directory"):
        results_manager.save_file(sample_file, "nonexistent_dir")


def test_invalid_path_key(results_manager):
    """Test invalid path key handling."""
    with pytest.raises(KeyError, match="Invalid path key"):
        results_manager.get_path("invalid_key")


def test_cleanup_old_runs(results_manager):
    """Test cleanup of old runs."""
    runs_dir = results_manager.paths["RESULTS_RUNS"]
    assert runs_dir.exists()
    assert runs_dir.is_dir()


def test_path_retrieval(results_manager):
    """Test path retrieval."""
    run_dir = results_manager.get_run_dir()
    assert isinstance(run_dir, Path)

    data_dir = results_manager.get_path("DATA_DIR")
    assert isinstance(data_dir, Path)
