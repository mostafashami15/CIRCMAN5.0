"""Unit tests for project paths configuration."""
import os
import pytest
from pathlib import Path
from circman5.config.project_paths import project_paths


def test_project_paths_initialization():
    """Test that all project paths are initialized correctly."""

    # Check that all expected paths exist
    expected_paths = [
        "DATA_DIR",
        "SYNTHETIC_DATA",
        "PROCESSED_DATA",
        "RAW_DATA",
        "RESULTS_DIR",
        "RESULTS_ARCHIVE",
        "RESULTS_LATEST",
        "RESULTS_RUNS",
        "LOGS_DIR",
        "LOGS_ARCHIVE",
    ]

    for path_key in expected_paths:
        assert path_key in project_paths.paths
        path = project_paths.paths[path_key]
        assert isinstance(path, Path)
        assert path.exists(), f"Path {path_key} does not exist: {path}"


def test_run_directory_creation():
    """Test creation of new run directory."""

    # Create new run directory
    run_dir = project_paths.get_run_directory()

    # Check that directory exists
    assert run_dir.exists()

    # Check required subdirectories
    required_subdirs = ["input_data", "visualizations", "reports"]
    for subdir in required_subdirs:
        assert (run_dir / subdir).exists()

    # Check latest symlink
    latest_link = Path(project_paths.get_path("RESULTS_LATEST"))
    assert latest_link.exists()
    assert latest_link.is_symlink()
    assert Path(os.path.realpath(latest_link)) == run_dir


def test_path_retrieval():
    """Test get_path method."""

    # Test valid path retrieval
    data_dir = project_paths.get_path("DATA_DIR")
    assert isinstance(data_dir, str)
    assert Path(data_dir).exists()

    # Test invalid path key
    with pytest.raises(KeyError):
        project_paths.get_path("INVALID_PATH")
