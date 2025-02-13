# tests/unit/manufacturing/reporting/test_visualization_paths.py

import pytest
from pathlib import Path
from circman5.manufacturing.reporting.visualization_paths import (
    VisualizationPathManager,
)


@pytest.fixture
def path_manager():
    return VisualizationPathManager()


def test_get_visualization_path(path_manager):
    """Test getting visualization path with metric type."""
    path = path_manager.get_visualization_path("efficiency")
    assert isinstance(path, Path)
    assert "efficiency_" in path.name
    assert path.suffix == ".png"


def test_get_visualization_path_with_filename(path_manager):
    """Test getting visualization path with custom filename."""
    path = path_manager.get_visualization_path("quality", "custom_viz.png")
    assert isinstance(path, Path)
    assert path.name == "custom_viz.png"


def test_ensure_visualization_directory(path_manager, tmp_path):
    """Test visualization directory creation."""
    viz_dir = path_manager.ensure_visualization_directory(tmp_path)
    assert viz_dir.exists()
    assert viz_dir.is_dir()
    assert viz_dir.name == "visualizations"
