# tests/unit/manufacturing/reporting/test_visualization_paths.py

import pytest
from pathlib import Path
from circman5.manufacturing.reporting.visualization_paths import (
    VisualizationPathManager,
)
from circman5.utils.results_manager import results_manager


@pytest.fixture(scope="module")
def viz_dir():
    """Get visualization directory from ResultsManager."""
    return results_manager.get_path("visualizations")


@pytest.fixture
def path_manager():
    return VisualizationPathManager()


def test_get_visualization_path(path_manager, viz_dir):
    """Test getting visualization path with metric type."""
    path = path_manager.get_visualization_path("efficiency")
    assert isinstance(path, Path)
    assert path.parent == viz_dir
    assert "efficiency_" in path.name
    assert path.suffix == ".png"


def test_get_visualization_path_with_filename(path_manager, viz_dir):
    """Test getting visualization path with custom filename."""
    filename = "custom_viz.png"
    path = path_manager.get_visualization_path("quality", filename)
    assert isinstance(path, Path)
    assert path.parent == viz_dir
    assert path.name == filename


def test_ensure_visualization_directory(path_manager, viz_dir):
    """Test visualization directory creation."""
    output_dir = path_manager.ensure_visualization_directory()
    assert output_dir == viz_dir
    assert output_dir.exists()
    assert output_dir.is_dir()
