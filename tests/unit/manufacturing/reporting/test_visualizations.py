# tests/unit/manufacturing/reporting/test_visualizations.py

import shutil
from typing import Union
from matplotlib import pyplot as plt
import pytest
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
from datetime import datetime
from pathlib import Path
from circman5.manufacturing.reporting.visualizations import ManufacturingVisualizer
from circman5.utils.errors import DataError
from circman5.utils.results_manager import results_manager


@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    """Setup test environment and ensure cleanup."""
    # Setup occurs through ResultsManager initialization
    plt.close("all")  # Close any existing plots
    yield
    plt.close("all")  # Cleanup any open figures


@pytest.fixture(scope="module")
def viz_dir():
    """Get visualizations directory."""
    viz_path = results_manager.get_path("visualizations")
    return viz_path


@pytest.fixture
def visualizer():
    """Initialize visualizer."""
    return ManufacturingVisualizer()


@pytest.fixture
def sample_production_data():
    """Generate sample production data."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "output_amount": np.random.uniform(90, 110, 10),
            "yield_rate": np.random.uniform(90, 98, 10),
            "production_line": ["Line A"] * 5 + ["Line B"] * 5,
            "cycle_time": np.random.uniform(45, 55, 10),
            "production_rate": np.random.uniform(80, 100, 10),
        }
    )


@pytest.fixture
def sample_quality_data():
    """Generate sample quality data."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "efficiency": np.random.uniform(85, 95, 10),
            "defect_rate": np.random.uniform(1, 5, 10),
            "thickness_uniformity": np.random.uniform(90, 98, 10),
            "quality_score": np.random.uniform(90, 98, 10),
        }
    )


@pytest.fixture
def sample_resource_data():
    """Generate sample resource data."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "material_consumption": np.random.uniform(900, 1100, 10),
            "resource_efficiency": np.random.uniform(85, 95, 10),
            "water_usage": np.random.uniform(100, 200, 10),
        }
    )


def test_visualize_production_trends(visualizer, sample_production_data, viz_dir):
    """Test production trends visualization."""
    filename = "production_trends.png"
    target_path = viz_dir / filename

    visualizer.visualize_production_trends(sample_production_data, filename)
    assert target_path.exists()


def test_visualize_quality_metrics(visualizer, sample_quality_data, viz_dir):
    """Test quality metrics visualization."""
    filename = "quality_metrics.png"
    save_path = results_manager.save_file(viz_dir / filename, "visualizations")

    visualizer.visualize_quality_metrics(sample_quality_data, None, str(save_path))
    assert save_path.exists()


def test_empty_data_handling(visualizer):
    """Test handling of empty datasets."""
    empty_df = pd.DataFrame()
    with pytest.raises(DataError):
        visualizer.visualize_quality_metrics(empty_df, None)


def test_create_performance_dashboard(
    visualizer,
    sample_production_data,
    sample_quality_data,
    sample_resource_data,
    viz_dir,
):
    """Test performance dashboard creation."""
    monitor_data = {
        "efficiency": sample_production_data,
        "quality": sample_quality_data,
        "resources": sample_resource_data,
    }

    filename = "performance_dashboard.png"
    save_path = results_manager.save_file(viz_dir / filename, "visualizations")

    visualizer.create_performance_dashboard(monitor_data, str(save_path))
    assert save_path.exists()


def test_plot_efficiency_trends(visualizer, sample_production_data, viz_dir):
    """Test efficiency trends plotting."""
    filename = "efficiency_trends.png"
    save_path = results_manager.save_file(viz_dir / filename, "visualizations")

    visualizer.plot_efficiency_trends(sample_production_data, str(save_path))
    assert save_path.exists()


def test_plot_quality_metrics(visualizer, sample_quality_data, viz_dir):
    """Test quality metrics plotting."""
    filename = "quality_metrics_plot.png"
    save_path = results_manager.save_file(viz_dir / filename, "visualizations")

    visualizer.plot_quality_metrics(sample_quality_data, str(save_path))
    assert save_path.exists()


def test_plot_resource_usage(visualizer, sample_resource_data, viz_dir):
    """Test resource usage plotting."""
    filename = "resource_usage.png"
    save_path = results_manager.save_file(viz_dir / filename, "visualizations")

    visualizer.plot_resource_usage(sample_resource_data, str(save_path))
    assert save_path.exists()


def test_create_kpi_dashboard(visualizer, viz_dir):
    """Test KPI dashboard creation."""
    metrics_data = {
        "efficiency": 95.5,
        "quality_score": 98.0,
        "resource_efficiency": 92.5,
        "energy_efficiency": 94.0,
    }

    filename = "kpi_dashboard.png"
    save_path = results_manager.save_file(viz_dir / filename, "visualizations")

    visualizer.create_kpi_dashboard(metrics_data, str(save_path))
    assert save_path.exists()


def test_error_handling(visualizer):
    """Test error handling for invalid inputs."""
    empty_df = pd.DataFrame()
    with pytest.raises(DataError):
        visualizer.visualize_production_trends(empty_df)

    with pytest.raises(DataError):
        visualizer.visualize_quality_metrics(empty_df)
