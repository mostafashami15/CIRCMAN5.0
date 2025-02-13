# tests/unit/manufacturing/reporting/test_visualizations.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from circman5.manufacturing.reporting.visualizations import ManufacturingVisualizer
from circman5.utils.errors import DataError
from circman5.utils.result_paths import get_run_directory


@pytest.fixture(scope="module")
def test_run_dir():
    """Create and return test run directory."""
    run_dir = get_run_directory()
    print(f"\nTest outputs will be saved in: {run_dir}")
    return run_dir


@pytest.fixture
def visualizer():
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
            "test_timestamp": dates,  # Using test_timestamp as required
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


def test_visualize_production_trends(visualizer, sample_production_data, test_run_dir):
    """Test production trends visualization."""
    save_path = test_run_dir / "visualizations" / "production_trends.png"
    visualizer.visualize_production_trends(sample_production_data, str(save_path))
    assert save_path.exists()
    print(f"\nProduction trends visualization saved to: {save_path}")


def test_visualize_quality_metrics(visualizer, sample_quality_data, test_run_dir):
    """Test quality metrics visualization."""
    save_path = test_run_dir / "visualizations" / "quality_metrics.png"
    visualizer.visualize_quality_metrics(sample_quality_data, None, str(save_path))
    assert save_path.exists()
    print(f"\nQuality metrics visualization saved to: {save_path}")


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
    test_run_dir,
):
    """Test performance dashboard creation."""
    monitor_data = {
        "efficiency": sample_production_data,
        "quality": sample_quality_data,
        "resources": sample_resource_data,
    }
    save_path = test_run_dir / "visualizations" / "performance_dashboard.png"
    visualizer.create_performance_dashboard(monitor_data, str(save_path))
    assert save_path.exists()
    print(f"\nPerformance dashboard saved to: {save_path}")


def test_plot_efficiency_trends(visualizer, sample_production_data, test_run_dir):
    """Test efficiency trends plotting."""
    save_path = test_run_dir / "visualizations" / "efficiency_trends.png"
    visualizer.plot_efficiency_trends(sample_production_data, str(save_path))
    assert save_path.exists()
    print(f"\nEfficiency trends plot saved to: {save_path}")


def test_plot_quality_metrics(visualizer, sample_quality_data, test_run_dir):
    """Test quality metrics plotting."""
    save_path = test_run_dir / "visualizations" / "quality_metrics_plot.png"
    visualizer.plot_quality_metrics(sample_quality_data, str(save_path))
    assert save_path.exists()
    print(f"\nQuality metrics plot saved to: {save_path}")


def test_plot_resource_usage(visualizer, sample_resource_data, test_run_dir):
    """Test resource usage plotting."""
    save_path = test_run_dir / "visualizations" / "resource_usage.png"
    visualizer.plot_resource_usage(sample_resource_data, str(save_path))
    assert save_path.exists()
    print(f"\nResource usage plot saved to: {save_path}")


def test_create_kpi_dashboard(visualizer, test_run_dir):
    """Test KPI dashboard creation."""
    metrics_data = {
        "efficiency": 95.5,
        "quality_score": 98.0,
        "resource_efficiency": 92.5,
        "energy_efficiency": 94.0,
    }
    save_path = test_run_dir / "visualizations" / "kpi_dashboard.png"
    visualizer.create_kpi_dashboard(metrics_data, str(save_path))
    assert save_path.exists()
    print(f"\nKPI dashboard saved to: {save_path}")
