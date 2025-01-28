import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from circman5.visualization import ManufacturingVisualizer


@pytest.fixture
def sample_data():
    """Create sample manufacturing data for testing visualizations."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")

    efficiency_data = pd.DataFrame(
        {
            "timestamp": dates,
            "production_rate": np.random.normal(100, 10, 10),
            "energy_efficiency": np.random.normal(0.8, 0.1, 10),
        }
    )

    quality_data = pd.DataFrame(
        {
            "timestamp": dates,
            "defect_rate": np.random.normal(2, 0.5, 10),
            "quality_score": np.random.normal(95, 2, 10),
        }
    )

    resource_data = pd.DataFrame(
        {
            "timestamp": dates,
            "material_consumption": np.random.normal(1000, 100, 10),
            "water_usage": np.random.normal(500, 50, 10),
            "waste_generated": np.random.normal(50, 5, 10),
            "resource_efficiency": np.random.normal(0.95, 0.02, 10),
        }
    )

    return {
        "efficiency": efficiency_data,
        "quality": quality_data,
        "resources": resource_data,
    }


@pytest.fixture
def visualizer():
    """Create visualization instance for testing."""
    return ManufacturingVisualizer()


def test_efficiency_plot(visualizer, sample_data, tmp_path):
    """Test efficiency trend plotting."""
    plot_path = tmp_path / "efficiency_plot.png"
    visualizer.plot_efficiency_trends(sample_data["efficiency"], str(plot_path))
    assert plot_path.exists()


def test_quality_plot(visualizer, sample_data, tmp_path):
    """Test quality metrics plotting."""
    plot_path = tmp_path / "quality_plot.png"
    visualizer.plot_quality_metrics(sample_data["quality"], str(plot_path))
    assert plot_path.exists()


def test_resource_plot(visualizer, sample_data, tmp_path):
    """Test resource usage plotting."""
    plot_path = tmp_path / "resource_plot.png"
    visualizer.plot_resource_usage(sample_data["resources"], str(plot_path))
    assert plot_path.exists()


def test_dashboard_creation(visualizer, sample_data, tmp_path):
    """Test comprehensive dashboard creation."""
    dashboard_path = tmp_path / "dashboard.png"
    visualizer.create_performance_dashboard(sample_data, str(dashboard_path))
    assert dashboard_path.exists()


def test_kpi_dashboard(visualizer, tmp_path):
    """Test KPI dashboard creation."""
    metrics = {
        "efficiency": 85.5,
        "quality_score": 92.3,
        "resource_efficiency": 78.9,
        "energy_efficiency": 88.7,
    }

    dashboard_path = tmp_path / "kpi_dashboard.png"
    visualizer.create_kpi_dashboard(metrics, str(dashboard_path))
    assert dashboard_path.exists()
