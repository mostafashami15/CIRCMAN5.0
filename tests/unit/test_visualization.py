import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from circman5.config.project_paths import project_paths
from circman5.visualization.manufacturing_visualizer import ManufacturingVisualizer


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


def test_efficiency_plot(visualizer, sample_data):
    """Test efficiency trend plotting."""
    visualizer.plot_efficiency_trends(sample_data["efficiency"])
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "visualizations" / "efficiency_trends.png").exists()


def test_quality_plot(visualizer, sample_data):
    """Test quality metrics plotting."""
    visualizer.plot_quality_metrics(sample_data["quality"])
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "visualizations" / "quality_metrics.png").exists()


def test_resource_plot(visualizer, sample_data):
    """Test resource usage plotting."""
    # Create sample resource data with correct columns
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")
    resource_data = pd.DataFrame(
        {
            "timestamp": dates,
            "material_consumption": np.random.normal(1000, 100, 10),
            "water_usage": np.random.normal(500, 50, 10),
            "waste_generated": np.random.normal(50, 5, 10),
            "resource_efficiency": np.random.normal(0.95, 0.02, 10),
        }
    )

    visualizer.plot_resource_usage(resource_data)
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "visualizations" / "resource_usage.png").exists()


def test_dashboard_creation(visualizer, sample_data):
    """Test comprehensive dashboard creation."""
    # Create complete sample data with all required columns
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")

    # Update the sample data with complete datasets
    complete_data = {
        "efficiency": pd.DataFrame(
            {
                "timestamp": dates,
                "production_rate": np.random.normal(100, 10, 10),
                "energy_efficiency": np.random.normal(0.8, 0.1, 10),
            }
        ),
        "quality": pd.DataFrame(
            {"timestamp": dates, "quality_score": np.random.normal(95, 2, 10)}
        ),
        "resources": pd.DataFrame(
            {
                "timestamp": dates,
                "material_consumption": np.random.normal(1000, 100, 10),
                "water_usage": np.random.normal(500, 50, 10),
                "resource_efficiency": np.random.normal(0.95, 0.02, 10),
            }
        ),
    }

    visualizer.create_performance_dashboard(complete_data)
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "visualizations" / "performance_dashboard.png").exists()


def test_kpi_dashboard(visualizer):
    """Test KPI dashboard creation."""
    metrics = {
        "efficiency": 85.5,
        "quality_score": 92.3,
        "resource_efficiency": 78.9,
        "energy_efficiency": 88.7,
    }
    visualizer.create_kpi_dashboard(metrics)
    run_dir = project_paths.get_run_directory()
    assert (run_dir / "visualizations" / "kpi_dashboard.png").exists()
