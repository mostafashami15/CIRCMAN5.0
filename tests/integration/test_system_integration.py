"""Integration tests for visualization system."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from circman5.visualization.manufacturing_visualizer import ManufacturingVisualizer
from circman5.visualization.lca_visualizer import LCAVisualizer
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths
from circman5.analysis.lca.core import LCAAnalyzer


@pytest.fixture(scope="module")
def test_data():
    """Generate test data for visualization."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
        "lca": generator.generate_complete_lca_dataset(),
    }


@pytest.fixture(scope="module")
def test_run_dir():
    """Create and maintain test run directory."""
    run_dir = project_paths.get_run_directory()

    # Create all required directories
    for subdir in ["visualizations", "reports", "input_data"]:
        (run_dir / subdir).mkdir(exist_ok=True)

    return run_dir


@pytest.fixture(scope="module")
def manufacturing_viz():
    """Create manufacturing visualizer instance."""
    return ManufacturingVisualizer()


@pytest.fixture(scope="module")
def lca_viz():
    """Create LCA visualizer instance."""
    return LCAVisualizer()


def test_manufacturing_visualization(manufacturing_viz, test_data, test_run_dir):
    """Test manufacturing visualization generation."""
    # Prepare directory
    viz_dir = test_run_dir / "visualizations"

    # Test efficiency trends visualization
    efficiency_data = pd.DataFrame(
        {
            "timestamp": test_data["production"]["timestamp"],
            "production_rate": test_data["production"]["output_amount"],
            "energy_efficiency": test_data["production"]["yield_rate"] / 100,
        }
    )
    manufacturing_viz.plot_efficiency_trends(
        efficiency_data, save_path=str(viz_dir / "efficiency_trends.png")
    )
    assert (viz_dir / "efficiency_trends.png").exists()

    # Test quality metrics visualization
    quality_data = test_data["quality"].copy()
    # Rename test_timestamp to timestamp for consistency
    quality_data = quality_data.rename(columns={"test_timestamp": "timestamp"})
    # Add quality score for visualization
    quality_data["quality_score"] = 100 - quality_data["defect_rate"]

    manufacturing_viz.plot_quality_metrics(
        quality_data, save_path=str(viz_dir / "quality_metrics.png")
    )

    # Test resource usage visualization
    resource_data = pd.DataFrame(
        {
            "timestamp": test_data["material"]["timestamp"],
            "material_consumption": test_data["material"]["quantity_used"],
            "water_usage": np.random.uniform(50, 100, len(test_data["material"])),
            "waste_generated": test_data["material"]["waste_generated"],
            "resource_efficiency": test_data["material"]["recycled_amount"]
            / test_data["material"]["quantity_used"]
            * 100,
        }
    )
    manufacturing_viz.plot_resource_usage(
        resource_data, save_path=str(viz_dir / "resource_usage.png")
    )
    assert (viz_dir / "resource_usage.png").exists()


def test_lca_visualization(lca_viz, test_data, test_run_dir):
    """Test LCA visualization generation."""
    viz_dir = test_run_dir / "visualizations"

    # Calculate LCA impacts
    lca = LCAAnalyzer()
    impact_data = {
        "Manufacturing": 45.2,
        "Use Phase": -120.5,
        "End of Life": 15.8,
        "Transport": 8.3,
    }

    # Test impact distribution visualization
    lca_viz.plot_impact_distribution(
        impact_data, save_path=str(viz_dir / "impact_distribution.png")
    )
    assert (viz_dir / "impact_distribution.png").exists()

    # Test lifecycle comparison visualization
    lca_viz.plot_lifecycle_comparison(
        manufacturing_impact=45.2,
        use_phase_impact=-120.5,
        end_of_life_impact=15.8,
        save_path=str(viz_dir / "lifecycle_comparison.png"),
    )
    assert (viz_dir / "lifecycle_comparison.png").exists()

    # Test material flow visualization
    lca_viz.plot_material_flow(
        test_data["material"], save_path=str(viz_dir / "material_flow.png")
    )
    assert (viz_dir / "material_flow.png").exists()


def test_comprehensive_visualization(
    manufacturing_viz, lca_viz, test_data, test_run_dir
):
    """Test comprehensive visualization report generation."""
    viz_dir = test_run_dir / "visualizations"

    # Create performance dashboard
    monitor_data = {
        "efficiency": pd.DataFrame(
            {
                "timestamp": test_data["production"]["timestamp"],
                "production_rate": test_data["production"]["output_amount"],
            }
        ),
        "quality": pd.DataFrame(
            {
                "timestamp": test_data["quality"]["test_timestamp"],
                "quality_score": 100 - test_data["quality"]["defect_rate"],
            }
        ),
        "resources": pd.DataFrame(
            {
                "timestamp": test_data["material"]["timestamp"],
                "material_consumption": test_data["material"]["quantity_used"],
                "resource_efficiency": test_data["material"]["recycled_amount"]
                / test_data["material"]["quantity_used"]
                * 100,
            }
        ),
    }

    manufacturing_viz.create_performance_dashboard(
        monitor_data, save_path=str(viz_dir / "performance_dashboard.png")
    )
    assert (viz_dir / "performance_dashboard.png").exists()

    # Create LCA comprehensive report
    output_dir = test_run_dir / "reports" / "lca_visualizations"
    output_dir.mkdir(exist_ok=True)

    lca_viz.create_comprehensive_report(
        impact_data={"Manufacturing": 45.2, "Use Phase": -120.5},
        material_data=test_data["material"],
        energy_data=test_data["energy"],
        output_dir=str(output_dir),
    )

    # Verify LCA report files
    expected_files = [
        "impact_distribution.png",
        "lifecycle_comparison.png",
        "material_flow.png",
        "energy_trends.png",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()


def test_visualization_integration(manufacturing_viz, lca_viz, test_data, test_run_dir):
    """Test integration between visualization components and other modules."""
    viz_dir = test_run_dir / "visualizations"

    # Create KPI dashboard with both manufacturing and LCA metrics
    kpi_data = {
        "efficiency": test_data["production"]["yield_rate"].mean(),
        "quality_score": 100 - test_data["quality"]["defect_rate"].mean(),
        "resource_efficiency": (
            test_data["material"]["recycled_amount"].sum()
            / test_data["material"]["quantity_used"].sum()
            * 100
        ),
        "energy_efficiency": test_data["energy"]["efficiency_rate"].mean(),
    }

    manufacturing_viz.create_kpi_dashboard(
        kpi_data, save_path=str(viz_dir / "kpi_dashboard.png")
    )
    assert (viz_dir / "kpi_dashboard.png").exists()


def test_visualization_data_consistency(test_data):
    """Test consistency of data used in visualizations."""
    # Verify timestamp ranges match across datasets
    start_date = min(
        test_data["production"]["timestamp"].min(),
        test_data["energy"]["timestamp"].min(),
        test_data["material"]["timestamp"].min(),
    )
    end_date = max(
        test_data["production"]["timestamp"].max(),
        test_data["energy"]["timestamp"].max(),
        test_data["material"]["timestamp"].max(),
    )

    # Check all datasets fall within the same time range
    for data_type in ["production", "energy", "material"]:
        assert test_data[data_type]["timestamp"].min() >= start_date
        assert test_data[data_type]["timestamp"].max() <= end_date

    # Verify unit consistency
    assert all(0 <= val <= 100 for val in test_data["quality"]["efficiency"])
    assert all(0 <= val <= 100 for val in test_data["production"]["yield_rate"])
