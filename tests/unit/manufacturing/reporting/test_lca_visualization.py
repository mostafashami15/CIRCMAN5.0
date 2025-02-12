"""Test suite for LCA visualization methods."""

import pytest
import pandas as pd
import os
from datetime import datetime, timedelta

from circman5.manufacturing.lifecycle import LCAVisualizer
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def visualizer():
    """Create LCAVisualizer instance for testing."""
    return LCAVisualizer()


@pytest.fixture
def test_data():
    """Generate test data for visualizations."""
    data_generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return data_generator.generate_complete_lca_dataset()


@pytest.fixture
def impact_data():
    """Create sample impact data for testing."""
    return {
        "Manufacturing Impact": 1000.0,
        "Use Phase Impact": -2000.0,  # Negative because it's an environmental benefit
        "End of Life Impact": 500.0,
    }


def test_impact_distribution_plot(visualizer, impact_data, tmp_path):
    """Test creation of impact distribution visualization."""
    output_path = tmp_path / "impact_distribution.png"
    visualizer.plot_impact_distribution(impact_data, str(output_path))
    assert output_path.exists()


def test_lifecycle_comparison_plot(visualizer, impact_data, tmp_path):
    """Test creation of lifecycle comparison visualization."""
    output_path = tmp_path / "lifecycle_comparison.png"
    visualizer.plot_lifecycle_comparison(
        impact_data["Manufacturing Impact"],
        impact_data["Use Phase Impact"],
        impact_data["End of Life Impact"],
        str(output_path),
    )
    assert output_path.exists()


def test_material_flow_plot(visualizer, test_data, tmp_path):
    """Test creation of material flow visualization."""
    output_path = tmp_path / "material_flow.png"
    visualizer.plot_material_flow(test_data["material_flow"], str(output_path))
    assert output_path.exists()


def test_energy_consumption_plot(visualizer, test_data, tmp_path):
    """Test creation of energy consumption visualization."""
    output_path = tmp_path / "energy_trends.png"
    visualizer.plot_energy_consumption_trends(
        test_data["energy_consumption"], str(output_path)
    )
    assert output_path.exists()


def test_comprehensive_report(visualizer, impact_data, test_data, tmp_path):
    """Test generation of comprehensive visualization report."""
    output_dir = tmp_path / "lca_report"
    visualizer.create_comprehensive_report(
        impact_data,
        test_data["material_flow"],
        test_data["energy_consumption"],
        str(output_dir),
    )

    # Check that all expected files were created
    expected_files = [
        "impact_distribution.png",
        "lifecycle_comparison.png",
        "material_flow.png",
        "energy_trends.png",
    ]

    for file in expected_files:
        assert (output_dir / file).exists()
