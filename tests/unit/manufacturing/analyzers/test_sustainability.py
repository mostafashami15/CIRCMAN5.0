# tests/unit/manufacturing/analyzers/test_sustainability.py
"""Test suite for sustainability analyzer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from circman5.manufacturing.analyzers.sustainability import SustainabilityAnalyzer
from circman5.config.project_paths import project_paths


@pytest.fixture
def sample_material_data():
    """Create sample material flow data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "material_type": ["Silicon", "Glass", "EVA", "Backsheet", "Frame"] * 2,
            "quantity_used": np.random.uniform(800, 1200, 10),
            "waste_generated": np.random.uniform(40, 60, 10),
            "recycled_amount": np.random.uniform(30, 50, 10),
        }
    )


@pytest.fixture
def sample_energy_data():
    """Create sample energy consumption data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "energy_source": ["grid", "solar", "wind"] * 3 + ["grid"],
            "energy_consumption": np.random.uniform(400, 600, 10),
            "efficiency_rate": np.random.uniform(0.8, 0.95, 10),
        }
    )


@pytest.fixture
def analyzer():
    """Create analyzer instance for testing."""
    return SustainabilityAnalyzer()


def test_material_efficiency_analysis(analyzer, sample_material_data, reports_dir):
    """Test material efficiency analysis functionality."""
    metrics = analyzer.analyze_material_efficiency(sample_material_data)

    assert "material_efficiency" in metrics
    assert "recycling_rate" in metrics
    assert "waste_reduction" in metrics

    # Test metric values are within expected ranges
    assert 0 <= metrics["material_efficiency"] <= 100
    assert 0 <= metrics["recycling_rate"] <= 100
    assert 0 <= metrics["waste_reduction"] <= 100

    # Save and verify metrics
    report_path = reports_dir / "material_efficiency.xlsx"
    pd.DataFrame([metrics]).to_excel(report_path)
    assert report_path.exists()


def test_carbon_footprint_calculation(analyzer, sample_energy_data, reports_dir):
    """Test carbon footprint calculation functionality."""
    footprint = analyzer.calculate_carbon_footprint(sample_energy_data)

    assert isinstance(footprint, float)
    assert footprint >= 0

    # Test with only renewable energy
    renewable_data = sample_energy_data.copy()
    renewable_data["energy_source"] = "solar"
    renewable_footprint = analyzer.calculate_carbon_footprint(renewable_data)
    assert renewable_footprint == 0

    # Save calculations
    report_path = reports_dir / "carbon_footprint.xlsx"
    pd.DataFrame(
        {"total_footprint": [footprint], "renewable_footprint": [renewable_footprint]}
    ).to_excel(report_path)
    assert report_path.exists()


def test_sustainability_score_calculation(analyzer, reports_dir):
    """Test sustainability score calculation."""
    score = analyzer.calculate_sustainability_score(80, 70, 90)

    assert 0 <= score <= 100
    assert isinstance(score, float)

    # Save score
    report_path = reports_dir / "sustainability_score.xlsx"
    pd.DataFrame(
        {
            "material_efficiency": [80],
            "recycling_rate": [70],
            "energy_efficiency": [90],
            "total_score": [score],
        }
    ).to_excel(report_path)
    assert report_path.exists()


def test_empty_data_handling(analyzer):
    """Test handling of empty data."""
    empty_data = pd.DataFrame()
    material_metrics = analyzer.analyze_material_efficiency(empty_data)
    carbon_footprint = analyzer.calculate_carbon_footprint(empty_data)

    assert isinstance(material_metrics, dict)
    assert len(material_metrics) == 0
    assert carbon_footprint == 0


def test_metric_calculation_accuracy(analyzer, sample_material_data, reports_dir):
    """Test accuracy of sustainability metric calculations."""
    metrics = analyzer.analyze_material_efficiency(sample_material_data)

    # Calculate expected metrics manually
    total_used = sample_material_data["quantity_used"].sum()
    total_waste = sample_material_data["waste_generated"].sum()
    total_recycled = sample_material_data["recycled_amount"].sum()

    expected_efficiency = (
        (total_used - total_waste) / total_used * 100 if total_used > 0 else 0
    )
    expected_recycling = total_recycled / total_waste * 100 if total_waste > 0 else 0

    assert abs(metrics["material_efficiency"] - expected_efficiency) < 0.01
    assert abs(metrics["recycling_rate"] - expected_recycling) < 0.01

    # Save calculations for verification
    report_path = reports_dir / "sustainability_calculations.xlsx"
    pd.DataFrame(
        {
            "calculated": metrics,
            "expected": {
                "material_efficiency": expected_efficiency,
                "recycling_rate": expected_recycling,
            },
        }
    ).to_excel(report_path)
    assert report_path.exists()


def test_visualization_output(
    analyzer, sample_material_data, sample_energy_data, visualizations_dir
):
    """Test that visualizations are saved to the correct directory."""
    viz_path = visualizations_dir / "sustainability_test.png"
    analyzer.plot_sustainability_metrics(
        sample_material_data, sample_energy_data, str(viz_path)
    )
    assert viz_path.exists()
    print(f"Created visualization at: {viz_path}")  # Debug print
