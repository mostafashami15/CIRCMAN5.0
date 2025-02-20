# tests/unit/manufacturing/analyzers/test_quality.py

"""Test suite for quality analyzer module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from circman5.manufacturing.analyzers.quality import QualityAnalyzer
from circman5.utils.results_manager import results_manager


@pytest.fixture
def sample_quality_data():
    """Create sample quality control data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")

    return pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(10)],
            "timestamp": dates,
            "efficiency": np.random.uniform(20, 22, 10),
            "defect_rate": np.random.uniform(1, 3, 10),
            "thickness_uniformity": np.random.uniform(94, 96, 10),
        }
    )


@pytest.fixture
def analyzer():
    """Create analyzer instance for testing."""
    return QualityAnalyzer()


def test_defect_rate_analysis(analyzer, sample_quality_data):
    """Test defect rate analysis functionality."""
    metrics = analyzer.analyze_defect_rates(sample_quality_data)

    assert "avg_defect_rate" in metrics
    assert "efficiency_score" in metrics
    assert "uniformity_score" in metrics

    # Test metric values are within expected ranges
    assert 0 <= metrics["avg_defect_rate"] <= 100
    assert 0 <= metrics["efficiency_score"] <= 100
    assert 0 <= metrics["uniformity_score"] <= 100

    # Save metrics report
    temp_path = Path("quality_metrics.xlsx")
    pd.DataFrame([metrics]).to_excel(temp_path)
    results_manager.save_file(temp_path, "reports")
    temp_path.unlink()


def test_quality_trends(analyzer, sample_quality_data):
    """Test quality trend analysis functionality."""
    trends = analyzer.identify_quality_trends(sample_quality_data)

    assert "defect_trend" in trends
    assert "efficiency_trend" in trends
    assert "uniformity_trend" in trends

    # Create a temporary file and ensure it exists before trying to unlink
    temp_viz = Path("quality_trends.png")
    analyzer.plot_trends(trends, str(temp_viz))

    # Verify file exists before trying operations
    assert temp_viz.exists(), "Visualization file was not created"
    results_manager.save_file(temp_viz, "visualizations")
    if temp_viz.exists():
        temp_viz.unlink()


def test_empty_data_handling(analyzer):
    """Test handling of empty data."""
    empty_data = pd.DataFrame()
    metrics = analyzer.analyze_defect_rates(empty_data)

    assert isinstance(metrics, dict)
    assert len(metrics) == 0


def test_metric_calculation_accuracy(analyzer, sample_quality_data):
    """Test accuracy of quality metric calculations."""
    metrics = analyzer.analyze_defect_rates(sample_quality_data)

    # Calculate expected metrics manually
    expected_defect_rate = sample_quality_data["defect_rate"].mean()
    expected_efficiency = sample_quality_data["efficiency"].mean()

    assert abs(metrics["avg_defect_rate"] - expected_defect_rate) < 0.01
    assert abs(metrics["efficiency_score"] - expected_efficiency) < 0.01

    # Save calculations for verification
    temp_path = Path("quality_calculations.xlsx")
    pd.DataFrame(
        {
            "calculated": metrics,
            "expected": {
                "avg_defect_rate": expected_defect_rate,
                "efficiency_score": expected_efficiency,
            },
        }
    ).to_excel(temp_path)
    results_manager.save_file(temp_path, "reports")
    temp_path.unlink()


def test_trend_calculation_consistency(analyzer, sample_quality_data):
    """Test consistency of trend calculations."""
    trends = analyzer.identify_quality_trends(sample_quality_data)

    # Save trend analysis
    temp_path = Path("quality_trends.xlsx")
    pd.DataFrame(trends).to_excel(temp_path)
    results_manager.save_file(temp_path, "reports")
    temp_path.unlink()


def test_visualization_output(analyzer, sample_quality_data):
    """Test that visualizations are saved to the correct directory."""
    temp_viz = Path("quality_test.png")
    trends = analyzer.identify_quality_trends(sample_quality_data)
    analyzer.plot_trends(trends, str(temp_viz))

    # Verify file exists before trying operations
    assert temp_viz.exists(), "Visualization file was not created"
    results_manager.save_file(temp_viz, "visualizations")
    if temp_viz.exists():
        temp_viz.unlink()
