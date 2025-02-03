"""
Test suite for efficiency analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from circman5.analysis.efficiency import EfficiencyAnalyzer


@pytest.fixture
def sample_production_data():
    """Create sample production data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")

    return pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(10)],
            "timestamp": dates,
            "output_amount": np.random.uniform(80, 100, 10),
            "input_amount": np.random.uniform(90, 110, 10),
            "cycle_time": np.random.uniform(45, 55, 10),
            "energy_used": np.random.uniform(140, 160, 10),
        }
    )


@pytest.fixture
def analyzer():
    """Create analyzer instance for testing."""
    return EfficiencyAnalyzer()


def test_batch_efficiency_analysis(analyzer, sample_production_data):
    """Test batch efficiency analysis functionality."""
    metrics = analyzer.analyze_batch_efficiency(sample_production_data)

    assert "yield_rate" in metrics
    assert "cycle_time_efficiency" in metrics
    assert "energy_efficiency" in metrics

    # Test metric values are within expected ranges
    assert 0 <= metrics["yield_rate"] <= 100
    assert metrics["cycle_time_efficiency"] > 0
    assert metrics["energy_efficiency"] > 0


def test_empty_data_handling(analyzer):
    """Test handling of empty data."""
    empty_data = pd.DataFrame()
    metrics = analyzer.analyze_batch_efficiency(empty_data)

    assert isinstance(metrics, dict)
    assert len(metrics) == 0


def test_metric_calculation_accuracy(analyzer, sample_production_data):
    """Test accuracy of efficiency metric calculations."""
    metrics = analyzer.analyze_batch_efficiency(sample_production_data)

    # Calculate expected yield rate manually
    expected_yield = (
        sample_production_data["output_amount"].mean()
        / sample_production_data["input_amount"].mean()
        * 100
    )

    # Use 0.5% tolerance for floating-point calculations
    assert abs(metrics["yield_rate"] - expected_yield) < 0.5


def test_data_validation(analyzer):
    """Test data validation checks."""
    # Test with minimal required columns
    minimal_data = pd.DataFrame(
        {"batch_id": ["TEST_001"], "output_amount": [90], "input_amount": [100]}
    )

    metrics = analyzer.analyze_batch_efficiency(minimal_data)
    assert "yield_rate" in metrics

    # Test with invalid data
    invalid_data = pd.DataFrame(
        {
            "batch_id": ["TEST_001"],
            "output_amount": [-100],  # Invalid negative value
            "input_amount": [100],
            "cycle_time": [50],
            "energy_used": [150],
        }
    )

    metrics = analyzer.analyze_batch_efficiency(invalid_data)
    assert metrics.get("yield_rate", 0) >= 0  # Should handle negative values gracefully


def test_missing_optional_columns(analyzer):
    """Test handling of missing optional columns."""
    # Test with only required columns
    basic_data = pd.DataFrame(
        {"batch_id": ["TEST_001"], "output_amount": [90], "input_amount": [100]}
    )

    metrics = analyzer.analyze_batch_efficiency(basic_data)
    assert "yield_rate" in metrics
    assert "cycle_time_efficiency" not in metrics
    assert "energy_efficiency" not in metrics
