"""
Test suite for quality analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from circman5.analysis.quality import QualityAnalyzer


@pytest.fixture
def sample_quality_data():
    """Create sample quality control data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="h")

    return pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(10)],
            "test_timestamp": dates,
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


def test_quality_trends(analyzer, sample_quality_data):
    """Test quality trend analysis functionality."""
    trends = analyzer.identify_quality_trends(sample_quality_data)

    assert "defect_trend" in trends
    assert "efficiency_trend" in trends
    assert "uniformity_trend" in trends

    assert len(trends["defect_trend"]) > 0
    assert len(trends["efficiency_trend"]) > 0
    assert len(trends["uniformity_trend"]) > 0


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


def test_trend_calculation_consistency(analyzer, sample_quality_data):
    """Test consistency of trend calculations."""
    trends = analyzer.identify_quality_trends(sample_quality_data)

    # Verify trend lengths match data grouping
    expected_length = len(
        pd.date_range(
            start=sample_quality_data["test_timestamp"].min(),
            end=sample_quality_data["test_timestamp"].max(),
            freq="D",
        )
    )

    assert len(trends["defect_trend"]) <= expected_length
