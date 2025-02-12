"""Integration tests for manufacturing analysis system."""

import pytest
from pathlib import Path
import pandas as pd
from circman5.manufacturing.core import SoliTekManufacturingAnalysis


@pytest.fixture
def analyzer():
    """Create analyzer fixture with test data."""
    return SoliTekManufacturingAnalysis()


def test_performance_report(analyzer, tmp_path):
    """Test performance report generation."""
    report_path = tmp_path / "performance_report.xlsx"
    metrics = analyzer.analyze_manufacturing_performance()
    analyzer.report_generator.generate_performance_report(
        metrics, save_path=report_path
    )
    assert report_path.exists()


def test_metric_calculations(analyzer):
    """Test metric calculations."""
    metrics = analyzer.analyze_manufacturing_performance()
    assert "efficiency" in metrics
    assert "quality" in metrics
    assert "sustainability" in metrics


def test_visualization_generation(analyzer, tmp_path):
    """Test visualization generation with validated data."""
    viz_dir = tmp_path / "visualizations"
    viz_dir.mkdir()

    # Test each visualization type
    for viz_type in ["production", "energy", "quality", "sustainability"]:
        viz_path = viz_dir / f"{viz_type}_viz.png"
        analyzer.generate_visualization(viz_type, str(viz_path))
        assert viz_path.exists()


def test_report_generation(analyzer, tmp_path):
    """Test comprehensive report generation."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    analyzer.generate_reports(output_dir=report_dir)
    assert any(report_dir.glob("*.xlsx"))


def test_data_validation(analyzer):
    """Test data validation functionality."""
    metrics = analyzer.analyze_manufacturing_performance()
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["efficiency", "quality", "sustainability"])
