# tests/integration/system/test_script.py
"""Integration tests for manufacturing analysis system."""

import pytest
from pathlib import Path
import pandas as pd
from circman5.manufacturing.core import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator


# This fixture returns an empty analyzer instance (if needed for some tests)
@pytest.fixture
def analyzer():
    """Create an analyzer fixture (empty instance)."""
    return SoliTekManufacturingAnalysis()


# This fixture returns an analyzer instance pre-populated with test data.
@pytest.fixture(scope="module")
def analyzer_with_data():
    """Create an analyzer instance pre-populated with test data."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    analyzer = SoliTekManufacturingAnalysis()
    analyzer.production_data = generator.generate_production_data()
    analyzer.quality_data = generator.generate_quality_data()
    analyzer.energy_data = generator.generate_energy_data()
    analyzer.material_flow = generator.generate_material_flow_data()
    analyzer.lca_data = {
        "material_flow": analyzer.material_flow,
        "energy_consumption": analyzer.energy_data,
        "process_data": generator.generate_lca_process_data(),
    }
    return analyzer


def test_performance_report(analyzer_with_data, tmp_path):
    """Test performance report generation."""
    report_path = tmp_path / "performance_report.xlsx"
    metrics = analyzer_with_data.analyze_manufacturing_performance()
    analyzer_with_data.report_generator.generate_performance_report(
        metrics, save_path=report_path
    )
    assert report_path.exists()


def test_metric_calculations(analyzer_with_data):
    """Test metric calculations."""
    metrics = analyzer_with_data.analyze_manufacturing_performance()
    assert "efficiency" in metrics
    assert "quality" in metrics
    assert "sustainability" in metrics


def test_visualization_generation(analyzer_with_data, tmp_path):
    """Test visualization generation with validated data."""
    viz_dir = tmp_path / "visualizations"
    viz_dir.mkdir()

    # Test each visualization type
    for viz_type in ["production", "energy", "quality", "sustainability"]:
        viz_path = viz_dir / f"{viz_type}_viz.png"
        analyzer_with_data.generate_visualization(viz_type, str(viz_path))
        assert viz_path.exists()


def test_report_generation(analyzer_with_data, tmp_path):
    """Test comprehensive report generation."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    analyzer_with_data.generate_reports(output_dir=report_dir)
    assert any(report_dir.glob("*.xlsx"))


def test_data_validation(analyzer_with_data):
    """Test data validation functionality."""
    metrics = analyzer_with_data.analyze_manufacturing_performance()
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["efficiency", "quality", "sustainability"])
