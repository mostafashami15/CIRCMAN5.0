"""
Integration tests for the complete manufacturing analysis system.
Tests the interaction between analyzers, main system, and visualization components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import TestDataGenerator


@pytest.fixture
def test_data():
    """Generate comprehensive test data."""
    generator = TestDataGenerator(start_date="2024-01-01", days=10)

    return {
        "production": generator.generate_production_data(),
        "energy": generator.generate_energy_data(),
        "quality": generator.generate_quality_data(),
        "material": generator.generate_material_flow_data(),
    }


@pytest.fixture
def analyzer():
    """Create analyzer instance with test data."""
    return SoliTekManufacturingAnalysis()


def test_end_to_end_analysis(analyzer, test_data, tmp_path):
    """
    Test complete analysis pipeline from data loading to report generation.
    """
    # Load test data
    analyzer.production_data = test_data["production"]
    analyzer.energy_data = test_data["energy"]
    analyzer.quality_data = test_data["quality"]
    analyzer.material_flow = test_data["material"]

    # Test efficiency analysis
    efficiency_metrics = analyzer.analyze_efficiency()
    assert efficiency_metrics is not None
    assert "yield_rate" in efficiency_metrics

    # Test quality analysis
    quality_metrics = analyzer.analyze_quality_metrics()
    assert quality_metrics is not None
    assert "avg_defect_rate" in quality_metrics

    # Test sustainability analysis
    sustainability_metrics = analyzer.calculate_sustainability_metrics()
    assert sustainability_metrics is not None
    assert "material_efficiency" in sustainability_metrics

    # Test visualization generation
    vis_path = tmp_path / "test_visualization.png"
    analyzer.generate_visualization("production", str(vis_path))
    assert os.path.exists(vis_path)

    # Test report generation
    report_path = tmp_path / "test_report.xlsx"
    analyzer.generate_comprehensive_report(str(report_path))
    assert os.path.exists(report_path)


def test_analyzer_integration(analyzer, test_data):
    """
    Test integration between different analyzers.
    """
    # Load test data
    analyzer.production_data = test_data["production"]
    analyzer.material_flow = test_data["material"]

    # Test efficiency and material analysis integration
    efficiency_metrics = analyzer.analyze_efficiency()
    material_metrics = analyzer._calculate_material_utilization()

    assert efficiency_metrics is not None
    assert material_metrics is not None

    # Verify metrics are consistent
    assert 0 <= efficiency_metrics.get("yield_rate", 0) <= 100
    assert 0 <= material_metrics <= 100


def test_optimization_pipeline(analyzer, test_data):
    """
    Test AI optimization pipeline integration.
    """
    # Load test data
    analyzer.production_data = test_data["production"]
    analyzer.quality_data = test_data["quality"]

    # Train optimization model
    if not analyzer.is_optimizer_trained:
        metrics = analyzer.train_optimization_model()
        assert metrics is not None
        assert "mse" in metrics
        assert "r2" in metrics

    # Test process optimization
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    optimized_params = analyzer.optimize_process_parameters(current_params)
    assert optimized_params is not None
    assert all(param in optimized_params for param in current_params.keys())


def test_error_handling_integration(analyzer):
    """
    Test error handling across integrated components.
    """
    # Test with empty data
    empty_metrics = analyzer.analyze_efficiency()
    assert empty_metrics.get("error") == "No production data available"

    # Test with invalid optimization parameters
    invalid_params = {
        "input_amount": -100.0,  # Invalid negative value
        "energy_used": 150.0,
    }

    # Should handle invalid parameters gracefully
    with pytest.raises(ValueError):
        analyzer.optimize_process_parameters(invalid_params)


def test_visualization_integration(analyzer, test_data, tmp_path):
    """
    Test integration of visualization components.
    """
    # Load test data
    analyzer.production_data = test_data["production"]
    analyzer.quality_data = test_data["quality"]
    analyzer.material_flow = test_data["material"]

    # Test different visualization types
    for metric_type in ["production", "quality", "sustainability"]:
        vis_path = tmp_path / f"test_{metric_type}.png"
        analyzer.generate_visualization(metric_type, str(vis_path))
        assert os.path.exists(vis_path)

    # Test KPI dashboard generation
    dashboard_path = tmp_path / "test_dashboard.png"
    analyzer.generate_performance_report(str(dashboard_path))
    assert os.path.exists(dashboard_path)
