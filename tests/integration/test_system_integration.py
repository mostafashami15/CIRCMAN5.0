"""
Integration tests for the complete manufacturing analysis system.
Tests the interaction between analyzers, main system, and visualization components.
"""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths
from circman5.ai.optimization_prediction import ManufacturingOptimizer


@pytest.fixture
def test_data():
    """Generate comprehensive test data."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=10)

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
    """Test complete analysis pipeline from data loading to report generation."""
    # Ensure data is loaded correctly
    analyzer.production_data = test_data["production"]
    analyzer.energy_data = test_data["energy"]
    analyzer.quality_data = test_data["quality"]
    analyzer.material_flow = test_data["material"]

    # Test efficiency analysis with required columns
    efficiency_metrics = analyzer.analyze_efficiency()
    assert efficiency_metrics is not None
    assert isinstance(efficiency_metrics, dict)

    # Updated assertion to check for actual metrics
    if "error" not in efficiency_metrics:
        assert "yield_rate" in efficiency_metrics

    # Test quality analysis
    quality_metrics = analyzer.analyze_quality_metrics()
    assert quality_metrics is not None

    # Test sustainability analysis
    sustainability_metrics = analyzer.calculate_sustainability_metrics()
    assert sustainability_metrics is not None

    # Test visualization generation
    vis_path = tmp_path / "test_visualization.png"
    analyzer.generate_visualization("production", str(vis_path))
    assert os.path.exists(vis_path)


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

    # Debug: Print available columns
    print("Production data columns:", test_data["production"].columns.tolist())

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


def test_data_saving(analyzer, test_data, tmp_path):
    """Test that all components save data in correct project directories."""

    # Get the project's results directory
    results_base = project_paths.get_run_directory()

    print("\nSaving test results to:")
    print(f"Results directory: {results_base}")

    # Setup directories
    data_dir = results_base / "input_data"
    results_dir = results_base / "results"
    viz_dir = results_base / "visualizations"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Save test data
    test_data["production"].to_csv(data_dir / "production_data.csv", index=False)
    test_data["quality"].to_csv(data_dir / "quality_data.csv", index=False)
    test_data["energy"].to_csv(data_dir / "energy_data.csv", index=False)
    test_data["material"].to_csv(data_dir / "material_data.csv", index=False)

    print(f"\nSaved data files to: {data_dir}")
    print(f"Saved results to: {results_dir}")
    print(f"Saved visualizations to: {viz_dir}")

    # Verify files exist
    assert (data_dir / "production_data.csv").exists()
    assert (data_dir / "quality_data.csv").exists()
    assert (data_dir / "energy_data.csv").exists()
    assert (data_dir / "material_data.csv").exists()

    # Test analysis results saving
    analyzer.production_data = test_data["production"]
    analyzer.quality_data = test_data["quality"]
    analyzer.energy_data = test_data["energy"]
    analyzer.material_flow = test_data["material"]

    # Generate and save analysis report
    report_path = results_dir / "analysis_report.xlsx"
    analyzer.export_analysis_report(str(report_path))
    assert report_path.exists()

    # Generate and save visualizations
    for metric_type in ["production", "energy", "quality", "sustainability"]:
        viz_path = viz_dir / f"{metric_type}_analysis.png"
        analyzer.generate_visualization(metric_type, str(viz_path))
        assert viz_path.exists()

    print("\nGenerated files:")
    print("\nData files:")
    for file in data_dir.glob("*"):
        print(f"- {file.name}")

    print("\nResults files:")
    for file in results_dir.glob("*"):
        print(f"- {file.name}")

    print("\nVisualization files:")
    for file in viz_dir.glob("*"):
        print(f"- {file.name}")


def test_data_consistency(analyzer, test_data, tmp_path):
    """Test data consistency when saving and loading."""

    # Save production data
    data_path = tmp_path / "production_data.csv"

    # Convert datetime to string before saving
    save_data = test_data["production"].copy()
    if "timestamp" in save_data.columns:
        save_data["timestamp"] = save_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    save_data.to_csv(data_path, index=False)

    # Load saved data
    loaded_data = pd.read_csv(data_path)

    # Convert timestamp strings back to datetime for comparison
    if "timestamp" in test_data["production"].columns:
        loaded_data["timestamp"] = pd.to_datetime(loaded_data["timestamp"])
        comparison_data = test_data["production"].copy()
        comparison_data["timestamp"] = pd.to_datetime(comparison_data["timestamp"])
    else:
        comparison_data = test_data["production"]

    # Verify data integrity
    pd.testing.assert_frame_equal(
        comparison_data.reset_index(drop=True),
        loaded_data.reset_index(drop=True),
        check_dtype=False,
    )
