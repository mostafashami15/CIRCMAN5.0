"""Integration tests for manufacturing optimization system."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from circman5.solitek_manufacturing import SoliTekManufacturingAnalysis
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.config.project_paths import project_paths


@pytest.fixture(scope="module")
def test_analyzer():
    """Create manufacturing analyzer instance."""
    return SoliTekManufacturingAnalysis()


@pytest.fixture(scope="module")
def test_data():
    """Generate test manufacturing data."""
    generator = ManufacturingDataGenerator(start_date="2024-01-01", days=5)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
    }


@pytest.fixture(scope="module")
def test_run_dir():
    """Create and maintain test run directory."""
    run_dir = project_paths.get_run_directory()

    # Create required subdirectories
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    return run_dir


def test_data_loading(test_analyzer, test_data, test_run_dir):
    """Test manufacturing data loading and validation."""
    # Load test data
    test_analyzer.production_data = test_data["production"]
    test_analyzer.quality_data = test_data["quality"]
    test_analyzer.energy_data = test_data["energy"]
    test_analyzer.material_flow = test_data["material"]

    # Verify data is loaded correctly
    assert not test_analyzer.production_data.empty, "Production data not loaded"
    assert not test_analyzer.quality_data.empty, "Quality data not loaded"
    assert not test_analyzer.energy_data.empty, "Energy data not loaded"
    assert not test_analyzer.material_flow.empty, "Material flow data not loaded"


def test_optimization_training(test_analyzer, test_run_dir):
    """Test AI optimization model training."""
    # Train optimization model
    metrics = test_analyzer.train_optimization_model()

    # Verify training metrics
    assert "mse" in metrics, "Missing MSE metric"
    assert "r2" in metrics, "Missing R² metric"
    assert metrics["r2"] > 0, "Invalid R² score"

    # Verify model is trained
    assert test_analyzer.is_optimizer_trained, "Model not marked as trained"


def test_process_optimization(test_analyzer, test_run_dir):
    """Test manufacturing process optimization."""
    # Define current parameters
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Add constraints
    constraints = {
        "input_amount": (80.0, 120.0),
        "energy_used": (100.0, 200.0),
        "cycle_time": (40.0, 60.0),
    }

    # Get optimized parameters
    optimized_params = test_analyzer.optimize_process_parameters(
        current_params, constraints=constraints
    )

    # Verify optimization results
    assert isinstance(optimized_params, dict), "Invalid optimization result type"
    assert all(
        key in optimized_params for key in current_params.keys()
    ), "Missing parameters"

    # Verify constraints are respected
    for param, (min_val, max_val) in constraints.items():
        assert (
            min_val <= optimized_params[param] <= max_val
        ), f"Constraint violated for {param}"


def test_performance_prediction(test_analyzer, test_run_dir):
    """Test manufacturing performance prediction."""
    # Define test parameters
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Get predictions
    predictions = test_analyzer.predict_batch_outcomes(test_params)

    # Verify prediction structure
    assert "predicted_output" in predictions, "Missing output prediction"
    assert "predicted_quality" in predictions, "Missing quality prediction"

    # Verify prediction values
    assert predictions["predicted_output"] > 0, "Invalid output prediction"
    assert 0 <= predictions["predicted_quality"] <= 100, "Invalid quality prediction"


def test_reporting_and_visualization(test_analyzer, test_run_dir):
    """Test report generation and visualization creation."""
    # Generate comprehensive report
    report_path = test_run_dir / "reports" / "manufacturing_analysis.xlsx"
    test_analyzer.generate_comprehensive_report(str(report_path))
    assert report_path.exists(), "Report not generated"

    # Generate visualizations
    for metric_type in ["production", "energy", "quality", "sustainability"]:
        viz_path = test_run_dir / "visualizations" / f"{metric_type}_analysis.png"
        test_analyzer.generate_visualization(metric_type, str(viz_path))
        assert viz_path.exists(), f"{metric_type} visualization not generated"


def test_optimization_analysis(test_analyzer, test_run_dir):
    """Test optimization potential analysis."""
    # Analyze optimization potential
    improvements = test_analyzer.analyze_optimization_potential()

    # Verify improvement metrics
    assert isinstance(improvements, dict), "Invalid improvements type"
    assert len(improvements) > 0, "No improvements found"

    # Key metrics should be present
    expected_metrics = ["input_amount", "energy_used", "cycle_time"]
    for metric in expected_metrics:
        assert metric in improvements, f"Missing {metric} improvement"

    # All improvements should be reasonable
    assert all(
        -100 <= v <= 100 for v in improvements.values()
    ), "Invalid improvement values"
