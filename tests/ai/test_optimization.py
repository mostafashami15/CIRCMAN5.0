"""Tests for the AI optimization module."""

import pytest
import pandas as pd
import numpy as np

from circman5.ai.optimization_prediction import ManufacturingOptimizer
from circman5.ai.optimization_types import PredictionDict, MetricsDict
from circman5.config.project_paths import project_paths


@pytest.fixture
def optimizer():
    """Create a test optimizer instance."""
    return ManufacturingOptimizer()


@pytest.fixture
def test_data():
    """Generate test data for optimization."""
    # Create sample production data
    np.random.seed(42)  # Added for reproducibility
    production_data = pd.DataFrame(
        {
            "input_amount": np.random.uniform(90, 110, 100),
            "energy_used": np.random.uniform(140, 160, 100),
            "cycle_time": np.random.uniform(45, 55, 100),
            "output_amount": np.random.uniform(85, 105, 100),
            "batch_id": [f"BATCH_{i:03d}" for i in range(100)],
        }
    )

    # Create sample quality data
    quality_data = pd.DataFrame(
        {
            "efficiency": np.random.uniform(20, 22, 100),
            "defect_rate": np.random.uniform(1, 3, 100),
            "thickness_uniformity": np.random.uniform(94, 96, 100),
            "batch_id": [f"BATCH_{i:03d}" for i in range(100)],
        }
    )

    return {"production": production_data, "quality": quality_data}


def test_optimizer_initialization(optimizer):
    """Test optimizer initialization."""
    assert optimizer is not None
    assert hasattr(optimizer, "efficiency_model")
    assert hasattr(optimizer, "quality_model")


def test_data_preparation(optimizer, test_data):
    """Test data preparation functionality."""
    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        test_data["production"], test_data["quality"]
    )

    assert X_scaled is not None
    assert y_scaled is not None
    assert X_scaled.shape[0] == y_scaled.shape[0]
    assert X_scaled.shape[1] == 6  # Number of features


def test_model_training(optimizer, test_data):
    """Test model training and metrics calculation."""
    # Prepare data
    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        test_data["production"], test_data["quality"]
    )

    # Train models
    metrics = optimizer.train_optimization_models(X_scaled, y_scaled)

    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] >= 0  # R² should be non-negative


def test_process_optimization(optimizer, test_data):
    """Test process parameter optimization."""
    np.random.seed(42)  # Add seed for reproducibility

    # Prepare and train
    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        test_data["production"], test_data["quality"]
    )
    optimizer.train_optimization_models(X_scaled, y_scaled)

    # Test optimization
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Define constraints
    constraints = {
        "input_amount": (90.0, 110.0),
        "energy_used": (140.0, 160.0),
        "cycle_time": (45.0, 55.0),
        "efficiency": (20.0, 22.0),
        "defect_rate": (1.0, 3.0),
        "thickness_uniformity": (94.0, 96.0),
    }

    optimized_params = optimizer.optimize_process_parameters(
        current_params, constraints
    )

    # Verify optimized parameters
    for param, value in optimized_params.items():
        min_val, max_val = constraints[param]
        try:
            assert (
                min_val <= float(value) <= max_val
            ), f"Parameter {param} = {value} outside range [{min_val}, {max_val}]"
        except AssertionError as e:
            print(
                f"Optimization failed for {param}: {value} not in [{min_val}, {max_val}]"
            )
            raise


def test_prediction_generation(optimizer, test_data):
    """Test manufacturing outcome predictions."""
    # Prepare and train
    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        test_data["production"], test_data["quality"]
    )
    optimizer.train_optimization_models(X_scaled, y_scaled)

    # Test predictions
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    predictions = optimizer.predict_manufacturing_outcomes(test_params)

    assert isinstance(predictions, dict)
    assert "predicted_output" in predictions
    assert "predicted_quality" in predictions
    assert isinstance(predictions["predicted_output"], float)
    assert isinstance(predictions["predicted_quality"], float)

    # Check predictions are saved
    results_dir = optimizer.results_dir
    assert (results_dir / "predictions.json").exists()


def test_error_handling(optimizer):
    """Test error handling in optimization engine."""
    # Test prediction without training
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    with pytest.raises(ValueError):
        optimizer.predict_manufacturing_outcomes(test_params)

    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters(test_params)
