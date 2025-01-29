"""
Test suite for AI-driven optimization engine.
"""

import pytest
import numpy as np
import pandas as pd
from circman5.ai.optimization import ManufacturingOptimizer


@pytest.fixture
def sample_manufacturing_data():
    """Create sample manufacturing data for testing."""
    np.random.seed(42)
    n_samples = 100

    production_data = pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(n_samples)],
            "input_amount": np.random.uniform(80, 120, n_samples),
            "energy_used": np.random.uniform(140, 160, n_samples),
            "cycle_time": np.random.uniform(45, 55, n_samples),
            "output_amount": np.random.uniform(75, 110, n_samples),
        }
    )

    quality_data = pd.DataFrame(
        {
            "batch_id": [f"BATCH_{i:03d}" for i in range(n_samples)],
            "efficiency": np.random.uniform(20, 22, n_samples),
            "defect_rate": np.random.uniform(1, 3, n_samples),
            "thickness_uniformity": np.random.uniform(94, 96, n_samples),
        }
    )

    return production_data, quality_data


@pytest.fixture
def optimizer():
    """Create optimizer instance for testing."""
    return ManufacturingOptimizer()


def test_data_preparation(optimizer, sample_manufacturing_data):
    """Test manufacturing data preparation."""
    production_data, quality_data = sample_manufacturing_data

    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        production_data, quality_data
    )

    assert isinstance(X_scaled, np.ndarray)
    assert isinstance(y_scaled, np.ndarray)
    assert X_scaled.shape[0] == y_scaled.shape[0]
    assert X_scaled.shape[1] == 6  # Number of features


def test_model_training(optimizer, sample_manufacturing_data):
    """Test optimization model training."""
    production_data, quality_data = sample_manufacturing_data

    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        production_data, quality_data
    )

    metrics = optimizer.train_optimization_models(X_scaled, y_scaled)

    assert "mse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] >= 0  # R² should be non-negative
    assert optimizer.is_trained


def test_process_optimization(optimizer, sample_manufacturing_data):
    """Test manufacturing process optimization."""
    production_data, quality_data = sample_manufacturing_data

    # Prepare and train model
    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        production_data, quality_data
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

    constraints = {
        "input_amount": (80, 120),
        "energy_used": (140, 160),
        "cycle_time": (45, 55),
        "efficiency": (20, 22),
        "defect_rate": (1, 3),
        "thickness_uniformity": (94, 96),
    }

    optimized = optimizer.optimize_process_parameters(current_params, constraints)

    assert isinstance(optimized, dict)
    assert set(optimized.keys()) == set(current_params.keys())

    # Check constraints are respected
    for param, value in optimized.items():
        min_val, max_val = constraints[param]
        assert min_val <= value <= max_val


def test_outcome_prediction(optimizer, sample_manufacturing_data):
    """Test manufacturing outcome prediction."""
    production_data, quality_data = sample_manufacturing_data

    # Prepare and train model
    X_scaled, y_scaled = optimizer.prepare_manufacturing_data(
        production_data, quality_data
    )
    optimizer.train_optimization_models(X_scaled, y_scaled)

    # Test prediction
    process_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    predictions = optimizer.predict_manufacturing_outcomes(process_params)

    assert "predicted_output" in predictions
    assert "predicted_quality" in predictions
    assert isinstance(predictions["predicted_output"], float)
    assert isinstance(predictions["predicted_quality"], float)


def test_error_handling(optimizer):
    """Test error handling in optimization engine."""
    # Test prediction without training
    process_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    with pytest.raises(ValueError):
        optimizer.predict_manufacturing_outcomes(process_params)

    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters(process_params)
