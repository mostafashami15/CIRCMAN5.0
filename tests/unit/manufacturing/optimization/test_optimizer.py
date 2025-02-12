# tests/manufacturing/optimization/test_optimizer.py
"""Tests for the process optimizer."""

import pytest
import pandas as pd
import numpy as np
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer


@pytest.fixture
def optimizer():
    """Create a test optimizer instance."""
    return ProcessOptimizer()


@pytest.fixture
def test_data():
    """Generate test data for optimization."""
    np.random.seed(42)
    production_data = pd.DataFrame(
        {
            "input_amount": np.random.uniform(90, 110, 100),
            "energy_used": np.random.uniform(140, 160, 100),
            "cycle_time": np.random.uniform(45, 55, 100),
            "output_amount": np.random.uniform(85, 105, 100),
            "batch_id": [f"BATCH_{i:03d}" for i in range(100)],
        }
    )

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
    assert optimizer.model is not None


def test_process_optimization(optimizer, test_data):
    """Test process parameter optimization."""
    # Train the model first
    optimizer.model.train_optimization_model(
        test_data["production"], test_data["quality"]
    )

    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

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

    assert isinstance(optimized_params, dict)
    # Verify optimized parameters are within constraints
    for param, value in optimized_params.items():
        min_val, max_val = constraints[param]
        assert min_val <= value <= max_val


def test_optimization_potential(optimizer, test_data):
    """Test optimization potential analysis."""
    improvements = optimizer.analyze_optimization_potential(
        test_data["production"], test_data["quality"]
    )

    assert isinstance(improvements, dict)
    assert all(isinstance(v, float) for v in improvements.values())


def test_error_handling(optimizer):
    """Test error handling in optimizer."""
    current_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Test optimization without training
    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters(current_params)
