# tests/unit/manufacturing/optimization/test_optimizer.py
"""Tests for the process optimizer."""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer
from circman5.utils.results_manager import results_manager


def test_optimizer_initialization(optimizer):
    """Test optimizer initialization."""
    assert optimizer is not None
    assert optimizer.model is not None
    assert hasattr(optimizer.model, "is_trained")


def test_process_optimization(optimizer, test_data, test_output_dir):
    """Test process parameter optimization."""
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
        "output_amount": 90.0,  # Add output_amount
    }

    constraints = {
        "input_amount": (90.0, 110.0),
        "energy_used": (140.0, 160.0),
        "cycle_time": (45.0, 55.0),
        "efficiency": (20.0, 22.0),
        "defect_rate": (1.0, 3.0),
        "thickness_uniformity": (94.0, 96.0),
        "output_amount": (80.0, 100.0),  # Add constraint for output_amount
    }

    optimized_params = optimizer.optimize_process_parameters(
        current_params, constraints
    )

    assert isinstance(optimized_params, dict)

    # Verify base parameters are within constraints
    for param, value in optimized_params.items():
        if param in constraints:  # Only check parameters that have constraints
            min_val, max_val = constraints[param]
            assert min_val <= value <= max_val, f"Parameter {param} out of bounds"


def test_optimization_potential(optimizer, test_data, metrics_dir):
    """Test optimization potential analysis."""
    improvements = optimizer.analyze_optimization_potential(
        test_data["production"], test_data["quality"]
    )

    assert isinstance(improvements, dict)
    assert all(isinstance(v, float) for v in improvements.values())

    # Verify analysis file was saved
    analysis_file = metrics_dir / "optimization_potential.json"
    assert analysis_file.exists()

    # Verify analysis content
    with open(analysis_file) as f:
        saved_analysis = json.load(f)
    assert "current_params" in saved_analysis
    assert "optimized_params" in saved_analysis
    assert "improvements" in saved_analysis


def test_optimization_with_invalid_constraints(optimizer, test_data):
    """Test optimization with invalid constraint values."""
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

    invalid_constraints = {
        "input_amount": (110.0, 90.0),  # min > max
    }

    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters(current_params, invalid_constraints)


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

    # Test with empty parameters
    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters({})


def test_optimization_with_missing_parameters(optimizer, test_data):
    """Test optimization with missing parameters."""
    # Train the model first
    optimizer.model.train_optimization_model(
        test_data["production"], test_data["quality"]
    )

    incomplete_params = {
        "input_amount": 100.0,
        # Missing other required parameters
    }

    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters(incomplete_params)


def test_optimization_with_null_values(optimizer, test_data):
    """Test optimization with null/None values in parameters."""
    # Train the model first
    optimizer.model.train_optimization_model(
        test_data["production"], test_data["quality"]
    )

    params_with_null = {
        "input_amount": None,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    with pytest.raises(ValueError):
        optimizer.optimize_process_parameters(params_with_null)
