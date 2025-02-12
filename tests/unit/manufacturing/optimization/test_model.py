# tests/unit/manufacturing/optimization/test_model.py

"""Tests for the manufacturing optimization model."""

import pytest
import pandas as pd
import numpy as np
from circman5.manufacturing.optimization.model import ManufacturingModel
from circman5.manufacturing.optimization.types import MetricsDict, PredictionDict


@pytest.fixture
def test_data():
    """Generate test data for model validation."""
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


@pytest.fixture
def model():
    """Create a test model instance."""
    return ManufacturingModel()


def test_model_initialization(model):
    """Test model initialization."""
    assert model is not None
    assert not model.is_trained
    assert model.model is not None
    assert model.feature_scaler is not None
    assert model.target_scaler is not None


def test_model_training(model, test_data):
    """Test model training and metrics calculation."""
    metrics = model.train_optimization_model(
        test_data["production"], test_data["quality"]
    )

    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] >= 0
    assert model.is_trained


def test_prediction(model, test_data):
    """Test predictions with trained model."""
    # Train the model first
    model.train_optimization_model(test_data["production"], test_data["quality"])

    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    predictions = model.predict_batch_outcomes(test_params)
    assert isinstance(predictions, dict)
    assert "predicted_output" in predictions
    assert "predicted_quality" in predictions


def test_error_handling(model):
    """Test error handling in model."""
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
    }

    # Test prediction without training
    with pytest.raises(ValueError):
        model.predict_batch_outcomes(test_params)
