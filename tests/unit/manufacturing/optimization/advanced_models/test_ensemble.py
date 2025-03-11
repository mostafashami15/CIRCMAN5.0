# tests/unit/manufacturing/optimization/advanced_models/test_ensemble.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.optimization.advanced_models.ensemble import EnsembleModel
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    data_gen = ManufacturingDataGenerator()
    production_data = data_gen.generate_production_data()
    features = production_data[["input_amount", "energy_used", "cycle_time"]].iloc[:50]
    targets = production_data["output_amount"].iloc[:50]
    return features, targets


def test_model_initialization():
    """Test that the ensemble model initializes correctly."""
    model = EnsembleModel()
    assert model is not None
    assert model.is_trained is False
    assert isinstance(model.base_models, list)
    assert len(model.base_models) > 0


def test_model_training(sample_data):
    """Test ensemble model training functionality."""
    X, y = sample_data
    model = EnsembleModel()

    # Train the model
    results = model.train(X, y)

    # Verify training results
    assert model.is_trained is True
    assert "base_models" in results
    assert "meta_model" in results
    assert "training_samples" in results
    assert results["training_samples"] == len(X)
    assert "base_model_metrics" in results
    assert "ensemble_metrics" in results


def test_model_prediction(sample_data):
    """Test ensemble model prediction functionality."""
    X, y = sample_data
    model = EnsembleModel()

    # Train the model
    model.train(X, y)

    # Test prediction on a few samples
    predictions = model.predict(X.iloc[:5])

    # Verify predictions
    assert predictions is not None
    assert len(predictions) == 5
    assert all(isinstance(pred, (int, float)) for pred in predictions.flatten())


def test_model_evaluation(sample_data):
    """Test ensemble model evaluation functionality."""
    X, y = sample_data
    model = EnsembleModel()

    # Train the model
    model.train(X, y)

    # Evaluate the model
    metrics = model.evaluate(X, y)

    # Verify metrics
    assert metrics is not None
    assert "mse" in metrics
    assert "r2" in metrics
    assert "mae" in metrics
    assert all(
        isinstance(metrics[key], (int, float))
        for key in metrics
        if isinstance(metrics[key], (int, float))
    )


def test_model_save_load(sample_data, tmp_path):
    """Test ensemble model saving and loading functionality."""
    X, y = sample_data
    model = EnsembleModel()

    # Train the model
    model.train(X, y)

    # Save the model
    save_path = tmp_path / "ensemble_model.pkl"
    model.save_model(save_path)

    # Verify model was saved
    assert save_path.exists()

    # Load the model into a new instance
    new_model = EnsembleModel()
    new_model.load_model(save_path)

    # Verify loaded model
    assert new_model.is_trained is True
