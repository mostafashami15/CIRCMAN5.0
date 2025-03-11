# tests/unit/manufacturing/optimization/advanced_models/test_deep_learning.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.optimization.advanced_models.deep_learning import (
    DeepLearningModel,
)
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
    """Test that the model initializes correctly."""
    model = DeepLearningModel()
    assert model is not None
    assert model.is_trained is False
    assert model.model_type == "lstm"  # Default value from config


def test_model_training(sample_data):
    """Test model training functionality."""
    X, y = sample_data
    model = DeepLearningModel()

    # Train the model
    results = model.train(X, y)

    # Verify training results
    assert model.is_trained is True
    assert "final_loss" in results
    assert "final_accuracy" in results
    assert model.history is not None


def test_model_prediction(sample_data):
    """Test model prediction functionality."""
    X, y = sample_data
    model = DeepLearningModel()

    # Train the model
    model.train(X, y)

    # Test prediction on a few samples
    predictions = model.predict(X.iloc[:5])

    # Verify predictions
    assert predictions is not None
    assert len(predictions) == 5
    assert all(isinstance(pred, (int, float)) for pred in predictions.flatten())


def test_model_evaluation(sample_data):
    """Test model evaluation functionality."""
    X, y = sample_data
    model = DeepLearningModel()

    # Train the model
    model.train(X, y)

    # Evaluate the model
    metrics = model.evaluate(X, y)

    # Verify metrics
    assert metrics is not None
    assert "mse" in metrics
    assert "r2" in metrics
    assert all(isinstance(metrics[key], (int, float)) for key in metrics)


def test_model_save_load(sample_data, tmp_path):
    """Test model saving and loading functionality."""
    X, y = sample_data
    model = DeepLearningModel()

    # Train the model
    model.train(X, y)

    # Save the model
    save_path = tmp_path / "test_model.pkl"
    model.save_model(save_path)

    # Verify model was saved
    assert save_path.exists()

    # Load the model into a new instance
    new_model = DeepLearningModel()
    new_model.load_model(save_path)

    # Verify loaded model
    assert new_model.is_trained is True
