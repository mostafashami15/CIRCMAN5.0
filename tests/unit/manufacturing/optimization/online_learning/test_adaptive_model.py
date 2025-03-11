# tests/unit/manufacturing/optimization/online_learning/test_adaptive_model.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.optimization.online_learning.adaptive_model import (
    AdaptiveModel,
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
    """Test that the adaptive model initializes correctly."""
    model = AdaptiveModel()
    assert model is not None
    assert model.base_model_type == "ensemble"  # Default value
    assert len(model.data_buffer_X) == 0
    assert len(model.data_buffer_y) == 0
    assert len(model.buffer_weights) == 0

    # Test with specified base model type
    model2 = AdaptiveModel(base_model_type="deep_learning")
    assert model2.base_model_type == "deep_learning"


def test_add_data_point(sample_data):
    """Test adding data points to the adaptive model."""
    X, y = sample_data
    model = AdaptiveModel()

    # Add a single data point
    updated = model.add_data_point(X.iloc[0].values, y.iloc[0])

    # Verify buffers
    assert len(model.data_buffer_X) == 1
    assert len(model.data_buffer_y) == 1
    assert len(model.buffer_weights) == 1

    # Add multiple data points
    for i in range(1, 10):
        model.add_data_point(X.iloc[i].values, y.iloc[i])

    # Verify buffers
    assert len(model.data_buffer_X) == 10
    assert len(model.data_buffer_y) == 10
    assert len(model.buffer_weights) == 10

    # Add enough points to trigger model update
    for i in range(10, model.update_frequency):
        model.add_data_point(X.iloc[i % len(X)].values, y.iloc[i % len(y)])

    # Verify model is initialized after update
    assert model.is_initialized


def test_predict_after_update(sample_data):
    """Test prediction after model update."""
    X, y = sample_data
    model = AdaptiveModel()

    # Add enough points to trigger model update
    for i in range(model.update_frequency + 5):
        model.add_data_point(X.iloc[i % len(X)].values, y.iloc[i % len(y)])

    # Model should be initialized and trained now
    assert model.is_initialized

    # Test prediction
    predictions = model.predict(X.iloc[:5])
    assert predictions is not None
    assert len(predictions) == 5


def test_buffer_management(sample_data):
    """Test buffer management with forgetting factor."""
    X, y = sample_data
    model = AdaptiveModel()
    model.window_size = 5  # Small window for testing

    # Add more points than window size
    for i in range(10):
        model.add_data_point(X.iloc[i % len(X)].values, y.iloc[i % len(y)])

    # Verify buffer is limited to window size
    assert len(model.data_buffer_X) == 5
    assert len(model.data_buffer_y) == 5
    assert len(model.buffer_weights) == 5

    # Verify forgetting factor applied (weights decrease over time)
    assert model.buffer_weights[0] < 1.0
