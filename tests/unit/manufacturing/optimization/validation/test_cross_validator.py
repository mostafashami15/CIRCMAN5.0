# tests/unit/manufacturing/optimization/validation/test_cross_validator.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.optimization.validation.cross_validator import (
    CrossValidator,
)
from circman5.manufacturing.optimization.advanced_models.deep_learning import (
    DeepLearningModel,
)
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def sample_data_and_model():
    """Generate sample data and model."""
    data_gen = ManufacturingDataGenerator()
    production_data = data_gen.generate_production_data()
    features = production_data[["input_amount", "energy_used", "cycle_time"]].iloc[:50]
    targets = production_data["output_amount"].iloc[:50]

    # Create and train model
    model = DeepLearningModel()
    model.train(features, targets.to_frame())

    return features, targets, model


def test_validator_initialization():
    """Test that the validator initializes correctly."""
    validator = CrossValidator()
    assert validator is not None
    assert validator.method in ["stratified_kfold", "kfold", "group_kfold"]
    assert validator.n_splits > 0
    assert len(validator.metrics) > 0


def test_validation(sample_data_and_model):
    """Test validation functionality."""
    X, y, model = sample_data_and_model
    validator = CrossValidator()

    # Run validation
    results = validator.validate(model, X, y)

    # Verify results structure
    assert results is not None
    assert "method" in results
    assert "n_splits" in results
    assert "metrics" in results
    assert "fold_scores" in results
    assert "timestamp" in results

    # Verify metrics
    for metric in validator.metrics:
        assert metric in results["metrics"]
        assert "mean" in results["metrics"][metric]
        assert "std" in results["metrics"][metric]
        assert "min" in results["metrics"][metric]
        assert "max" in results["metrics"][metric]
        assert "values" in results["metrics"][metric]

    # Verify fold scores
    assert len(results["fold_scores"]) == validator.n_splits
    for fold_result in results["fold_scores"]:
        assert "fold" in fold_result
        assert "train_size" in fold_result
        assert "test_size" in fold_result
        assert "scores" in fold_result
        assert all(metric in fold_result["scores"] for metric in validator.metrics)
