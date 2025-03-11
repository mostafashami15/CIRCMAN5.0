# tests/unit/manufacturing/optimization/validation/test_uncertainty.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.optimization.validation.uncertainty import (
    UncertaintyQuantifier,
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


def test_quantifier_initialization():
    """Test that the uncertainty quantifier initializes correctly."""
    quantifier = UncertaintyQuantifier()
    assert quantifier is not None
    assert quantifier.method in ["monte_carlo_dropout", "bootstrap", "ensemble"]
    assert quantifier.samples > 0
    assert 0 < quantifier.confidence_level < 1


def test_uncertainty_quantification(sample_data_and_model):
    """Test uncertainty quantification functionality."""
    X, y, model = sample_data_and_model
    quantifier = UncertaintyQuantifier()

    # Get uncertainty estimates
    results = quantifier.quantify_uncertainty(model, X.iloc[:5])

    # Verify results structure
    assert results is not None
    assert "predictions" in results
    assert "std_dev" in results
    assert "confidence_intervals" in results
    assert "prediction_intervals" in results

    # Verify dimensions
    assert len(results["predictions"]) == 5
    assert len(results["std_dev"]) == 5
    assert results["confidence_intervals"].shape == (5, 2)
    assert results["prediction_intervals"].shape == (5, 2)

    # Verify interval relationships
    for i in range(5):
        assert (
            results["confidence_intervals"][i, 0]
            <= results["predictions"][i]
            <= results["confidence_intervals"][i, 1]
        )
        assert (
            results["prediction_intervals"][i, 0]
            <= results["confidence_intervals"][i, 0]
        )
        assert (
            results["confidence_intervals"][i, 1]
            <= results["prediction_intervals"][i, 1]
        )


def test_calibration(sample_data_and_model):
    """Test uncertainty calibration functionality."""
    X, y, model = sample_data_and_model
    quantifier = UncertaintyQuantifier()

    # Calibrate the uncertainty estimator
    calibration_params = quantifier.calibrate(model, X, y)

    # Verify calibration
    assert quantifier.is_calibrated is True
    assert calibration_params is not None
    assert "temperature" in calibration_params

    # Generate uncertainty with calibration
    results = quantifier.quantify_uncertainty(model, X.iloc[:5])

    # Basic verification of calibrated results
    assert results["predictions"] is not None
    assert results["std_dev"] is not None
