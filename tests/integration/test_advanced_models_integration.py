# tests/integration/test_advanced_models_integration.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.optimization.advanced_models.deep_learning import (
    DeepLearningModel,
)
from circman5.manufacturing.optimization.advanced_models.ensemble import EnsembleModel
from circman5.manufacturing.optimization.validation.cross_validator import (
    CrossValidator,
)
from circman5.manufacturing.optimization.validation.uncertainty import (
    UncertaintyQuantifier,
)
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def sample_data():
    """Generate more substantial sample data for integration testing."""
    data_gen = ManufacturingDataGenerator()
    production_data = data_gen.generate_production_data()
    quality_data = data_gen.generate_quality_data()

    # Merge datasets
    merged_data = pd.merge(production_data, quality_data, on="batch_id")

    features = merged_data[
        [
            "input_amount",
            "energy_used",
            "cycle_time",
            "efficiency",
            "defect_rate",
            "thickness_uniformity",
        ]
    ]
    targets = merged_data["output_amount"]

    return features, targets


def test_model_validation_integration(sample_data):
    """Test end-to-end model training, validation, and uncertainty."""
    X, y = sample_data

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Create and train models
    dl_model = DeepLearningModel()
    dl_model.train(X_train, y_train)

    ensemble_model = EnsembleModel()
    ensemble_model.train(X_train, y_train)

    # Create validator
    validator = CrossValidator()

    # Validate both models
    dl_results = validator.validate(dl_model, X_train, y_train)
    ensemble_results = validator.validate(ensemble_model, X_train, y_train)

    # Verify validation results
    assert dl_results is not None
    assert ensemble_results is not None

    # Create uncertainty quantifier
    uq = UncertaintyQuantifier()

    # Quantify uncertainty for test data
    dl_uncertainty = uq.quantify_uncertainty(dl_model, X_test)
    ensemble_uncertainty = uq.quantify_uncertainty(ensemble_model, X_test)

    # Verify uncertainty results
    assert dl_uncertainty is not None
    assert ensemble_uncertainty is not None

    # Compare model performances
    dl_metrics = dl_model.evaluate(X_test, y_test)
    ensemble_metrics = ensemble_model.evaluate(X_test, y_test)

    # Log results (would be more useful in a real integration test)
    print(f"Deep Learning R²: {dl_metrics.get('r2', 'N/A')}")
    print(f"Ensemble R²: {ensemble_metrics.get('r2', 'N/A')}")
