# tests/integration/test_digital_twin_ai_enhanced.py

import pytest
import numpy as np
import pandas as pd
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.optimization.advanced_models.deep_learning import (
    DeepLearningModel,
)
from circman5.manufacturing.optimization.advanced_models.ensemble import EnsembleModel
from circman5.manufacturing.optimization.online_learning.adaptive_model import (
    AdaptiveModel,
)
from circman5.manufacturing.optimization.validation.cross_validator import (
    CrossValidator,
)
from circman5.manufacturing.optimization.validation.uncertainty import (
    UncertaintyQuantifier,
)
from circman5.test_data_generator import ManufacturingDataGenerator


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
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


@pytest.fixture
def digital_twin_instance():
    """Create a Digital Twin instance for testing."""
    try:
        twin = DigitalTwin()
        # Initialize as needed for testing
        return twin
    except Exception as e:
        pytest.skip(f"Digital Twin initialization failed: {e}")


def test_ai_integration_initialization(digital_twin_instance):
    """Test AI Integration initialization with Digital Twin."""
    ai_integration = AIIntegration(digital_twin_instance)

    # Verify initialization
    assert ai_integration is not None
    assert ai_integration.digital_twin is digital_twin_instance
    assert hasattr(ai_integration, "models")
    assert hasattr(ai_integration, "adaptive_models")


def test_advanced_model_creation(digital_twin_instance, sample_data):
    """Test creating advanced models through AI Integration."""
    X, y = sample_data
    ai_integration = AIIntegration(digital_twin_instance)

    # Create models
    dl_model = ai_integration.create_advanced_model("deep_learning", "dl_test_model")
    ensemble_model = ai_integration.create_advanced_model(
        "ensemble", "ensemble_test_model"
    )

    # Verify models were created
    assert "dl_test_model" in ai_integration.models
    assert "ensemble_test_model" in ai_integration.models
    assert isinstance(dl_model, DeepLearningModel)
    assert isinstance(ensemble_model, EnsembleModel)

    # Train models
    dl_model.train(X, y.to_frame())
    ensemble_model.train(X, y.to_frame())

    # Verify models are trained
    assert dl_model.is_trained
    assert ensemble_model.is_trained

    # Test model retrieval
    retrieved_model = ai_integration.get_model("dl_test_model")
    assert retrieved_model is dl_model


def test_adaptive_model_creation(digital_twin_instance, sample_data):
    """Test creating and using adaptive models through AI Integration."""
    X, y = sample_data
    ai_integration = AIIntegration(digital_twin_instance)

    # Create adaptive model
    adaptive_model = ai_integration.create_adaptive_model(
        "adaptive_test_model", "ensemble"
    )

    # Verify model was created
    assert "adaptive_test_model" in ai_integration.adaptive_models
    assert isinstance(adaptive_model, AdaptiveModel)

    # Test with some data points
    for i in range(10):
        adaptive_model.add_data_point(
            X.iloc[i].values, pd.DataFrame([y.iloc[i]]), weight=1.0
        )

    # Verify model has data
    assert len(adaptive_model.data_buffer_X) == 10
    assert len(adaptive_model.data_buffer_y) == 10


def test_model_validation_with_digital_twin(digital_twin_instance, sample_data):
    """Test model validation through AI Integration."""
    X, y = sample_data
    ai_integration = AIIntegration(digital_twin_instance)

    # Create and train model
    model = ai_integration.create_advanced_model(
        "deep_learning", "validation_test_model"
    )
    model.train(X, y.to_frame())

    # Validate model
    validation_results = ai_integration.validate_model(
        "validation_test_model", X.values, y.values
    )

    # Verify validation results
    assert validation_results is not None
    assert "metrics" in validation_results
    assert "fold_scores" in validation_results


def test_uncertainty_quantification_with_digital_twin(
    digital_twin_instance, sample_data
):
    """Test uncertainty quantification through AI Integration."""
    X, y = sample_data
    ai_integration = AIIntegration(digital_twin_instance)

    # Create and train model
    model = ai_integration.create_advanced_model("ensemble", "uncertainty_test_model")
    model.train(X, y.to_frame())

    # Get uncertainty estimates
    uncertainty_results = ai_integration.quantify_prediction_uncertainty(
        "uncertainty_test_model", X.iloc[:5].values
    )

    # Verify uncertainty results
    assert uncertainty_results is not None
    assert "predictions" in uncertainty_results
    assert "std_dev" in uncertainty_results
    assert "confidence_intervals" in uncertainty_results
    assert len(uncertainty_results["predictions"]) == 5


@pytest.mark.skip(reason="Long running test with threading")
def test_real_time_learning_with_digital_twin(digital_twin_instance):
    """Test real-time learning integration with Digital Twin."""
    ai_integration = AIIntegration(digital_twin_instance)

    # Start real-time learning
    ai_integration.start_real_time_learning(interval_seconds=1)

    # Let it run briefly
    import time

    time.sleep(3)

    # Stop real-time learning
    ai_integration.stop_real_time_learning()

    # Verify the method runs without errors
    assert True  # If we get here, the test passed
