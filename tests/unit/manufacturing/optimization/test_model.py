# tests/unit/manufacturing/optimization/test_model.py

"""Tests for the manufacturing optimization model."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from circman5.manufacturing.optimization.model import ManufacturingModel
from circman5.manufacturing.optimization.types import MetricsDict, PredictionDict
from circman5.test_data_generator import ManufacturingDataGenerator
from circman5.utils.results_manager import results_manager


def test_model_initialization(model):
    """Test model initialization."""
    assert model is not None
    assert not model.is_trained
    assert model.model is not None
    assert model.feature_scaler is not None
    assert model.target_scaler is not None


def test_model_training(model, test_data, metrics_dir):
    """Test model training and metrics calculation."""
    metrics = model.train_optimization_model(
        test_data["production"], test_data["quality"]
    )

    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] >= 0
    assert model.is_trained

    # Verify metrics file was saved
    metrics_file = metrics_dir / "training_metrics.json"
    assert metrics_file.exists()


def test_prediction(model, test_data, test_output_dir):
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
        "output_amount": 90.0,
    }

    predictions = model.predict_batch_outcomes(test_params)
    assert isinstance(predictions, dict)
    assert "predicted_output" in predictions
    assert "predicted_quality" in predictions
    assert "confidence_score" in predictions

    # Verify predictions file was saved
    predictions_file = test_output_dir / "latest_predictions.json"
    assert predictions_file.exists()


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
        "output_amount": 90.0,  # Add estimated output amount
    }


def test_model_saving(model, test_data, test_output_dir):
    """Test model saving functionality."""
    # Train the model first
    model.train_optimization_model(test_data["production"], test_data["quality"])

    # Save the model
    model.save_model("test_model")

    # Verify saved files exist in the correct location
    saved_files = [
        test_output_dir / "test_model_model.joblib",
        test_output_dir / "test_model_scalers.joblib",
        test_output_dir / "test_model_config.json",
    ]

    for file_path in saved_files:
        assert file_path.exists(), f"Expected file {file_path} not found"


def test_error_handling(model):
    """Test error handling in model."""
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
        "output_amount": 95.0,  # Added for calculated features
    }

    # Test prediction without training
    with pytest.raises(ValueError):
        model.predict_batch_outcomes(test_params)

    # Test saving without training
    with pytest.raises(ValueError):
        model.save_model("test_model")


def test_empty_data_handling(model):
    """Test handling of empty input data."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        model.train_optimization_model(empty_df, empty_df)


def test_model_training_with_invalid_data(model):
    """Test model training with invalid or incomplete data."""
    invalid_production_data = pd.DataFrame(
        {"input_amount": [100.0], "batch_id": ["BATCH_001"]}
    )

    invalid_quality_data = pd.DataFrame(
        {"efficiency": [95.0], "batch_id": ["BATCH_001"]}
    )

    with pytest.raises(Exception):  # Should raise some form of error
        model.train_optimization_model(invalid_production_data, invalid_quality_data)


def test_model_overfitting():
    """Test that model does not overfit with new validation approach."""
    model = ManufacturingModel()

    # Generate synthetic data
    data_generator = ManufacturingDataGenerator()
    production_data = data_generator.generate_production_data()
    quality_data = data_generator.generate_quality_data()

    # Train model
    metrics = model.train_optimization_model(production_data, quality_data)

    # Assert cross-validation stability
    assert metrics["cv_r2_std"] < 0.1, "Cross-validation scores show high variance"
    assert metrics["cv_r2_mean"] > 0.7, "Model shows poor generalization"
    assert (
        abs(metrics["r2"] - metrics["cv_r2_mean"]) < 0.15
    ), "Model shows signs of overfitting"


def test_uncertainty_quantification():
    """Test uncertainty estimation capabilities."""
    model = ManufacturingModel()

    # Generate synthetic data and train model
    data_generator = ManufacturingDataGenerator()
    production_data = data_generator.generate_production_data()
    quality_data = data_generator.generate_quality_data()
    model.train_optimization_model(production_data, quality_data)

    # Define test parameters
    test_params = {
        "input_amount": 100.0,
        "energy_used": 150.0,
        "cycle_time": 50.0,
        "efficiency": 21.0,
        "defect_rate": 2.0,
        "thickness_uniformity": 95.0,
        "output_amount": 90.0,
    }

    predictions = model.predict_batch_outcomes(test_params)
    assert isinstance(predictions, dict)
    assert "uncertainty" in predictions
    assert isinstance(predictions["uncertainty"], float)
    assert 0 <= predictions["uncertainty"] <= 10.0  # Reasonable range for uncertainty
