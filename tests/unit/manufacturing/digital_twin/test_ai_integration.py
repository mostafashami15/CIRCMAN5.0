# tests/unit/manufacturing/digital_twin/test_ai_integration.py

import pytest
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.optimization.model import ManufacturingModel
from circman5.manufacturing.optimization.optimizer import ProcessOptimizer
from circman5.utils.results_manager import results_manager


@pytest.fixture
def digital_twin_fixture():
    """Create a mock digital twin for testing."""
    dt = MagicMock()

    # Mock current state
    current_state = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_status": "running",
        "production_line": {
            "status": "running",
            "temperature": 25.0,
            "energy_consumption": 50.0,
            "production_rate": 100.0,
            "cycle_time": 30.0,
            "efficiency": 0.92,
            "defect_rate": 0.03,
        },
        "materials": {
            "silicon_wafer": {"inventory": 500, "quality": 0.95},
            "solar_glass": {"inventory": 300, "quality": 0.98},
        },
        "environment": {"temperature": 22.0, "humidity": 45.0},
    }

    # Set up mock methods
    dt.get_current_state.return_value = current_state
    dt.get_state_history.return_value = [current_state]
    dt.simulate.return_value = [current_state]

    return dt


@pytest.fixture
def mock_model():
    """Create a mock AI model for testing."""
    model = MagicMock()
    model.is_trained = True
    model.predict_batch_outcomes.return_value = {
        "predicted_output": 110.0,
        "predicted_quality": 98.0,
        "confidence_score": 0.95,
        "uncertainty": 0.05,
    }
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = MagicMock()
    optimizer.optimize_process_parameters.return_value = {
        "input_amount": 120.0,
        "energy_used": 45.0,
        "cycle_time": 28.0,
        "efficiency": 0.94,
        "defect_rate": 0.02,
        "thickness_uniformity": 97.0,
        "output_amount": 110.0,
    }
    return optimizer


@pytest.fixture
def ai_integration(digital_twin_fixture, mock_model, mock_optimizer):
    """Create an AIIntegration instance with mock components."""
    integration = AIIntegration(
        digital_twin=digital_twin_fixture, model=mock_model, optimizer=mock_optimizer
    )
    return integration


def test_init(digital_twin_fixture):
    """Test AIIntegration initialization."""
    # Test with all components provided
    model = MagicMock()
    optimizer = MagicMock()
    integration = AIIntegration(
        digital_twin=digital_twin_fixture, model=model, optimizer=optimizer
    )

    assert integration.digital_twin == digital_twin_fixture
    assert integration.model == model
    assert integration.optimizer == optimizer

    # Test with default model and optimizer
    with patch(
        "circman5.manufacturing.digital_twin.integration.ai_integration.ManufacturingModel"
    ):
        with patch(
            "circman5.manufacturing.digital_twin.integration.ai_integration.ProcessOptimizer"
        ):
            integration = AIIntegration(digital_twin=digital_twin_fixture)
            assert integration.digital_twin == digital_twin_fixture
            assert integration.optimization_history == []


def test_extract_parameters_from_state(ai_integration):
    """Test extracting parameters from state."""
    # Extract from provided state
    test_state = {
        "production_line": {
            "production_rate": 90.0,
            "energy_consumption": 40.0,
            "temperature": 24.0,
            "cycle_time": 35.0,
        },
        "materials": {
            "material_1": {"inventory": 200, "quality": 0.92},
            "material_2": {"inventory": 300, "quality": 0.94},
        },
    }

    params = ai_integration.extract_parameters_from_state(test_state)

    assert params["output_amount"] == 90.0
    assert params["energy_used"] == 40.0
    assert params["cycle_time"] == 35.0
    assert params["input_amount"] == 500.0  # 200 + 300
    assert params["efficiency"] == pytest.approx(0.93)  # (0.92 + 0.94) / 2

    # Test extraction from current state
    params = ai_integration.extract_parameters_from_state()
    assert isinstance(params, dict)
    assert "output_amount" in params


def test_predict_outcomes(ai_integration):
    """Test predicting outcomes."""
    # Test with provided parameters
    test_params = {
        "input_amount": 100.0,
        "energy_used": 50.0,
        "cycle_time": 30.0,
        "efficiency": 0.9,
        "defect_rate": 0.05,
        "thickness_uniformity": 95.0,
    }

    prediction = ai_integration.predict_outcomes(test_params)

    assert prediction["predicted_output"] == 110.0
    assert prediction["confidence_score"] == 0.95

    # Test using current state
    prediction = ai_integration.predict_outcomes()
    assert prediction["predicted_output"] == 110.0


def test_optimize_parameters(ai_integration):
    """Test parameter optimization."""
    # Test with provided parameters
    test_params = {
        "input_amount": 100.0,
        "energy_used": 50.0,
        "cycle_time": 30.0,
        "efficiency": 0.9,
        "defect_rate": 0.05,
        "thickness_uniformity": 95.0,
    }

    optimized = ai_integration.optimize_parameters(test_params)

    assert optimized["input_amount"] == 120.0
    assert optimized["energy_used"] == 45.0

    # Verify optimization history is updated
    assert len(ai_integration.optimization_history) == 1
    assert "original_params" in ai_integration.optimization_history[0]
    assert "optimized_params" in ai_integration.optimization_history[0]

    # Test using current state
    optimized = ai_integration.optimize_parameters()
    assert optimized["input_amount"] == 120.0


def test_apply_optimized_parameters(ai_integration):
    """Test applying optimized parameters."""
    test_params = {
        "input_amount": 120.0,
        "energy_used": 45.0,
        "output_amount": 110.0,
        "cycle_time": 28.0,
        "efficiency": 0.94,
        "defect_rate": 0.02,
        "thickness_uniformity": 97.0,
    }

    result = ai_integration.apply_optimized_parameters(test_params)
    assert result is True

    # Verify digital twin.simulate was called
    ai_integration.digital_twin.simulate.assert_called_once()


def test_convert_parameters_to_state(ai_integration):
    """Test converting optimization parameters to state format."""
    test_params = {
        "output_amount": 110.0,
        "energy_used": 45.0,
        "cycle_time": 28.0,
        "temperature": 24.0,
    }

    state_updates = ai_integration._convert_parameters_to_state(test_params)

    assert "production_line" in state_updates
    assert state_updates["production_line"]["production_rate"] == 110.0
    assert state_updates["production_line"]["energy_consumption"] == 45.0
    assert state_updates["production_line"]["cycle_time"] == 28.0
    assert state_updates["production_line"]["status"] == "running"
    assert state_updates["system_status"] == "optimized"


def test_generate_optimization_report(ai_integration):
    """Test generating optimization report."""
    # Add mock history
    ai_integration.optimization_history = [
        {
            "timestamp": "2025-01-01T12:00:00",
            "original_params": {"input_amount": 100.0, "energy_used": 50.0},
            "optimized_params": {"input_amount": 110.0, "energy_used": 45.0},
            "improvements": {"input_amount": 10.0, "energy_used": -10.0},
        },
        {
            "timestamp": "2025-01-02T12:00:00",
            "original_params": {"input_amount": 110.0, "energy_used": 45.0},
            "optimized_params": {"input_amount": 120.0, "energy_used": 40.0},
            "improvements": {"input_amount": 9.09, "energy_used": -11.11},
        },
    ]

    with patch("pathlib.Path.unlink"):
        with patch("builtins.open", MagicMock()):
            with patch.object(results_manager, "save_file"):
                report = ai_integration.generate_optimization_report()

    assert report["total_optimizations"] == 2
    assert "average_improvement" in report
    assert "parameter_trends" in report
    assert "current_state" in report


def test_train_model_from_digital_twin(ai_integration):
    """Test training model from digital twin data."""
    # Mock the model to return metrics
    ai_integration.model.train_optimization_model.return_value = {
        "r2": 0.95,
        "mse": 0.1,
        "rmse": 0.316,
        "mae": 0.25,
        "cv_r2_mean": 0.94,
        "cv_r2_std": 0.02,
        "mean_uncertainty": 0.1,
        "feature_importance": {"input_amount": 0.3, "energy_used": 0.2},
    }

    # Create enough state history
    history = []
    for i in range(10):
        state = {
            "timestamp": (
                datetime.datetime.now() - datetime.timedelta(days=i)
            ).isoformat(),
            "production_line": {
                "production_rate": 100.0 + i,
                "energy_consumption": 50.0 - i * 0.5,
                "efficiency": 0.9 + i * 0.01,
                "defect_rate": 0.05 - i * 0.005,
            },
            "materials": {
                f"material_{i}": {"inventory": 100 + i * 10, "quality": 0.9 + i * 0.01}
            },
        }
        history.append(state)

    ai_integration.digital_twin.get_state_history.return_value = history

    result = ai_integration.train_model_from_digital_twin()
    assert result is True

    # Test with insufficient data
    ai_integration.digital_twin.get_state_history.return_value = history[:2]
    result = ai_integration.train_model_from_digital_twin()
    assert result is False
