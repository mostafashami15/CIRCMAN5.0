"""Test suite for AI optimization engine."""

import pytest
import pandas as pd
from circman5.ai.optimization import AIOptimizationEngine
from circman5.test_data_generator import TestDataGenerator


@pytest.fixture
def test_data():
    """Generate test data for AI optimization tests."""
    generator = TestDataGenerator(start_date="2024-01-01", days=30)
    return {
        "production": generator.generate_production_data(),
        "quality": generator.generate_quality_data(),
        "energy": generator.generate_energy_data(),
        "material": generator.generate_material_flow_data(),
    }


@pytest.fixture
def ai_engine():
    """Create AI optimization engine instance."""
    return AIOptimizationEngine()


def test_process_optimization(ai_engine, test_data):
    """Test process parameter optimization."""
    result = ai_engine.optimize_process_parameters(
        test_data["production"], test_data["quality"]
    )

    assert "cycle_time" in result
    assert "temperature" in result
    assert "pressure" in result
    assert "confidence_score" in result

    assert 0 <= result["confidence_score"] <= 1
    assert result["cycle_time"] > 0
    assert isinstance(result["temperature"], float)


def test_maintenance_prediction(ai_engine, test_data):
    """Test predictive maintenance capabilities."""
    result = ai_engine.predict_maintenance_needs(
        test_data["production"], test_data["energy"]
    )

    assert "next_maintenance" in result
    assert "failure_probability" in result
    assert "critical_components" in result

    assert 0 <= result["failure_probability"] <= 1
    assert isinstance(result["next_maintenance"], pd.Timestamp)


def test_quality_prediction(ai_engine, test_data):
    """Test quality prediction model."""
    result = ai_engine.predict_quality_metrics(
        test_data["production"], test_data["material"]
    )

    assert "predicted_defect_rate" in result
    assert "confidence_interval" in result
    assert "quality_factors" in result

    assert 0 <= result["predicted_defect_rate"] <= 1


def test_model_training(ai_engine, test_data):
    """Test model training process."""
    # Initial state
    assert not ai_engine.is_trained

    # Train through optimization
    ai_engine.optimize_process_parameters(test_data["production"], test_data["quality"])

    # Check trained state
    assert ai_engine.is_trained
