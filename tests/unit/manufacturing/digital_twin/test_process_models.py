# tests/unit/manufacturing/digital_twin/test_process_models.py

import pytest
import numpy as np
from circman5.manufacturing.digital_twin.simulation.process_models import (
    SiliconPurificationModel,
    WaferProductionModel,
)


def test_silicon_purification_model(process_model_silicon):
    """Test silicon purification process model."""
    # Create test input state
    input_state = {
        "input_amount": 100.0,
        "temperature": 1450.0,  # Optimal temperature
        "energy_consumption": 5000.0,
    }

    # Run model
    output_state = process_model_silicon.simulate_step(input_state)

    # Verify outputs
    assert "output_amount" in output_state
    assert "yield_rate" in output_state
    assert "waste_generated" in output_state
    assert "purity" in output_state

    # Verify reasonable values
    assert 0 < output_state["output_amount"] <= input_state["input_amount"]
    assert 0 <= output_state["yield_rate"] <= 100
    assert output_state["waste_generated"] >= 0
    assert 90 <= output_state["purity"] <= 100


def test_silicon_purification_temperature_effect(process_model_silicon):
    """Test temperature effect on silicon purification."""
    # Optimal temperature
    optimal_state = {
        "input_amount": 100.0,
        "temperature": 1450.0,
        "energy_consumption": 5000.0,
    }

    # Sub-optimal temperature
    suboptimal_state = {
        "input_amount": 100.0,
        "temperature": 1500.0,
        "energy_consumption": 5000.0,
    }

    # Run model for both cases
    optimal_output = process_model_silicon.simulate_step(optimal_state)
    suboptimal_output = process_model_silicon.simulate_step(suboptimal_state)

    # Optimal temperature should give better yield
    assert optimal_output["yield_rate"] >= suboptimal_output["yield_rate"]


def test_wafer_production_model(process_model_wafer):
    """Test wafer production process model."""
    # Create test input state
    input_state = {
        "input_amount": 100.0,
        "cutting_speed": 10.0,  # Optimal speed
        "wire_tension": 25.0,  # Optimal tension
    }

    # Run model
    output_state = process_model_wafer.simulate_step(input_state)

    # Verify outputs
    assert "output_amount" in output_state
    assert "yield_rate" in output_state
    assert "waste_generated" in output_state
    assert "thickness_uniformity" in output_state
    assert "kerf_loss" in output_state

    # Verify reasonable values
    assert 0 < output_state["output_amount"] <= input_state["input_amount"]
    assert 0 <= output_state["yield_rate"] <= 100
    assert output_state["waste_generated"] >= 0
    assert 90 <= output_state["thickness_uniformity"] <= 100
    assert output_state["kerf_loss"] > 0


def test_wafer_production_parameter_effects(process_model_wafer):
    """Test parameter effects on wafer production."""
    # Optimal parameters
    optimal_state = {"input_amount": 100.0, "cutting_speed": 10.0, "wire_tension": 25.0}

    # Sub-optimal parameters
    suboptimal_state = {
        "input_amount": 100.0,
        "cutting_speed": 15.0,
        "wire_tension": 15.0,
    }

    # Run model for both cases
    optimal_output = process_model_wafer.simulate_step(optimal_state)
    suboptimal_output = process_model_wafer.simulate_step(suboptimal_state)

    # Optimal parameters should give better yield and uniformity
    assert optimal_output["yield_rate"] > suboptimal_output["yield_rate"]
    assert (
        optimal_output["thickness_uniformity"]
        > suboptimal_output["thickness_uniformity"]
    )
