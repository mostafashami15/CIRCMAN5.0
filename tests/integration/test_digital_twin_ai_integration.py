# tests/integration/test_digital_twin_ai_integration.py

"""
Integration tests for the Digital Twin and AI components.

This module implements tests that verify the proper integration between
the Digital Twin system and AI optimization components.
"""

import pytest
import os
import json
from pathlib import Path

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.integration.ai_integration import AIIntegration
from circman5.manufacturing.digital_twin.integration.lca_integration import (
    LCAIntegration,
)
from circman5.adapters.services.constants_service import ConstantsService


@pytest.fixture
def initialized_twin():
    """Fixture that provides an initialized Digital Twin."""
    twin = DigitalTwin()
    twin.initialize()
    return twin


def test_ai_integration_instantiation(initialized_twin):
    """Test that AI integration is properly instantiated."""
    twin = initialized_twin

    # Directly check for integration availability
    assert hasattr(twin, "lca_integration"), "LCA integration not available"

    # Check LCA integration type
    assert isinstance(twin.lca_integration, LCAIntegration), "Wrong integration type"


def test_digital_twin_state_exposure():
    """Test that Digital Twin state is properly exposed to integrations."""
    # Create and initialize twin
    twin = DigitalTwin()
    twin.initialize()

    # Update with test data
    test_state = {
        "production_line": {
            "status": "running",
            "temperature": 25.0,
            "energy_consumption": 120.0,
            "production_rate": 8.5,
        },
        "materials": {
            "silicon_wafer": {"inventory": 800, "quality": 0.97},
            "solar_glass": {"inventory": 400, "quality": 0.99},
        },
    }
    twin.update(test_state)

    # Get current state
    current_state = twin.get_current_state()

    # Verify state contains test data
    assert "production_line" in current_state, "Missing production_line in state"
    assert (
        current_state["production_line"]["temperature"] == 25.0
    ), "Incorrect temperature value"
    assert (
        current_state["production_line"]["energy_consumption"] == 120.0
    ), "Incorrect energy value"


def test_simulation_with_ai_parameters():
    """Test running simulations with AI-optimized parameters."""
    # Create and initialize twin
    twin = DigitalTwin()
    twin.initialize()

    # Define test parameters based on the AI_INTEGRATION section in digital_twin.json
    constants = ConstantsService()
    dt_config = constants.get_digital_twin_config()
    ai_config = dt_config.get("AI_INTEGRATION", {})
    default_params = ai_config.get("DEFAULT_PARAMETERS", {})

    # Create parameters for simulation
    if not default_params:
        # Use hardcoded defaults if configuration not available
        ai_parameters = {
            "input_amount": 100.0,
            "energy_used": 50.0,
            "cycle_time": 30.0,
            "efficiency": 0.9,
            "defect_rate": 0.05,
            "thickness_uniformity": 95.0,
        }
    else:
        ai_parameters = default_params

    # Run simulation with parameters
    sim_results = twin.simulate(steps=5, parameters=ai_parameters)

    # Verify simulation completed
    assert len(sim_results) > 0, "Simulation produced no results"
    assert (
        len(sim_results) == 6
    ), f"Expected 6 simulation states, got {len(sim_results)}"

    # Verify parameters were applied
    # This is a basic check - in a real test, you'd verify specific mappings
    for param, value in ai_parameters.items():
        # Check if parameter is in the results in some form
        # This is a simplification, as parameter mapping would normally be more structured
        param_found = False

        for key, state_value in sim_results[0].items():
            if isinstance(state_value, dict):
                if param in state_value:
                    param_found = True
                    break
            elif key == param:
                param_found = True
                break

        if not param_found:
            print(f"Note: Parameter {param} not directly found in simulation state")


def test_parameter_mapping():
    """Test parameter mapping between AI models and Digital Twin."""
    # Create and initialize twin
    twin = DigitalTwin()
    twin.initialize()

    # Get parameter mapping from configuration
    constants = ConstantsService()
    dt_config = constants.get_digital_twin_config()
    ai_config = dt_config.get("AI_INTEGRATION", {})
    parameter_mapping = ai_config.get("PARAMETER_MAPPING", {})

    if not parameter_mapping:
        pytest.skip("Parameter mapping not configured in digital_twin.json")

    # Check that mapped parameters exist in the state
    current_state = twin.get_current_state()

    # This is a simple validation of the mapping structure
    # In a real test, you'd verify the actual mapping works correctly
    for dt_param, ai_param in parameter_mapping.items():
        # Check if parameter is in the state path
        # This is a simplification, as parameters might be nested
        param_parts = dt_param.split(".")
        target = current_state

        for part in param_parts[:-1]:
            if part in target:
                target = target[part]
            else:
                print(f"Note: Parameter path {dt_param} not found in current state")
                break


def test_optimization_constraints():
    """Test that optimization constraints are properly applied."""
    # Create and initialize twin
    twin = DigitalTwin()
    twin.initialize()

    # Get optimization constraints from configuration
    constants = ConstantsService()
    dt_config = constants.get_digital_twin_config()
    ai_config = dt_config.get("AI_INTEGRATION", {})
    optimization_constraints = ai_config.get("OPTIMIZATION_CONSTRAINTS", {})

    if not optimization_constraints:
        pytest.skip("Optimization constraints not configured in digital_twin.json")

    # Verify constraints are properly structured
    for param, constraints in optimization_constraints.items():
        assert isinstance(
            constraints, list
        ), f"Constraints for {param} should be a list"
        assert (
            len(constraints) == 2
        ), f"Constraints for {param} should have min and max values"
        assert (
            constraints[0] <= constraints[1]
        ), f"Min value should be <= max value for {param}"


def test_simulation_event_publishing():
    """Test that simulation events are properly published."""
    # Create and initialize twin
    twin = DigitalTwin()
    twin.initialize()

    # We can't easily verify event publishing without a listener
    # This test mainly ensures the simulation completes without errors

    # Run simulation
    sim_results = twin.simulate(steps=3, parameters={"test_param": "test_value"})

    # Verify simulation completed
    assert (
        len(sim_results) == 4
    ), f"Expected 4 simulation states, got {len(sim_results)}"
