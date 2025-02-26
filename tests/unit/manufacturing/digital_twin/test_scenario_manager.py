# tests/unit/manufacturing/digital_twin/test_scenario_manager.py

import pytest
import tempfile
from pathlib import Path
import json
from circman5.manufacturing.digital_twin.simulation.scenario_manager import (
    SimulationScenario,
    ScenarioManager,
)


def test_scenario_creation(scenario_manager):
    """Test creating a simulation scenario."""
    # Create test parameters
    params = {"temperature": 25.0, "production_rate": 10.0}

    # Create scenario
    scenario = scenario_manager.create_scenario(
        name="Test Scenario", parameters=params, description="A test scenario"
    )

    # Verify scenario
    assert scenario.name == "Test Scenario"
    assert scenario.description == "A test scenario"
    assert scenario.parameters == params
    assert scenario.name in scenario_manager.scenarios


def test_scenario_results(scenario_manager):
    """Test setting and calculating scenario results."""
    # Create scenario
    scenario = scenario_manager.create_scenario(
        name="Result Test", parameters={"test": 123}
    )

    # Set results
    results = [
        {
            "timestamp": "2025-02-15T12:00:00",
            "production_line": {"production_rate": 10.0, "energy_consumption": 100.0},
            "materials": {"silicon_wafer": {"inventory": 500, "quality": 0.95}},
        }
    ]
    scenario.set_results(results)

    # Calculate metrics
    metrics = scenario.calculate_metrics()

    # Verify metrics
    assert "production_rate" in metrics
    assert metrics["production_rate"] == 10.0
    assert "energy_efficiency" in metrics
    assert metrics["energy_efficiency"] == 0.1  # 10.0 / 100.0


def test_scenario_comparison(scenario_manager):
    """Test comparing multiple scenarios."""
    # Create first scenario
    scenario1 = scenario_manager.create_scenario(
        name="Scenario1", parameters={"temp": 20.0}
    )
    scenario1.set_results(
        [{"production_line": {"production_rate": 10.0, "energy_consumption": 100.0}}]
    )
    scenario1.calculate_metrics()

    # Create second scenario
    scenario2 = scenario_manager.create_scenario(
        name="Scenario2", parameters={"temp": 25.0}
    )
    scenario2.set_results(
        [{"production_line": {"production_rate": 12.0, "energy_consumption": 110.0}}]
    )
    scenario2.calculate_metrics()

    # Compare scenarios
    comparison = scenario_manager.compare_scenarios(
        ["Scenario1", "Scenario2"], metrics=["production_rate", "energy_efficiency"]
    )

    # Verify comparison
    assert "Scenario1" in comparison
    assert "Scenario2" in comparison
    assert "production_rate" in comparison["Scenario1"]
    assert comparison["Scenario1"]["production_rate"] == 10.0
    assert comparison["Scenario2"]["production_rate"] == 12.0


def test_scenario_save_load(scenario_manager):
    """Test saving and loading scenarios."""
    # Create a scenario
    scenario_manager.create_scenario(
        name="SaveTest", parameters={"test": 456}, description="Testing save/load"
    )

    # Create temporary file for saving
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Save scenarios
        result = scenario_manager.save_scenarios(temp_path)
        assert result is True

        # Create new manager for loading
        new_manager = ScenarioManager()

        # Load scenarios
        result = new_manager.load_scenarios(temp_path)
        assert result is True

        # Verify loaded scenario
        assert "SaveTest" in new_manager.scenarios
        loaded = new_manager.scenarios["SaveTest"]
        assert loaded.name == "SaveTest"
        assert loaded.description == "Testing save/load"
        assert loaded.parameters == {"test": 456}

    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()
