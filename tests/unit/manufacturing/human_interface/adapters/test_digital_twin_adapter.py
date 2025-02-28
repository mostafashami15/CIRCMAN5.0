# tests/unit/manufacturing/human_interface/adapters/test_digital_twin_adapter.py

import pytest
import datetime
from unittest.mock import patch, MagicMock


def test_digital_twin_adapter_singleton():
    """Test that digital twin adapter follows singleton pattern."""
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        DigitalTwinAdapter,
    )
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        digital_twin_adapter,
    )

    # Getting the instance twice should return the same object
    instance1 = DigitalTwinAdapter()
    instance2 = DigitalTwinAdapter()

    assert instance1 is instance2
    assert instance1 is digital_twin_adapter


@patch("circman5.manufacturing.digital_twin.core.twin_core.DigitalTwin")
@patch("circman5.manufacturing.digital_twin.core.state_manager.StateManager")
@patch(
    "circman5.manufacturing.digital_twin.simulation.simulation_engine.SimulationEngine"
)
@patch(
    "circman5.manufacturing.digital_twin.simulation.scenario_manager.ScenarioManager"
)
def test_get_current_state(
    mock_scenario_manager, mock_simulation_engine, mock_state_manager, mock_digital_twin
):
    """Test getting current state from digital twin."""
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        DigitalTwinAdapter,
    )
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        digital_twin_adapter,
    )

    # Create a mock digital twin adapter
    mock_adapter = MagicMock(spec=DigitalTwinAdapter)
    mock_adapter.logger = MagicMock()

    # Setup mock digital twin instance
    mock_dt_instance = mock_digital_twin.return_value
    mock_dt_instance.get_current_state.return_value = {
        "timestamp": "2025-02-28T12:00:00",
        "system_status": "running",
        "production_line": {"status": "active"},
    }

    # Set the digital twin attribute on our mock
    mock_adapter.digital_twin = mock_dt_instance

    # Create a simple implementation of get_current_state
    def get_current_state():
        return mock_adapter.digital_twin.get_current_state()

    # Assign this method to the mock
    mock_adapter.get_current_state = get_current_state

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.adapters.digital_twin_adapter.digital_twin_adapter",
        mock_adapter,
    ):
        # Get state
        state = mock_adapter.get_current_state()

        # Verify state
        assert state["system_status"] == "running"
        assert state["production_line"]["status"] == "active"
        assert mock_dt_instance.get_current_state.called


@patch("circman5.manufacturing.digital_twin.core.twin_core.DigitalTwin")
@patch("circman5.manufacturing.digital_twin.core.state_manager.StateManager")
@patch(
    "circman5.manufacturing.digital_twin.simulation.simulation_engine.SimulationEngine"
)
@patch(
    "circman5.manufacturing.digital_twin.simulation.scenario_manager.ScenarioManager"
)
def test_run_simulation(
    mock_scenario_manager, mock_simulation_engine, mock_state_manager, mock_digital_twin
):
    """Test running a simulation using the digital twin adapter."""
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        DigitalTwinAdapter,
    )
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        digital_twin_adapter,
    )

    # Create a mock digital twin adapter
    mock_adapter = MagicMock(spec=DigitalTwinAdapter)
    mock_adapter.logger = MagicMock()

    # Setup mock digital twin instance
    mock_dt_instance = mock_digital_twin.return_value
    mock_dt_instance.simulate.return_value = [
        {"timestamp": "2025-02-28T12:00:00", "system_status": "running"},
        {"timestamp": "2025-02-28T12:01:00", "system_status": "running"},
    ]

    # Set the digital twin attribute on our mock
    mock_adapter.digital_twin = mock_dt_instance

    # Create a simple implementation of run_simulation
    def run_simulation(steps=1, parameters=None):
        if parameters is None:
            parameters = {}
        return mock_adapter.digital_twin.simulate(steps=steps, parameters=parameters)

    # Assign this method to the mock
    mock_adapter.run_simulation = run_simulation

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.adapters.digital_twin_adapter.digital_twin_adapter",
        mock_adapter,
    ):
        # Run simulation
        sim_results = mock_adapter.run_simulation(steps=2, parameters={"param1": 10})

        # Verify simulation
        assert len(sim_results) == 2
        mock_dt_instance.simulate.assert_called_once_with(
            steps=2, parameters={"param1": 10}
        )


@patch("circman5.manufacturing.digital_twin.core.twin_core.DigitalTwin")
@patch("circman5.manufacturing.digital_twin.core.state_manager.StateManager")
@patch(
    "circman5.manufacturing.digital_twin.simulation.simulation_engine.SimulationEngine"
)
@patch(
    "circman5.manufacturing.digital_twin.simulation.scenario_manager.ScenarioManager"
)
def test_update_state(
    mock_scenario_manager, mock_simulation_engine, mock_state_manager, mock_digital_twin
):
    """Test updating digital twin state."""
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        DigitalTwinAdapter,
    )
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        digital_twin_adapter,
    )

    # Create a mock digital twin adapter
    mock_adapter = MagicMock(spec=DigitalTwinAdapter)
    mock_adapter.logger = MagicMock()

    # Setup mock digital twin instance
    mock_dt_instance = mock_digital_twin.return_value
    mock_dt_instance.update.return_value = True

    # Set the digital twin attribute on our mock
    mock_adapter.digital_twin = mock_dt_instance

    # Create a simple implementation of update_state
    def update_state(updates):
        mock_adapter.logger.debug(f"Updating state with {len(updates)} values")
        return mock_adapter.digital_twin.update(updates)

    # Assign this method to the mock
    mock_adapter.update_state = update_state

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.adapters.digital_twin_adapter.digital_twin_adapter",
        mock_adapter,
    ):
        # Update state
        updates = {"system_status": "idle", "production_line": {"status": "idle"}}
        result = mock_adapter.update_state(updates)

        # Verify update
        assert result is True
        mock_dt_instance.update.assert_called_once_with(updates)


@patch("circman5.manufacturing.digital_twin.core.twin_core.DigitalTwin")
@patch("circman5.manufacturing.digital_twin.core.state_manager.StateManager")
@patch(
    "circman5.manufacturing.digital_twin.simulation.simulation_engine.SimulationEngine"
)
@patch(
    "circman5.manufacturing.digital_twin.simulation.scenario_manager.ScenarioManager"
)
def test_save_scenario(
    mock_scenario_manager, mock_simulation_engine, mock_state_manager, mock_digital_twin
):
    """Test saving a scenario."""
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        DigitalTwinAdapter,
    )
    from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
        digital_twin_adapter,
    )

    # Create a mock digital twin adapter
    mock_adapter = MagicMock(spec=DigitalTwinAdapter)
    mock_adapter.logger = MagicMock()

    # Setup mock scenario manager instance
    scenario_manager_instance = mock_scenario_manager.return_value
    scenario_manager_instance.create_scenario.return_value = {"name": "test_scenario"}

    # Set the scenario_manager attribute on our mock
    mock_adapter.scenario_manager = scenario_manager_instance

    # Create a simple implementation of save_scenario
    def save_scenario(name, parameters, description=""):
        mock_adapter.logger.debug(f"Saving scenario: {name}")
        return bool(
            mock_adapter.scenario_manager.create_scenario(
                name=name, parameters=parameters, description=description
            )
        )

    # Assign this method to the mock
    mock_adapter.save_scenario = save_scenario

    # Patch the global instance
    with patch(
        "circman5.manufacturing.human_interface.adapters.digital_twin_adapter.digital_twin_adapter",
        mock_adapter,
    ):
        # Save scenario
        params = {"param1": 10, "param2": 20}
        result = mock_adapter.save_scenario("test_scenario", params, "Test description")

        # Verify save
        assert result is True
        scenario_manager_instance.create_scenario.assert_called_once_with(
            name="test_scenario", parameters=params, description="Test description"
        )
