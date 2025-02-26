# tests/unit/manufacturing/digital_twin/test_twin_core.py

"""
Unit tests for DigitalTwin core component.

This module contains tests for the core Digital Twin functionality.
"""

import pytest
import json
import time
from pathlib import Path

from circman5.manufacturing.digital_twin.core.twin_core import (
    DigitalTwin,
    DigitalTwinConfig,
)


class TestDigitalTwin:
    """Tests for the DigitalTwin class."""

    def test_init(self):
        """Test initialization of DigitalTwin."""
        # Default initialization
        twin = DigitalTwin()
        assert twin.config.name == "SoliTek_DigitalTwin"
        assert twin.is_running is False

        # Custom initialization
        custom_config = DigitalTwinConfig(
            name="TestTwin", update_frequency=0.5, history_length=100
        )
        custom_twin = DigitalTwin(custom_config)
        assert custom_twin.config.name == "TestTwin"
        assert custom_twin.config.update_frequency == 0.5
        assert custom_twin.config.history_length == 100

    def test_initialize(self):
        """Test initialization of the digital twin."""
        twin = DigitalTwin()
        result = twin.initialize()
        assert result is True
        assert twin.is_running is True

        # Check that initial state was set
        state = twin.get_current_state()
        assert "timestamp" in state
        assert "system_status" in state
        assert state["system_status"] == "initialized"
        assert "production_line" in state
        assert "materials" in state

    def test_update(self, initialized_twin):
        """Test updating the digital twin state."""
        # Update with external data
        update_result = initialized_twin.update(
            {"test_data": {"value": 123}, "production_line": {"status": "running"}}
        )
        assert update_result is True

        # Check that update was applied
        state = initialized_twin.get_current_state()
        assert "test_data" in state
        assert state["test_data"]["value"] == 123
        assert state["production_line"]["status"] == "running"

        # Check that timestamp was updated
        assert "timestamp" in state

    def test_simulate(self, initialized_twin):
        """Test simulation capability."""
        # Set initial state for better simulation testing
        initialized_twin.state_manager.set_state(
            {
                "timestamp": "2025-02-25T10:00:00",
                "production_line": {
                    "status": "running",
                    "temperature": 25.0,
                    "energy_consumption": 100.0,
                },
            }
        )

        # Run simulation for 5 steps
        sim_results = initialized_twin.simulate(steps=5)
        # Check that simulation generated the expected number of states
        assert (
            len(sim_results) == 6
        ), f"Expected 6 states in simulation result, got {len(sim_results)}"

        # Check that timestamps are different, which confirms simulation happened
        timestamps = [s.get("timestamp") for s in sim_results if "timestamp" in s]
        assert len(set(timestamps)) == len(
            timestamps
        ), "All timestamps in simulation should be unique"

        # Check that at least one of the values changes
        if "production_line" in sim_results[-1]:
            # Don't check specific values but ensure the simulation is doing something
            assert (
                sim_results[-1]["production_line"]["temperature"]
                >= sim_results[0]["production_line"]["temperature"]
                and sim_results[-1]["production_line"]["energy_consumption"]
                >= sim_results[0]["production_line"]["energy_consumption"]
            )

            # Check that at least one value changed
            assert (
                sim_results[-1]["production_line"]["temperature"]
                > sim_results[0]["production_line"]["temperature"]
                or sim_results[-1]["production_line"]["energy_consumption"]
                > sim_results[0]["production_line"]["energy_consumption"]
            )

    def test_simulate_with_parameters(self, initialized_twin):
        """Test simulation with modified parameters."""
        # Run simulation with modified parameters
        sim_results = initialized_twin.simulate(
            steps=3, parameters={"production_line": {"status": "maintenance"}}
        )

        # Check that parameters were used
        assert sim_results[0]["production_line"]["status"] == "maintenance"

    def test_save_load_state(self, initialized_twin, temp_json_file):
        """Test saving and loading state."""
        # Update with test data
        initialized_twin.update({"test_key": "test_value"})

        # Save state
        save_result = initialized_twin.save_state(temp_json_file)
        assert save_result is True

        # Verify file contents
        with open(temp_json_file, "r") as f:
            saved_data = json.load(f)
        assert "test_key" in saved_data
        assert saved_data["test_key"] == "test_value"

        # Create new twin and load state
        new_twin = DigitalTwin()
        load_result = new_twin.load_state(temp_json_file)
        assert load_result is True

        # Check that state was loaded
        loaded_state = new_twin.get_current_state()
        assert "test_key" in loaded_state
        assert loaded_state["test_key"] == "test_value"

    def test_get_state_history(self, initialized_twin):
        """Test retrieving state history."""
        # Create a few states
        for i in range(3):
            initialized_twin.update({"counter": i})

        # Get history
        history = initialized_twin.get_state_history()
        assert len(history) > 0

        # Get limited history
        limited = initialized_twin.get_state_history(limit=1)
        assert len(limited) == 1
