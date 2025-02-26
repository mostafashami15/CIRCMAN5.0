# tests/unit/manufacturing/digital_twin/test_state_manager.py
"""
Unit tests for StateManager component.

This module contains tests for the state management functionality of the Digital Twin system.
"""

import pytest
import json
from pathlib import Path

from circman5.manufacturing.digital_twin.core.state_manager import StateManager


class TestStateManager:
    """Tests for the StateManager class."""

    def test_init(self):
        """Test initialization of StateManager."""
        state_manager = StateManager(history_length=10)
        assert state_manager.history_length == 10
        assert state_manager.current_state == {}
        assert len(state_manager.state_history) == 0

    def test_set_state(self, sample_state):
        """Test setting state and history tracking."""
        state_manager = StateManager(history_length=3)

        # Set initial state
        state_manager.set_state(sample_state)
        assert state_manager.current_state["system_status"] == "running"
        assert len(state_manager.state_history) == 0

        # Set second state
        new_state = sample_state.copy()
        new_state["system_status"] = "paused"
        state_manager.set_state(new_state)
        assert state_manager.current_state["system_status"] == "paused"
        assert len(state_manager.state_history) == 1
        assert state_manager.state_history[0]["system_status"] == "running"

        # Set more states to test history length limit
        for status in ["maintenance", "error", "shutdown"]:
            modified_state = sample_state.copy()
            modified_state["system_status"] = status
            state_manager.set_state(modified_state)

        assert state_manager.current_state["system_status"] == "shutdown"
        assert len(state_manager.state_history) == 3
        # The history order is: paused, maintenance, error (as most recent states are kept)
        assert state_manager.state_history[0]["system_status"] == "paused"
        assert state_manager.state_history[1]["system_status"] == "maintenance"
        assert state_manager.state_history[2]["system_status"] == "error"

    def test_update_state(self, sample_state):
        """Test updating parts of the state."""
        state_manager = StateManager()

        # Set initial state
        state_manager.set_state(sample_state)

        # Update part of the state
        updates = {
            "production_line": {
                "status": "maintenance",
                "maintenance_task": "calibration",
            },
            "new_section": {"value": 42},
        }
        state_manager.update_state(updates)

        # Check that updates were applied correctly
        current = state_manager.get_current_state()
        assert current["production_line"]["status"] == "maintenance"
        assert (
            current["production_line"]["temperature"] == 22.5
        )  # Original value preserved
        assert (
            current["production_line"]["maintenance_task"] == "calibration"
        )  # New field added
        assert current["new_section"]["value"] == 42  # New section added

    def test_get_history(self, sample_state):
        """Test retrieving history with and without limits."""
        state_manager = StateManager(history_length=5)

        # Add some states
        for i in range(5):
            modified_state = sample_state.copy()
            modified_state["system_status"] = f"status_{i}"
            state_manager.set_state(modified_state)

        # Get all history
        history = state_manager.get_history()
        assert len(history) == 4  # 5 states, but first one isn't in history

        # Get limited history
        limited = state_manager.get_history(limit=2)
        assert len(limited) == 2
        assert limited[0]["system_status"] == "status_2"
        assert limited[1]["system_status"] == "status_3"

    def test_validate_state(self, sample_state):
        """Test state validation."""
        state_manager = StateManager()

        # Test valid state
        is_valid, message = state_manager.validate_state(sample_state)
        assert is_valid is True

        # Test invalid state (not a dictionary)
        is_valid, message = state_manager.validate_state("not a dictionary")
        assert is_valid is False
        assert "must be a dictionary" in message

        # Test invalid timestamp
        invalid_timestamp = sample_state.copy()
        invalid_timestamp["timestamp"] = "invalid-format"
        is_valid, message = state_manager.validate_state(invalid_timestamp)
        assert is_valid is False
        assert "Invalid timestamp format" in message

    def test_export_import_state(self, sample_state, temp_json_file):
        """Test exporting and importing state."""
        state_manager = StateManager()
        state_manager.set_state(sample_state)

        # Export state
        result = state_manager.export_state(temp_json_file)
        assert result is True

        # Verify file contents
        with open(temp_json_file, "r") as f:
            saved_data = json.load(f)
        assert saved_data["system_status"] == sample_state["system_status"]

        # Create new state manager and import
        new_manager = StateManager()
        import_result = new_manager.import_state(temp_json_file)
        assert import_result is True

        # Check imported state
        imported = new_manager.get_current_state()
        assert imported["system_status"] == sample_state["system_status"]
        assert (
            imported["production_line"]["temperature"]
            == sample_state["production_line"]["temperature"]
        )

    def test_clear_history(self, sample_state):
        """Test clearing state history."""
        state_manager = StateManager()

        # Add several states
        for i in range(5):
            modified_state = sample_state.copy()
            modified_state["system_status"] = f"status_{i}"
            state_manager.set_state(modified_state)

        assert len(state_manager.state_history) == 4

        # Clear history
        state_manager.clear_history()
        assert len(state_manager.state_history) == 0

        # Current state should remain unchanged
        assert state_manager.current_state["system_status"] == "status_4"

    def test_get_state_at_time(self, sample_state):
        """Test retrieving state at a specific time."""
        state_manager = StateManager()

        # Add states with different timestamps
        timestamps = []
        for i in range(3):
            modified_state = sample_state.copy()
            timestamp = f"2025-02-{20+i}T10:00:00"
            modified_state["timestamp"] = timestamp
            modified_state["system_status"] = f"status_{i}"
            state_manager.set_state(modified_state)
            timestamps.append(timestamp)

        # Get state at specific timestamp
        target_state = state_manager.get_state_at_time(timestamps[1])
        assert target_state is not None
        assert target_state["system_status"] == "status_1"

        # Test timestamp not found
        not_found = state_manager.get_state_at_time("2025-01-01T00:00:00")
        assert not_found is None
