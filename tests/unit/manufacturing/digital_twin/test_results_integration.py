# tests/unit/manufacturing/digital_twin/test_results_integration.py

"""
Tests for integration with results_manager in Digital Twin components.

This module tests that the Digital Twin components correctly use results_manager
for saving and loading data.
"""

import pytest
import json
import time
from pathlib import Path

from circman5.manufacturing.digital_twin.core import DigitalTwin, StateManager
from circman5.utils.results_manager import results_manager


class TestResultsIntegration:
    """Tests for results_manager integration."""

    def test_digital_twin_directory_exists(self):
        """Test that the digital_twin directory is created by results_manager."""
        # Get digital_twin directory path
        dt_dir = results_manager.get_path("digital_twin")

        # Check that directory exists or can be created
        assert dt_dir.exists() or dt_dir.mkdir(parents=True, exist_ok=True)
        assert dt_dir.is_dir()

    def test_state_saving(self, initialized_twin, temp_json_file):
        """Test saving state with results_manager."""
        # First, check direct file saving
        save_result = initialized_twin.save_state(temp_json_file)
        assert save_result is True
        assert Path(temp_json_file).exists()

        # Check file content
        with open(temp_json_file, "r") as f:
            saved_data = json.load(f)
        assert "system_status" in saved_data

        # Now test saving with results_manager
        save_result = initialized_twin.save_state()  # No path means use results_manager
        assert save_result is True

        # Verify that digital_twin directory exists in results
        dt_dir = results_manager.get_path("digital_twin")
        assert dt_dir.exists()

        # Check that at least one file exists in the directory
        # We can't check exact filename due to timestamp
        files = list(dt_dir.glob("digital_twin_state_*.json"))
        assert len(files) > 0

    def test_state_manager_export(self, state_manager, sample_state):
        """Test exporting state with results_manager."""
        # Set a test state
        state_manager.set_state(sample_state)

        # Export state using results_manager
        export_result = (
            state_manager.export_state()
        )  # No path means use results_manager
        assert export_result is True

        # Verify digital_twin directory exists
        dt_dir = results_manager.get_path("digital_twin")
        assert dt_dir.exists()

        # Check that at least one state file exists
        files = list(dt_dir.glob("state_*.json"))
        assert len(files) > 0
