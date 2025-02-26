# tests/unit/manufacturing/digital_twin/test_synchronization.py

"""
Unit tests for SynchronizationManager component.

This module contains tests for the synchronization functionality of the Digital Twin system.
"""

import pytest
import time
import threading
from unittest.mock import MagicMock

from circman5.manufacturing.digital_twin.core.synchronization import (
    SynchronizationManager,
    SyncMode,
)


class TestSynchronizationManager:
    """Tests for the SynchronizationManager class."""

    def test_init(self, state_manager):
        """Test initialization of SynchronizationManager."""
        sync_manager = SynchronizationManager(state_manager)
        assert sync_manager.sync_mode == SyncMode.REAL_TIME
        assert sync_manager.is_running is False
        assert sync_manager.sync_interval == 1.0

        # Custom initialization
        custom_sync = SynchronizationManager(
            state_manager, sync_mode=SyncMode.BATCH, sync_interval=5.0
        )
        assert custom_sync.sync_mode == SyncMode.BATCH
        assert custom_sync.sync_interval == 5.0

    def test_register_data_source(self, sync_manager):
        """Test registering data sources."""

        # Register a test data source
        def test_source():
            return {"test": "data"}

        sync_manager.register_data_source("test_source", test_source)
        assert "test_source" in sync_manager.data_sources

    def test_synchronize_now(self, sync_manager):
        """Test immediate synchronization."""

        # Register a test data source
        def test_source():
            return {"test_value": 123}

        sync_manager.register_data_source("test_source", test_source)

        # Perform synchronization
        result = sync_manager.synchronize_now()
        assert "timestamp" in result
        assert "test_source" in result
        assert result["test_source"]["test_value"] == 123

        # Check that state was updated
        state = sync_manager.state_manager.get_current_state()
        assert "test_source" in state
        assert state["test_source"]["test_value"] == 123

    def test_multiple_data_sources(self, sync_manager):
        """Test synchronization with multiple data sources."""

        # Register multiple test data sources
        def source_a():
            return {"a_value": 123}

        def source_b():
            return {"b_value": "test"}

        sync_manager.register_data_source("source_a", source_a)
        sync_manager.register_data_source("source_b", source_b)

        # Perform synchronization
        result = sync_manager.synchronize_now()

        # Check results
        assert "source_a" in result
        assert "source_b" in result
        assert result["source_a"]["a_value"] == 123
        assert result["source_b"]["b_value"] == "test"

    def test_set_sync_mode(self, sync_manager):
        """Test changing the synchronization mode."""
        # Initial mode is MANUAL (from fixture)
        assert sync_manager.sync_mode == SyncMode.MANUAL

        # Change to BATCH mode
        sync_manager.set_sync_mode(SyncMode.BATCH)
        assert sync_manager.sync_mode == SyncMode.BATCH

        # Change to REAL_TIME mode
        sync_manager.set_sync_mode(SyncMode.REAL_TIME)
        assert sync_manager.sync_mode == SyncMode.REAL_TIME

    def test_set_sync_interval(self, sync_manager):
        """Test setting the synchronization interval."""
        # Store the initial interval value from the fixture
        initial_interval = sync_manager.sync_interval

        # Change interval
        sync_manager.set_sync_interval(2.5)
        assert sync_manager.sync_interval == 2.5

        # Change back to original
        sync_manager.set_sync_interval(initial_interval)
        assert sync_manager.sync_interval == initial_interval

    def test_start_stop_manual_mode(self, state_manager):
        """Test starting and stopping in MANUAL mode."""
        sync_manager = SynchronizationManager(state_manager, sync_mode=SyncMode.MANUAL)

        # Start in manual mode (doesn't start a thread)
        result = sync_manager.start_synchronization()
        assert result is True
        assert sync_manager.is_running is True
        assert sync_manager._sync_thread is None

        # Stop
        stop_result = sync_manager.stop_synchronization()
        assert stop_result is True
        assert sync_manager.is_running is False

    def test_error_handling_in_data_source(self, sync_manager):
        """Test handling errors in data sources."""

        # Register a source that raises an exception
        def error_source():
            raise ValueError("Test error")

        sync_manager.register_data_source("error_source", error_source)

        # Register a valid source
        def valid_source():
            return {"valid": "data"}

        sync_manager.register_data_source("valid_source", valid_source)

        # Synchronization should continue despite error
        result = sync_manager.synchronize_now()
        assert "valid_source" in result
        assert "error_source" not in result

    def test_start_stop_realtime_mode(self, state_manager):
        """Test starting and stopping in REAL_TIME mode."""
        # Use a longer interval to avoid too many synchronizations
        sync_manager = SynchronizationManager(
            state_manager, sync_mode=SyncMode.REAL_TIME, sync_interval=10.0
        )

        # Add a mock for synchronize_now
        sync_manager.synchronize_now = MagicMock(return_value={})

        # Start synchronization
        start_result = sync_manager.start_synchronization()
        assert start_result is True
        assert sync_manager.is_running is True
        assert sync_manager._sync_thread is not None

        # Let it run briefly
        time.sleep(0.1)

        # Stop synchronization
        stop_result = sync_manager.stop_synchronization()
        assert stop_result is True
        assert sync_manager.is_running is False

        # Check that synchronize_now was called at least once
        sync_manager.synchronize_now.assert_called()
