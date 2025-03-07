# tests/integration/human_interface/conftest.py

import pytest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path

from circman5.manufacturing.human_interface.core.interface_manager import (
    interface_manager,
)
from circman5.manufacturing.human_interface.adapters.digital_twin_adapter import (
    digital_twin_adapter,
)
from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.core.state_manager import StateManager
from circman5.utils.results_manager import results_manager


@pytest.fixture
def setup_test_environment():
    """Set up a clean test environment with initialized components."""
    # Create test directories
    test_dir = results_manager.get_path("temp")

    # Initialize core components with test configuration
    state_manager = StateManager()
    digital_twin = DigitalTwin()

    # Initialize interface components
    interface_manager.initialize()

    # Return initialized components for testing
    return {
        "interface_manager": interface_manager,
        "digital_twin_adapter": digital_twin_adapter,
        "digital_twin": digital_twin,
        "state_manager": state_manager,
        "test_dir": test_dir,
    }


@pytest.fixture
def sample_manufacturing_state():
    """Provide a sample manufacturing state for testing."""
    return {
        "system_status": "running",
        "timestamp": "2025-02-28T10:15:30",
        "production_line": {
            "status": "running",
            "temperature": 85.2,
            "energy_consumption": 120.5,
            "production_rate": 42.3,
            "defect_rate": 0.8,
        },
        "manufacturing_processes": {
            "material_preparation": {
                "status": "active",
                "metrics": {"yield_rate": 99.2, "throughput": 45.6},
            },
            "cell_assembly": {
                "status": "active",
                "metrics": {"yield_rate": 98.7, "throughput": 43.1},
            },
        },
    }


@pytest.fixture
def mock_digital_twin():
    """Fixture that provides a mocked digital twin."""
    with patch(
        "circman5.manufacturing.digital_twin.core.twin_core.DigitalTwin"
    ) as mock:
        mock_instance = MagicMock()

        # Configure mock state
        mock_instance.get_current_state.return_value = {
            "timestamp": "2025-02-28T12:00:00",
            "system_status": "running",
            "production_line": {
                "status": "active",
                "temperature": 150.5,
                "energy_consumption": 45.2,
                "production_rate": 120.0,
                "defect_rate": 2.3,
            },
            "manufacturing_processes": {
                "process1": {"status": "active", "metrics": {"temperature": 160.0}},
                "process2": {"status": "active", "metrics": {"temperature": 145.2}},
            },
        }

        # Configure mock history
        mock_instance.get_state_history.return_value = [
            {
                "timestamp": "2025-02-28T11:58:00",
                "system_status": "running",
                "production_line": {"status": "active", "production_rate": 118.5},
            },
            {
                "timestamp": "2025-02-28T11:59:00",
                "system_status": "running",
                "production_line": {"status": "active", "production_rate": 119.2},
            },
            {
                "timestamp": "2025-02-28T12:00:00",
                "system_status": "running",
                "production_line": {"status": "active", "production_rate": 120.0},
            },
        ]

        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_event_adapter():
    """Fixture that provides a mocked event adapter."""
    with patch(
        "circman5.manufacturing.human_interface.adapters.event_adapter.EventAdapter"
    ) as mock:
        mock_instance = MagicMock()

        # Configure mock events
        mock_instance.get_recent_events.return_value = [
            {
                "event_id": "evt-001",
                "timestamp": "2025-02-28T11:55:00",
                "category": "system",
                "severity": "info",
                "message": "System started",
                "source": "system",
                "acknowledged": True,
            },
            {
                "event_id": "evt-002",
                "timestamp": "2025-02-28T11:58:00",
                "category": "process",
                "severity": "warning",
                "message": "Temperature high",
                "source": "process1",
                "acknowledged": False,
            },
        ]

        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def initialized_interface():
    """Fixture that provides an initialized interface environment."""
    from circman5.manufacturing.human_interface.core.interface_manager import (
        InterfaceManager,
        interface_manager,
    )
    from circman5.manufacturing.human_interface.core.interface_state import (
        InterfaceState,
        interface_state,
    )
    from circman5.manufacturing.human_interface.core.dashboard_manager import (
        DashboardManager,
        dashboard_manager,
    )

    # Create mock versions of the managers for testing
    mock_interface_manager = MagicMock(spec=InterfaceManager)
    mock_interface_state = MagicMock(spec=InterfaceState)
    mock_dashboard_manager = MagicMock(spec=DashboardManager)

    # Set up necessary methods and properties
    mock_interface_manager.components = {}
    mock_interface_manager.event_handlers = {}
    mock_interface_manager.register_component = (
        lambda id, comp: mock_interface_manager.components.update({id: comp})
    )
    mock_interface_manager.register_event_handler = (
        lambda event, handler: mock_interface_manager.event_handlers.setdefault(
            event, []
        ).append(handler)
    )
    mock_interface_manager.trigger_event = lambda event, data: [
        h(data) for h in mock_interface_manager.event_handlers.get(event, [])
    ]

    mock_interface_state.active_view = "main_dashboard"
    mock_interface_state.expanded_panels = set(["status", "kpi"])
    mock_interface_state.process_control_mode = "monitor"
    mock_interface_state._save_state = MagicMock()
    mock_interface_state._load_state = MagicMock()

    mock_dashboard_manager.layouts = {}
    mock_dashboard_manager.components = {}

    # Patch the global instances
    with patch(
        "circman5.manufacturing.human_interface.core.interface_manager.interface_manager",
        mock_interface_manager,
    ):
        with patch(
            "circman5.manufacturing.human_interface.core.interface_state.interface_state",
            mock_interface_state,
        ):
            with patch(
                "circman5.manufacturing.human_interface.core.dashboard_manager.dashboard_manager",
                mock_dashboard_manager,
            ):
                # Mock initialization
                with patch(
                    "circman5.manufacturing.human_interface.adapters.event_adapter.EventAdapter"
                ):
                    yield mock_interface_manager
