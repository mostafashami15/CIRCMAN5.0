# tests/validation/conftest.py

"""
Pytest fixtures for validation tests.
"""

import pytest
import json
from pathlib import Path
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

from circman5.manufacturing.digital_twin.core.twin_core import DigitalTwin
from circman5.manufacturing.digital_twin.event_notification.event_types import Event

# Mock classes for HMI components


class MockInterfaceManager:
    """Mock implementation of InterfaceManager for testing."""

    def __init__(self):
        self.components = {}
        self.initialize_mock_components()

    def initialize_mock_components(self):
        """Initialize mock components for testing."""
        self.components["digital_twin_adapter"] = MockDigitalTwinAdapter()
        self.components["alert_panel"] = MockAlertPanel()
        self.components["process_control"] = MockProcessControl()
        self.components["parameter_control"] = MockParameterControl()
        self.components["scenario_control"] = MockScenarioControl()

    def get_component(self, component_name):
        """Get a component by name."""
        if component_name in self.components:
            return self.components[component_name]
        else:
            raise KeyError(f"Component not found: {component_name}")


class MockDigitalTwinAdapter:
    """Mock implementation of DigitalTwinAdapter for testing."""

    def __init__(self):
        self.digital_twin = None

    def set_digital_twin(self, digital_twin):
        """Set the digital twin reference."""
        self.digital_twin = digital_twin

    def get_current_state(self):
        """Get the current state from the digital twin."""
        if self.digital_twin:
            return self.digital_twin.get_current_state()
        return {}

    def run_simulation(self, steps=5, parameters=None):
        """Run a simulation."""
        if self.digital_twin:
            return self.digital_twin.simulate(steps=steps, parameters=parameters)
        return []


class MockAlertPanel:
    """Mock implementation of AlertPanel for testing."""

    def __init__(self):
        self.alerts = []

    def add_alert(self, alert):
        """Add an alert to the panel."""
        self.alerts.append(alert)

    def get_filtered_alerts(self, filters):
        """Get alerts filtered by criteria."""
        # In a real implementation, this would apply filters
        return self.alerts.copy()


class MockProcessControl:
    """Mock implementation of ProcessControl for testing."""

    def __init__(self):
        self.digital_twin = None

    def set_digital_twin(self, digital_twin):
        """Set the digital twin reference."""
        self.digital_twin = digital_twin

    def start_process(self):
        """Start the manufacturing process."""
        if self.digital_twin:
            self.digital_twin.update(
                {"system_status": "running", "production_line": {"status": "running"}}
            )
            return {"success": True}
        return {"success": False, "error": "Digital Twin not available"}

    def stop_process(self):
        """Stop the manufacturing process."""
        if self.digital_twin:
            self.digital_twin.update(
                {"system_status": "idle", "production_line": {"status": "idle"}}
            )
            return {"success": True}
        return {"success": False, "error": "Digital Twin not available"}


class MockParameterControl:
    """Mock implementation of ParameterControl for testing."""

    def __init__(self):
        self.digital_twin = None

    def set_digital_twin(self, digital_twin):
        """Set the digital twin reference."""
        self.digital_twin = digital_twin

    def update_parameter(self, path, value):
        """Update a parameter in the digital twin state."""
        if not self.digital_twin:
            return {"success": False, "error": "Digital Twin not available"}

        # Parse path and update state
        path_parts = path.split(".")
        update_data = {}
        current = update_data

        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1:
                current[part] = value
            else:
                current[part] = {}
                current = current[part]

        self.digital_twin.update(update_data)
        return {"success": True}


class MockScenarioControl:
    """Mock implementation of ScenarioControl for testing."""

    def __init__(self):
        self.digital_twin = None

    def set_digital_twin(self, digital_twin):
        """Set the digital twin reference."""
        self.digital_twin = digital_twin

    def run_scenario(self, scenario):
        """Run a simulation scenario."""
        if not self.digital_twin:
            return {"success": False, "error": "Digital Twin not available"}

        steps = scenario.get("steps", 5)
        parameters = scenario.get("parameters", {})

        simulation_results = self.digital_twin.simulate(
            steps=steps, parameters=parameters
        )
        return {"success": True, "results": simulation_results}


class MockEventAdapter:
    """Mock implementation of EventAdapter for testing."""

    def __init__(self):
        self.callbacks = []
        self.events = []

    def register_callback(self, callback):
        """Register a callback for events."""
        self.callbacks.append(callback)

    def handle_event(self, event):
        """Handle an event."""
        self.events.append(event)
        for callback in self.callbacks:
            callback(event)


@pytest.fixture
def setup_test_environment():
    """
    Set up a test environment with Digital Twin and mocked Interface Manager.

    Returns:
        dict: Dictionary containing test components
    """
    # Create Digital Twin
    digital_twin = DigitalTwin()
    digital_twin.initialize()

    # Create Interface Manager with mock components
    interface_manager = MockInterfaceManager()

    # Connect mock components to Digital Twin
    digital_twin_adapter = interface_manager.get_component("digital_twin_adapter")
    digital_twin_adapter.set_digital_twin(digital_twin)

    process_control = interface_manager.get_component("process_control")
    process_control.set_digital_twin(digital_twin)

    parameter_control = interface_manager.get_component("parameter_control")
    parameter_control.set_digital_twin(digital_twin)

    scenario_control = interface_manager.get_component("scenario_control")
    scenario_control.set_digital_twin(digital_twin)

    # Create mock event adapter and add to interface manager
    event_adapter = MockEventAdapter()
    interface_manager.components["event_adapter"] = event_adapter

    # Create test environment dictionary
    env = {"digital_twin": digital_twin, "interface_manager": interface_manager}

    return env
